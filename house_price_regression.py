import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]  # location category
RANDOM_STATE = 42

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_and_clean(path):
    df = pd.read_csv(path)
    report = {"rows_raw": int(len(df))}

    df = df.drop_duplicates()
    report["missing_values"] = {c: int(n) for c, n in df.isna().sum().items() if n}

    capped = df[TARGET] >= 500_001
    report["capped_target_rows_removed"] = int(capped.sum())
    df = df[~capped].reset_index(drop=True)

    report["rows_clean"] = int(len(df))
    return df, report

def engineer_features(df):
    df = df.copy()
    # Totals per district are dominated by district size; per-household ratios
    # describe the typical house, which is what price depends on.
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["people_per_household"] = df["population"] / df["households"]
    # Strongly right-skewed size columns -> log scale so a linear model fits better
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        df[col] = np.log1p(df[col])
    return df


def build_pipeline(numeric_cols):
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, CATEGORICAL)]
    )
    return Pipeline([("prep", pre), ("model", LinearRegression())])


def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_actual_vs_pred(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y_true / 1e3, y_pred / 1e3, s=6, alpha=0.25, color=BLUE, linewidths=0)
    lim = [0, max(y_true.max(), y_pred.max()) / 1e3]
    ax.plot(lim, lim, color=INK, linewidth=1.5, label="Perfect prediction")
    style_axes(ax, "Predicted vs actual price (test set)",
               "Actual price ($ thousands)", "Predicted price ($ thousands)")
    ax.legend(frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(residuals, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(residuals / 1e3, bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=INK, linewidth=1.5)
    style_axes(ax, "Prediction errors (test set)",
               "Actual minus predicted ($ thousands)", "Districts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coefficients(coefs, path):
    coefs = coefs.sort_values()
    colors = [BLUE if v >= 0 else ORANGE for v in coefs]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.barh(coefs.index, coefs / 1e3, color=colors, height=0.65)
    for bar, v in zip(bars, coefs):
        ax.text(bar.get_width() + (1 if v >= 0 else -1), bar.get_y() + bar.get_height() / 2,
                f"{v / 1e3:+.1f}k", va="center", ha="left" if v >= 0 else "right",
                color=MUTED, fontsize=8)
    ax.axvline(0, color=INK, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("What moves the price (blue raises it, orange lowers it)",
                 loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Change in price per +1 standard deviation ($ thousands)", color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/housing.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, clean_report = load_and_clean(args.data)
    df = engineer_features(df)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = build_pipeline(numeric_cols)
    model.fit(X_train, y_train)

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    pred = model.predict(X_test)

    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    results = {
        "data_cleaning": clean_report,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_predict_the_mean": metrics(y_test, baseline.predict(X_test)),
        "linear_regression_test": metrics(y_test, pred),
        "linear_regression_train": metrics(y_train, model.predict(X_train)),
        "cv_r2_5fold_mean": float(cv_r2.mean()),
        "cv_r2_5fold_std": float(cv_r2.std()),
    }

    # Coefficients, in dollars per +1 standard deviation (numeric) or vs. the
    # average of the categories (one-hot columns).
    names = model.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coefs = pd.Series(model.named_steps["model"].coef_, index=names)
    coefs.sort_values(key=abs, ascending=False).to_csv(
        os.path.join(args.out, "coefficients.csv"), header=["coefficient_usd"]
    )

    plot_actual_vs_pred(y_test.values, pred, os.path.join(args.out, "actual_vs_predicted.png"))
    plot_residuals(y_test.values - pred, os.path.join(args.out, "residuals.png"))
    plot_coefficients(coefs, os.path.join(args.out, "coefficients.png"))

    joblib.dump(model, os.path.join(args.out, "house_price_model.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    example = pd.DataFrame([{
        "longitude": -122.25, "latitude": 37.85, "housing_median_age": 30,
        "total_rooms": 2500, "total_bedrooms": 500, "population": 1200,
        "households": 450, "median_income": 5.5, "ocean_proximity": "NEAR BAY",
    }])
    example_pred = model.predict(engineer_features(example))[0]
    results["example_prediction_usd"] = float(example_pred)

    print(json.dumps(results, indent=2))
    print("\nTop coefficients (USD per +1 std dev):")
    print(coefs.sort_values(key=abs, ascending=False).head(8).round(0).to_string())


if __name__ == "__main__":
    main()
