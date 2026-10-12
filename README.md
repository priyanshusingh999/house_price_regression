# House Price Prediction with Linear Regression

## Dataset
California Housing Prices (1990 census, 20,640 districts), the version hosted on Kaggle.
Kaggle itself was unreachable from the build environment, so the identical CSV was
downloaded from a public GitHub mirror. To use your own copy from Kaggle:
`python house_price_regression.py --data path/to/housing.csv`

Features: location (longitude, latitude, `ocean_proximity`), size (total rooms,
bedrooms, population, households), housing age, and median income.
Target: `median_house_value` (US dollars).

## Preprocessing
1. Dropped duplicates (none found).
2. 207 missing `total_bedrooms` values filled with the training-set median.
3. Removed 965 districts whose price is capped at $500,001 (the data records any
   higher price as exactly that value, so it is not a true price).
4. New features: rooms per household, bedrooms per room, people per household.
5. Log-transformed the skewed size columns (rooms, bedrooms, population, households).
6. Standardized numeric columns; one-hot encoded `ocean_proximity`.
7. 80/20 train/test split (random_state=42). All fitting happens inside a
   scikit-learn Pipeline, so no test information leaks into training.

## Results (test set, 3,935 districts)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Predict the average price | $99,865 | $80,135 | 0.00 |
| Linear regression | $59,418 | $44,013 | 0.65 |

Train R² 0.639 and 5-fold cross-validated R² 0.631 (±0.015) match the test R²,
so the model is not overfitting. It is underfitting somewhat, which is typical of a
straight-line model on housing data.

## Reading the model
- Median income is the strongest dependable driver of price.
- Inland districts are cheaper than coastal ones, all else equal.
- The ISLAND coefficient is huge but rests on only 5 districts. Do not trust it.
- Population and households are strongly correlated, so their opposite-signed
  coefficients should be read together, not individually.

## Limitations
- Predicts the median price of a district, not a single house.
- Prices are from 1990. Use the model to learn the method, not to value homes today.
- Errors grow for expensive districts (see `actual_vs_predicted.png`): the line
  under-predicts them. Tree-based models or a log-price target would help.

## Files
- `house_price_regression.py`: full pipeline
- `outputs/metrics.json`, `outputs/coefficients.csv`: numbers
- `outputs/*.png`: predicted vs actual, residuals, coefficients
- `outputs/house_price_model.joblib`: trained model (load with `joblib.load`)
