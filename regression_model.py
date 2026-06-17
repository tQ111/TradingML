import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

df = features.merge(labels, on="date")

feature_cols = ['return_5d', 'return_10d', 'return_20d', 'dist_from_ma20', 'dist_from_ma50',
                'volatility_10d', 'volume_ratio', 'consecutive_up', 'rsi_14', 'macd',
                'day_of_week', 'dist_from_ma200', 'high_low_range']
df = df.dropna(subset=feature_cols)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:].copy()

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# train on actual return percentage instead of good/bad category
y_train_long = train_df['long_return_pct']
y_test_long = test_df['long_return_pct']

long_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
long_model.fit(X_train, y_train_long)

preds = long_model.predict(X_test)
mae = mean_absolute_error(y_test_long, preds)
print(f"Long model - Mean Absolute Error: {mae:.4f}")
print(f"Actual return std dev (for reference): {y_test_long.std():.4f}")

# check correlation between predicted and actual - this tells us if predictions have any real relationship to outcomes
correlation = pd.Series(preds).corr(pd.Series(y_test_long.values))
print(f"Correlation between predicted and actual returns: {correlation:.4f}")

# same for short
y_train_short = train_df['short_return_pct']
y_test_short = test_df['short_return_pct']

short_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
short_model.fit(X_train, y_train_short)

preds_short = short_model.predict(X_test)
correlation_short = pd.Series(preds_short).corr(pd.Series(y_test_short.values))
print(f"\nShort model correlation between predicted and actual returns: {correlation_short:.4f}")