import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

features = pd.read_csv("features_intraday.csv")
labels   = pd.read_csv("labels_intraday.csv")

features['datetime'] = pd.to_datetime(features['datetime'])
labels['datetime']   = pd.to_datetime(labels['datetime'])

df = features.merge(labels[['datetime', 'direction', 'label']], on='datetime', how='inner')

feature_cols = [
    'time_of_day', 'return_5b', 'return_10b', 'return_20b',
    'ema_diff', 'ema9_slope', 'ema21_slope', 'price_vs_ema9',
    'price_vs_ema21', 'atr10', 'atr_ratio', 'price_vs_vwap',
    'volatility_10b', 'volatility_20b', 'volume_ratio',
    'candle_dir', 'high_low_range', 'rsi_14', 'ut_direction', 'adx_proxy'
]

# train long and short models separately
for direction in ['long', 'short']:
    subset = df[df['direction'] == direction].copy()
    subset = subset.sort_values('datetime').reset_index(drop=True)

    X = subset[feature_cols]
    y = subset['label']

    # walk forward split — 80% train, 20% test, no shuffling
    split = int(len(subset) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # handle label imbalance
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {direction.upper()} MODEL ===")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Label balance (test): {y_test.mean():.2%} positive")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 5 features:")
    print(importance.sort_values(ascending=False).head())

    model.save_model(f"model_{direction}_intraday.json")
    print(f"Saved model_{direction}_intraday.json")