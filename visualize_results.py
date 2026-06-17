import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = pd.read_csv("spy_data.csv")
features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

df = features.merge(labels, on="date")
df['long_good'] = (df['long_outcome'] == 'take_profit').astype(int)
df['short_good'] = (df['short_outcome'] == 'take_profit').astype(int)

feature_cols = ['return_5d', 'return_10d', 'return_20d', 'dist_from_ma20', 'dist_from_ma50',
                'volatility_10d', 'volume_ratio', 'consecutive_up', 'rsi_14', 'macd',
                'day_of_week', 'dist_from_ma200', 'high_low_range']
df = df.dropna(subset=feature_cols)

# train on first 80%, we'll visualize predictions on the last 20% (the test set)
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:].copy()

X_train = train_df[feature_cols]

# train long model
y_train_long = train_df['long_good']
scale_long = (y_train_long == 0).sum() / (y_train_long == 1).sum()
long_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_long)
long_model.fit(X_train, y_train_long)

# train short model
y_train_short = train_df['short_good']
scale_short = (y_train_short == 0).sum() / (y_train_short == 1).sum()
short_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_short)
short_model.fit(X_train, y_train_short)

# predict on test set
test_df['long_pred'] = long_model.predict(test_df[feature_cols])
test_df['short_pred'] = short_model.predict(test_df[feature_cols])

# merge with price data so we have OHLC for charting
chart_data = data.merge(
    test_df[['date', 'long_pred', 'long_good', 'long_outcome', 'short_pred', 'short_good', 'short_outcome']],
    left_on='Date', right_on='date'
)

# build candlestick chart
fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Candlestick(
    x=chart_data['Date'],
    open=chart_data['Open'],
    high=chart_data['High'],
    low=chart_data['Low'],
    close=chart_data['Close'],
    name='SPY',
    increasing_line_width=1,
    decreasing_line_width=1,
))

# mark where model predicted "good long"
buy_signals = chart_data[chart_data['long_pred'] == 1]
fig.add_trace(go.Scatter(
    x=buy_signals['Date'],
    y=buy_signals['Low'] * 0.98,
    mode='markers',
    marker=dict(symbol='triangle-up', size=6, color='lime'),
    name='Model says: Long'
))

# mark where model predicted "good short"
sell_signals = chart_data[chart_data['short_pred'] == 1]
fig.add_trace(go.Scatter(
    x=sell_signals['Date'],
    y=sell_signals['High'] * 1.02,
    mode='markers',
    marker=dict(symbol='triangle-down', size=6, color='red'),
    name='Model says: Short'
))

# mark exits for long signals - green if won, gray if lost/timed out
long_wins = chart_data[(chart_data['long_pred'] == 1) & (chart_data['long_outcome'] == 'take_profit')]
long_losses = chart_data[(chart_data['long_pred'] == 1) & (chart_data['long_outcome'] != 'take_profit')]
fig.add_trace(go.Scatter(
    x=long_wins['Date'], y=long_wins['High'] * 1.015,
    mode='markers', marker=dict(symbol='circle', size=6, color='springgreen'),
    name='Long: Won'
))
fig.add_trace(go.Scatter(
    x=long_losses['Date'], y=long_losses['High'] * 1.015,
    mode='markers', marker=dict(symbol='x', size=6, color='gray'),
    name='Long: Lost/Timed Out'
))

fig.update_layout(
    title='SPY Price with Long Signals (Test Period)',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    dragmode='zoom',
    xaxis=dict(rangeslider=dict(visible=False), fixedrange=False),
    yaxis=dict(fixedrange=False),
)
fig.update_layout(hovermode='x unified')
fig.write_html("chart.html", config={'scrollZoom': True})
print("Saved chart.html")