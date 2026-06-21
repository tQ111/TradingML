import pandas as pd
import numpy as np

df = pd.read_csv("spy_intraday.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# RTH only: 9:30am - 4:00pm ET
df = df[(df['datetime'].dt.time >= pd.Timestamp('09:30').time()) &
        (df['datetime'].dt.time <= pd.Timestamp('16:00').time())]
df = df.reset_index(drop=True)

features = pd.DataFrame()
features['datetime'] = df['datetime']

# time of day — how far into the session are we (0.0 = open, 1.0 = close)
market_open = pd.Timestamp('09:30').time()
market_close = pd.Timestamp('16:00').time()
session_minutes = 390
def minutes_since_open(t):
    return (t.hour * 60 + t.minute) - (9 * 60 + 30)
features['time_of_day'] = df['datetime'].apply(lambda x: minutes_since_open(x.time()) / session_minutes).clip(0, 1)

# intraday returns
features['return_5b']  = df['Close'].pct_change(5)
features['return_10b'] = df['Close'].pct_change(10)
features['return_20b'] = df['Close'].pct_change(20)

# EMAs (matching his 9/21 settings)
ema9  = df['Close'].ewm(span=9).mean()
ema21 = df['Close'].ewm(span=21).mean()
features['ema_diff'] = (ema9 - ema21) / df['Close']
features['ema9_slope']  = ema9.diff(3) / df['Close']
features['ema21_slope'] = ema21.diff(3) / df['Close']
features['price_vs_ema9']  = (df['Close'] - ema9)  / df['Close']
features['price_vs_ema21'] = (df['Close'] - ema21) / df['Close']

# ATR (matching his 10-bar setting)
high_low   = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift()).abs()
low_close  = (df['Low']  - df['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr10 = tr.rolling(10).mean()
features['atr10'] = atr10 / df['Close']
features['atr_ratio'] = atr10 / tr.rolling(20).mean()

# VWAP — resets each session
df['date'] = df['datetime'].dt.date
df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
df['cumvol']   = df.groupby('date')['Volume'].cumsum()
df['cumtpvol'] = df.groupby('date').apply(lambda x: (x['tp'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
df['vwap']     = df['cumtpvol'] / df['cumvol']
features['price_vs_vwap'] = (df['Close'] - df['vwap']) / df['Close']

# volatility
features['volatility_10b'] = df['Close'].pct_change().rolling(10).std()
features['volatility_20b'] = df['Close'].pct_change().rolling(20).std()

# volume ratio
features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

# candle direction
features['candle_dir'] = (df['Close'] - df['Open']) / df['Open']

# high low range
features['high_low_range'] = (df['High'] - df['Low']) / df['Close']

# RSI 14
delta = df['Close'].diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = -delta.where(delta < 0, 0).rolling(14).mean()
rs    = gain / loss
features['rsi_14'] = 100 - (100 / (1 + rs))

# UT Bot direction proxy (ATR trailing stop direction)
k   = 1.5
atr = atr10
src = (df['High'] + df['Low']) / 2
ts  = pd.Series(np.nan, index=df.index)
ps  = pd.Series(0, index=df.index)
for i in range(1, len(df)):
    nl   = k * atr.iloc[i]
    prev = ts.iloc[i-1] if not np.isnan(ts.iloc[i-1]) else src.iloc[i]
    s    = src.iloc[i]
    s1   = src.iloc[i-1]
    if s > prev and s1 > prev:
        ts.iloc[i] = max(prev, s - nl)
    elif s < prev and s1 < prev:
        ts.iloc[i] = min(prev, s + nl)
    elif s > prev:
        ts.iloc[i] = s - nl
    else:
        ts.iloc[i] = s + nl
    if s1 < ts.iloc[i-1] and s > ts.iloc[i]:
        ps.iloc[i] = 1
    elif s1 > ts.iloc[i-1] and s < ts.iloc[i]:
        ps.iloc[i] = -1
    else:
        ps.iloc[i] = ps.iloc[i-1]
features['ut_direction'] = ps

# ADX proxy
features['adx_proxy'] = tr.rolling(14).mean() / df['Close'].rolling(14).std()

features = features.dropna()
features.to_csv("features_intraday.csv", index=False)
print(f"Saved {len(features)} rows to features_intraday.csv")
print(features.head())