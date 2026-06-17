import pandas as pd

data = pd.read_csv("spy_data.csv")

features = pd.DataFrame()
features['date'] = data['Date']

# return over last N days
features['return_5d'] = data['Close'].pct_change(5)
features['return_10d'] = data['Close'].pct_change(10)
features['return_20d'] = data['Close'].pct_change(20)

# moving averages, expressed as % distance from price
features['dist_from_ma20'] = (data['Close'] - data['Close'].rolling(20).mean()) / data['Close'].rolling(20).mean()
features['dist_from_ma50'] = (data['Close'] - data['Close'].rolling(50).mean()) / data['Close'].rolling(50).mean()

# volatility (rolling std of daily returns)
features['volatility_10d'] = data['Close'].pct_change().rolling(10).std()

# volume relative to its own recent average
features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

# consecutive up/down days
direction = (data['Close'].diff() > 0).astype(int)
features['consecutive_up'] = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
features['consecutive_up'] = features['consecutive_up'] * direction  # zero out if currently a down day

features.to_csv("features.csv", index=False)
print("Saved features. Rows:", len(features))
print(features.head(25))