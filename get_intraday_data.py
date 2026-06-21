import os
import requests
import pandas as pd
import time

api_key = os.environ.get("POLYGON_API_KEY")

if not api_key:
    raise ValueError("POLYGON_API_KEY not found - check that the Codespaces secret is set correctly")

ticker = "SPY"
multiplier = 5  # 5-minute bars
timespan = "minute"
start_date = "2023-01-01"
end_date = "2024-12-31"

url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

all_results = []
next_url = url

while next_url:
    if next_url == url:
        response = requests.get(next_url, params=params)
    else:
        # next_url from polygon doesn't include the key, so add it explicitly
        response = requests.get(next_url, params={"apiKey": api_key})

    data = response.json()

    if "results" not in data:
        print("Error or no data:", data)
        break

    all_results.extend(data["results"])
    next_url = data.get("next_url")

    if next_url:
        time.sleep(13)

df = pd.DataFrame(all_results)
df['datetime'] = pd.to_datetime(df['t'], unit='ms')
df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

df.to_csv("spy_intraday.csv", index=False)
print(f"Saved {len(df)} rows to spy_intraday.csv")
print(df.head())