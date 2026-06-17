import yfinance as sp
import pandas as pd

ticker = "SPY"
data = sp.download(ticker, start="2010-01-01", end="2024-01-01")

data.to_csv("spy_data.csv")
print("Saved data. Rows:", len(data))