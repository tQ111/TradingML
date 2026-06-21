from flask import Flask, request, jsonify
import os
import json
import time
import threading
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
LOG_FILE = "signals_log.jsonl"

# load trained models once at startup
model_long = XGBClassifier()
model_long.load_model("model_long_intraday.json")
model_short = XGBClassifier()
model_short.load_model("model_short_intraday.json")

FEATURE_COLS = [
    'time_of_day', 'return_5b', 'return_10b', 'return_20b',
    'ema_diff', 'ema9_slope', 'ema21_slope', 'price_vs_ema9',
    'price_vs_ema21', 'atr10', 'atr_ratio', 'price_vs_vwap',
    'volatility_10b', 'volatility_20b', 'volume_ratio',
    'candle_dir', 'high_low_range', 'rsi_14', 'ut_direction', 'adx_proxy'
]

# in-memory open positions: {position_id: {direction, entry_time, entry_price}}
open_positions = {}
positions_lock = threading.Lock()


def log_event(record):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(record)


def fetch_recent_bars(minutes_back=200):
    """Pull recent 5-min SPY bars from Polygon, enough history to compute all features."""
    end = datetime.utcnow()
    start = end - timedelta(minutes=minutes_back)
    url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    params = {"adjusted": "true", "sort": "asc", "limit": 100, "apiKey": POLYGON_API_KEY}
    r = requests.get(url, params=params)
    data = r.json()
    if "results" not in data:
        return None
    df = pd.DataFrame(data["results"])
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df.reset_index(drop=True)


def build_live_features(df):
    """Same feature logic as build_features_intraday.py, applied to live data."""
    features = pd.DataFrame()
    features['datetime'] = df['datetime']

    def minutes_since_open(t):
        return (t.hour * 60 + t.minute) - (9 * 60 + 30)
    features['time_of_day'] = df['datetime'].apply(lambda x: minutes_since_open(x.time()) / 390).clip(0, 1)

    features['return_5b']  = df['Close'].pct_change(5)
    features['return_10b'] = df['Close'].pct_change(10)
    features['return_20b'] = df['Close'].pct_change(20)

    ema9  = df['Close'].ewm(span=9).mean()
    ema21 = df['Close'].ewm(span=21).mean()
    features['ema_diff'] = (ema9 - ema21) / df['Close']
    features['ema9_slope']  = ema9.diff(3) / df['Close']
    features['ema21_slope'] = ema21.diff(3) / df['Close']
    features['price_vs_ema9']  = (df['Close'] - ema9)  / df['Close']
    features['price_vs_ema21'] = (df['Close'] - ema21) / df['Close']

    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr10 = tr.rolling(10).mean()
    features['atr10'] = atr10 / df['Close']
    features['atr_ratio'] = atr10 / tr.rolling(20).mean()

    df['date'] = df['datetime'].dt.date
    df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['cumvol']   = df.groupby('date')['Volume'].cumsum()
    df['cumtpvol'] = df.groupby('date').apply(lambda x: (x['tp'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
    df['vwap']     = df['cumtpvol'] / df['cumvol']
    features['price_vs_vwap'] = (df['Close'] - df['vwap']) / df['Close']

    features['volatility_10b'] = df['Close'].pct_change().rolling(10).std()
    features['volatility_20b'] = df['Close'].pct_change().rolling(20).std()
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    features['candle_dir'] = (df['Close'] - df['Open']) / df['Open']
    features['high_low_range'] = (df['High'] - df['Low']) / df['Close']

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    rs    = gain / loss
    features['rsi_14'] = 100 - (100 / (1 + rs))

    k = 1.5
    atr = atr10
    src = (df['High'] + df['Low']) / 2
    ts = pd.Series(np.nan, index=df.index)
    ps = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        nl = k * atr.iloc[i]
        prev = ts.iloc[i-1] if not np.isnan(ts.iloc[i-1]) else src.iloc[i]
        s, s1 = src.iloc[i], src.iloc[i-1]
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
    features['adx_proxy'] = tr.rolling(14).mean() / df['Close'].rolling(14).std()

    return features


def evaluate_position(pos_id, position):
    """Pull fresh data, build features, run model, decide hold/exit."""
    df = fetch_recent_bars()
    if df is None or len(df) < 25:
        log_event({"event": "eval_skipped", "reason": "insufficient_data", "pos_id": pos_id})
        return

    features = build_live_features(df)
    latest = features.dropna().iloc[-1:]
    if latest.empty:
        return

    X = latest[FEATURE_COLS]
    model = model_long if position['direction'] == 'long' else model_short
    proba = model.predict_proba(X)[0][1]  # probability of "good" outcome continuing

    record = {
        "event": "model_eval",
        "pos_id": pos_id,
        "direction": position['direction'],
        "confidence": float(proba),
        "current_price": float(df.iloc[-1]['Close']),
        "entry_price": position['entry_price'],
        "bars_held": position['bars_held']
    }

    # simple decision rule: exit if confidence drops below 0.4
    if proba < 0.4:
        record["decision"] = "EXIT_RECOMMENDED"
        with positions_lock:
            if pos_id in open_positions:
                del open_positions[pos_id]
    else:
        record["decision"] = "HOLD"
        with positions_lock:
            if pos_id in open_positions:
                open_positions[pos_id]['bars_held'] += 1

    log_event(record)


def polling_loop():
    """Runs every 5 minutes, evaluates all open positions."""
    while True:
        time.sleep(300)
        with positions_lock:
            positions_snapshot = dict(open_positions)
        for pos_id, pos in positions_snapshot.items():
            evaluate_position(pos_id, pos)


@app.route('/webhook', methods=['POST'])
def receive_signal():
    data = request.get_json(force=True, silent=True)
    if data is None:
        raw = request.data.decode('utf-8')
        data = {"raw": raw}

    record = {"received_at": datetime.utcnow().isoformat(), "payload": data}
    log_event(record)

    text = str(data.get("text", data.get("raw", ""))).upper()

    if "LONG" in text and "GO" in text:
        pos_id = f"long_{int(time.time())}"
        entry_df = fetch_recent_bars(minutes_back=15)
        entry_price = float(entry_df.iloc[-1]['Close']) if entry_df is not None and len(entry_df) > 0 else None
        with positions_lock:
            open_positions[pos_id] = {
                "direction": "long",
                "entry_time": datetime.utcnow().isoformat(),
                "entry_price": entry_price,
                "bars_held": 0
            }
        log_event({"event": "position_opened", "pos_id": pos_id, "direction": "long", "entry_price": entry_price})

    elif "SHORT" in text and "GO" in text:
        pos_id = f"short_{int(time.time())}"
        entry_df = fetch_recent_bars(minutes_back=15)
        entry_price = float(entry_df.iloc[-1]['Close']) if entry_df is not None and len(entry_df) > 0 else None
        with positions_lock:
            open_positions[pos_id] = {
                "direction": "short",
                "entry_time": datetime.utcnow().isoformat(),
                "entry_price": entry_price,
                "bars_held": 0
            }
        log_event({"event": "position_opened", "pos_id": pos_id, "direction": "short", "entry_price": entry_price})

    elif "EXIT" in text:
        log_event({"event": "pinescript_exit_signal", "note": "reference only, not acted on"})

    return jsonify({"status": "received"}), 200


@app.route('/')
def home():
    return "Webhook listener is running."


@app.route('/positions')
def view_positions():
    with positions_lock:
        return jsonify(open_positions)


# start the background polling thread once, at import time
threading.Thread(target=polling_loop, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)