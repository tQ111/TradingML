from flask import Flask, request, jsonify
import os
import json
import threading
import requests
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

LOG_FILE = "signals_log.jsonl"
BAR_FILE = "bars_log.jsonl"

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

MAX_BUFFER = 200  # enough bars for all rolling features

bar_buffer = []          # list of dicts: datetime, Open, High, Low, Close, Volume
open_positions = {}      # pos_id -> {direction, entry_time, entry_price, bars_held}
data_lock = threading.Lock()

GITHUB_TOKEN = os.environ.get("GH_PAT", "")
GITHUB_REPO = "tQ111/TradingML"
GITHUB_FILE = "data/bars_log.jsonl"
GITHUB_BRANCH = "main"

def push_bar_to_github(bar_record):
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            info = r.json()
            current = __import__('base64').b64decode(info['content']).decode('utf-8')
            sha = info['sha']
        else:
            current = ""
            sha = None
        new_content = current + bar_record + "\n"
        encoded = __import__('base64').b64encode(new_content.encode('utf-8')).decode('utf-8')
        payload = {"message": f"bar: {bar_record[:40]}", "content": encoded, "branch": GITHUB_BRANCH}
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except Exception as e:
        print(f"GitHub push failed: {e}")
        
def log_event(record):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(record)


def build_live_features(df):
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

    df = df.copy()
    df['date'] = df['datetime'].dt.date
    df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tpvol'] = df['tp'] * df['Volume']
    df['cumvol']   = df.groupby('date')['Volume'].cumsum()
    df['cumtpvol'] = df.groupby('date')['tpvol'].cumsum()
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


def evaluate_all_positions():
    """Called every time a new bar arrives. Re-evaluates every open position."""
    with data_lock:
        if len(bar_buffer) < 25:
            return
        df = pd.DataFrame(bar_buffer)
        positions_snapshot = dict(open_positions)

    features = build_live_features(df)
    latest = features.dropna().iloc[-1:]
    if latest.empty:
        return
    X = latest[FEATURE_COLS]
    current_price = float(df.iloc[-1]['Close'])

    for pos_id, pos in positions_snapshot.items():
        model = model_long if pos['direction'] == 'long' else model_short
        proba = float(model.predict_proba(X)[0][1])

        record = {
            "event": "model_eval",
            "pos_id": pos_id,
            "direction": pos['direction'],
            "confidence": proba,
            "current_price": current_price,
            "entry_price": pos['entry_price'],
            "bars_held": pos['bars_held']
        }

        if proba < 0.4:
            record["decision"] = "EXIT_RECOMMENDED"
            with data_lock:
                if pos_id in open_positions:
                    del open_positions[pos_id]
        else:
            record["decision"] = "HOLD"
            with data_lock:
                if pos_id in open_positions:
                    open_positions[pos_id]['bars_held'] += 1

        log_event(record)


@app.route('/webhook', methods=['POST'])
def receive_signal():
    raw = request.data.decode('utf-8')
    try:
        data = json.loads(raw)
    except Exception:
        data = {"raw": raw}

    log_event({"received_at": datetime.now(timezone.utc).isoformat(), "payload": data})

    msg_type = data.get("type", "")

    if msg_type == "bar":
        bar = {
            "datetime": pd.to_datetime(int(data["time"]), unit='s'),
            "Open": float(data["open"]),
            "High": float(data["high"]),
            "Low": float(data["low"]),
            "Close": float(data["close"]),
            "Volume": float(data["volume"])
        }
        with data_lock:
            bar_buffer.append(bar)
            if len(bar_buffer) > MAX_BUFFER:
                bar_buffer.pop(0)
        bar_record = json.dumps({"datetime": str(bar["datetime"]), "Open": bar["Open"], "High": bar["High"], "Low": bar["Low"], "Close": bar["Close"], "Volume": bar["Volume"]})
        with open(BAR_FILE, "a") as f:
            f.write(bar_record + "\n")
        threading.Thread(target=push_bar_to_github, args=(bar_record,), daemon=True).start()
        evaluate_all_positions()

    elif msg_type == "long":
        pos_id = f"long_{int(datetime.now(timezone.utc).timestamp())}"
        with data_lock:
            open_positions[pos_id] = {
                "direction": "long",
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "entry_price": float(data.get("price", 0)),
                "bars_held": 0
            }
        log_event({"event": "position_opened", "pos_id": pos_id, "direction": "long", "entry_price": data.get("price")})

    elif msg_type == "short":
        pos_id = f"short_{int(datetime.now(timezone.utc).timestamp())}"
        with data_lock:
            open_positions[pos_id] = {
                "direction": "short",
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "entry_price": float(data.get("price", 0)),
                "bars_held": 0
            }
        log_event({"event": "position_opened", "pos_id": pos_id, "direction": "short", "entry_price": data.get("price")})

    elif msg_type in ("exit_long", "exit_short"):
        log_event({"event": "pinescript_exit_signal", "note": "reference only, not acted on", "data": data})

    return jsonify({"status": "received"}), 200


@app.route('/')
def home():
    return "Webhook listener is running."


@app.route('/positions')
def view_positions():
    with data_lock:
        return jsonify(open_positions)


@app.route('/buffer')
def view_buffer():
    with data_lock:
        return jsonify(len(bar_buffer))
@app.route('/export_bars')
def export_bars():
    try:
        with open(BAR_FILE, "r") as f:
            return f.read(), 200, {'Content-Type': 'application/json'}
    except FileNotFoundError:
        return "", 200

@app.route('/export_signals')
def export_signals():
    try:
        with open(LOG_FILE, "r") as f:
            return f.read(), 200, {'Content-Type': 'application/json'}
    except FileNotFoundError:
        return "", 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)