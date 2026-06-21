import pandas as pd
import numpy as np

df = pd.read_csv("spy_intraday.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# RTH only
df = df[(df['datetime'].dt.time >= pd.Timestamp('09:30').time()) &
        (df['datetime'].dt.time <= pd.Timestamp('16:00').time())]
df = df.reset_index(drop=True)
df['date'] = df['datetime'].dt.date

# ── ATR (10-bar, matches his atrPer) ────────────────────────────────────────
high_low   = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift()).abs()
low_close  = (df['Low']  - df['Close'].shift()).abs()
tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr10      = tr.rolling(10).mean()
atr_avg20  = atr10.rolling(20).mean()
df['atr10'] = atr10

# ── UT Bot direction (matches his f_ut function, keyVal=1.5, atrPer=10) ────
k = 1.5
src = (df['High'] + df['Low']) / 2
ts = pd.Series(np.nan, index=df.index)
ps = pd.Series(0, index=df.index)
for i in range(1, len(df)):
    nl = k * atr10.iloc[i]
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
df['ut_direction'] = ps

# ── Regime engine (matches his ADX/EMA logic, defaults) ─────────────────────
ema_fast = df['Close'].ewm(span=9).mean()
ema_slow = df['Close'].ewm(span=21).mean()
spr_pct  = (ema_fast - ema_slow).abs() / df['Close'] * 100
ema_wide = spr_pct >= 0.05

bull = ema_fast > ema_slow
bear = ema_fast < ema_slow
slope_up   = (ema_fast > ema_fast.shift(3)) & (ema_slow > ema_slow.shift(3))
slope_down = (ema_fast < ema_fast.shift(3)) & (ema_slow < ema_slow.shift(3))

# DI+ / DI- (Wilder's, 14-period, matches adxLen default)
up_move   = df['High'].diff()
down_move = -df['Low'].diff()
plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
atr14 = tr.ewm(alpha=1/14).mean()
plus_di  = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14).mean() / atr14
minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14).mean() / atr14
dx  = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
adx = dx.ewm(alpha=1/14).mean()

dir_bull = bull & slope_up   & (plus_di > minus_di)
dir_bear = bear & slope_down & (minus_di > plus_di)

strong_up = (adx >= 22) & dir_bull & ema_wide
strong_dn = (adx >= 22) & dir_bear & ema_wide
weak_up   = (~strong_up) & bull & (adx >= 15)
weak_dn   = (~strong_dn) & bear & (adx >= 15)

raw_score = pd.Series(0, index=df.index)
raw_score[strong_up] = 2
raw_score[strong_dn] = -2
raw_score[weak_up] = raw_score[weak_up].where(raw_score[weak_up] != 0, 1)
raw_score[weak_dn] = raw_score[weak_dn].where(raw_score[weak_dn] != 0, -1)
# simpler reassignment to avoid overwrite bugs
raw_score = pd.Series(0, index=df.index)
for i in range(len(df)):
    if strong_up.iloc[i]: raw_score.iloc[i] = 2
    elif strong_dn.iloc[i]: raw_score.iloc[i] = -2
    elif weak_up.iloc[i]: raw_score.iloc[i] = 1
    elif weak_dn.iloc[i]: raw_score.iloc[i] = -1

# regime confirmation (confirmRg=2 default -> needs streak >= 1)
streak = pd.Series(0, index=df.index)
for i in range(1, len(df)):
    streak.iloc[i] = streak.iloc[i-1] + 1 if raw_score.iloc[i] == raw_score.iloc[i-1] else 0

regime = pd.Series(0, index=df.index)
current = 0
for i in range(len(df)):
    if streak.iloc[i] >= 1:
        current = raw_score.iloc[i]
    regime.iloc[i] = current

reg_up   = regime >= 1
reg_down = regime <= -1

# ── Operational filters ──────────────────────────────────────────────────
slow_day = atr10 < 0.70 * atr_avg20

# ── Full timproto1 entry condition ──────────────────────────────────────
can_long  = (df['ut_direction'] == 1)  & reg_up   & (~slow_day)
can_short = (df['ut_direction'] == -1) & reg_down & (~slow_day)

long_trig  = can_long  & (~can_long.shift(1).fillna(False))
short_trig = can_short & (~can_short.shift(1).fillna(False))

print(f"Total long entry signals:  {long_trig.sum()}")
print(f"Total short entry signals: {short_trig.sum()}")

# ── Label trades from entry signals only ─────────────────────────────────
def label_trade(entry_idx, direction):
    entry_price = df.iloc[entry_idx]['Close']
    entry_atr   = df.iloc[entry_idx]['atr10']
    entry_date  = df.iloc[entry_idx]['date']

    if pd.isna(entry_atr) or entry_atr == 0:
        return None

    stop_dist   = 2.0 * entry_atr
    target_dist = 3.0 * entry_atr

    if direction == 'long':
        stop_price   = entry_price - stop_dist
        target_price = entry_price + target_dist
    else:
        stop_price   = entry_price + stop_dist
        target_price = entry_price - target_dist

    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        if row['date'] != entry_date:
            final_return = (row['Close'] - entry_price) / entry_price
            if direction == 'short':
                final_return = -final_return
            return {'outcome': 'timeout', 'return_pct': final_return,
                    'label': 1 if final_return > 0 else 0, 'bars_held': i - entry_idx}

        high, low = row['High'], row['Low']
        if direction == 'long':
            if low <= stop_price:
                loss = (stop_price - entry_price) / entry_price
                return {'outcome': 'stop', 'return_pct': loss, 'label': 0, 'bars_held': i - entry_idx}
            if high >= target_price:
                gain = (target_price - entry_price) / entry_price
                return {'outcome': 'target', 'return_pct': gain, 'label': 1, 'bars_held': i - entry_idx}
        else:
            if high >= stop_price:
                loss = (entry_price - stop_price) / entry_price
                return {'outcome': 'stop', 'return_pct': -loss, 'label': 0, 'bars_held': i - entry_idx}
            if low <= target_price:
                gain = (entry_price - target_price) / entry_price
                return {'outcome': 'target', 'return_pct': gain, 'label': 1, 'bars_held': i - entry_idx}
    return None

results = []
for i in df.index[long_trig]:
    r = label_trade(i, 'long')
    if r:
        results.append({'datetime': df.iloc[i]['datetime'], 'direction': 'long', **r})

for i in df.index[short_trig]:
    r = label_trade(i, 'short')
    if r:
        results.append({'datetime': df.iloc[i]['datetime'], 'direction': 'short', **r})

labels_df = pd.DataFrame(results)
labels_df.to_csv("labels_intraday.csv", index=False)
print(f"\nSaved {len(labels_df)} rows to labels_intraday.csv")
print(labels_df['outcome'].value_counts())
print(f"Label balance: {labels_df['label'].mean():.2%} positive")