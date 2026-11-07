"""Five-day Crash Bounce diagnostics, exclusively from local cached CLOSED 15m candles.

Read-only: no network, order submission, extra subscriptions, or score changes.
Missing history is reported as missing rather than inferred from 16h of WS bars.
"""
import json
import math
import os

STEP_MS = 900_000
COUNT = 480
MAX_MISSING = 2
BUFFER = 0.0015


def _price(value):
    try:
        p = float(value)
        return p if math.isfinite(p) and p > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _valid(row, start, end, now_ms):
    try:
        t, close_t = int(row[0]), int(row[6])
        op, hi, lo, close = (_price(row[j]) for j in (1, 2, 3, 4))
        if (t % STEP_MS or not start <= t < end or
                close_t != t + STEP_MS - 1 or close_t >= now_ms or
                None in (op, hi, lo, close) or
                not lo <= min(op, close) <= max(op, close) <= hi):
            return None
        return t
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _binance(row):
    return len(row) <= 12 or row[12] not in ("BYBIT_LINEAR", "COINBASE_SPOT")


def load_cached(directory, symbol, ws_rows, now):
    """Merge local 5d history_cache with existing WS rows; NEVER fetch data.

    Binance candles are preferred to reference-venue candles for identical slots.
    The history collector stores 10d, this projects the last exactly 480 slots.
    """
    if not isinstance(symbol, str) or not symbol.endswith("USDT") or not symbol.isascii() or not symbol.isalnum():
        return []
    path = os.path.join(directory, "crash_research", "history_cache", symbol + ".json")
    try:
        with open(path, encoding="utf-8") as fh:
            disk = json.load(fh)
        if not isinstance(disk, list):
            disk = []
    except (OSError, ValueError, TypeError):
        disk = []
    end = int(now * 1000) // STEP_MS * STEP_MS
    start = end - COUNT * STEP_MS
    now_ms = int(now * 1000)
    slots = {}
    for row in (disk + (ws_rows if isinstance(ws_rows, list) else [])):
        if not isinstance(row, (list, tuple)):
            continue
        t = _valid(row, start, end, now_ms)
        if t is None:
            continue
        previous = slots.get(t)
        if previous is None or _binance(row) or not _binance(previous):
            slots[t] = row
    return [slots[t] for t in sorted(slots)]


def _swings(bars, field, higher):
    points = []
    for i in range(2, len(bars) - 2):
        v = bars[i][field]
        neighbor = [bars[k][field] for k in (i - 2, i - 1, i + 1, i + 2)]
        if (v > max(neighbor)) if higher else (v < min(neighbor)):
            points.append((bars[i][0], v))
    return points


def _pair(points, rising):
    # At least 12 hours between pivots. The lookback is the actual five-day
    # history, not the last 64 WS candles.
    for right in range(len(points) - 1, 0, -1):
        x2, v2 = points[right]
        for left in range(right - 1, -1, -1):
            x1, v1 = points[left]
            span = x2 - x1
            if span > 420:
                break
            if span >= 48 and ((v2 >= v1 * 1.005) if rising else (v2 <= v1 * 0.995)):
                return (x1, v1), (x2, v2)
    return None


def _channel(bars):
    # Fit before the last five candles: a breakout must not move its own boundary.
    training = bars[:-5]
    high_pivots = _swings(training, 1, True)
    low_pivots = _swings(training, 2, False)
    candidates = []
    for rising in (False, True):
        hp, lp = _pair(high_pivots, rising), _pair(low_pivots, rising)
        if not hp or not lp:
            continue
        (hx1, hv1), (hx2, hv2) = hp
        (lx1, lv1), (lx2, lv2) = lp
        hs, ls = (hv2 - hv1) / (hx2 - hx1), (lv2 - lv1) / (lx2 - lx1)
        if hs == 0 or ls == 0 or not 0.4 <= (ls / hs) <= 2.5:
            continue
        upper = lambda x: hv2 + hs * (x - hx2)
        lower = lambda x: lv2 + ls * (x - lx2)
        check = [min(hx1, lx1), max(hx2, lx2), bars[-1][0]]
        widths = [upper(x) - lower(x) for x in check]
        if min(widths) <= 0 or max(widths) > bars[-1][3] * 0.35:
            continue
        if upper(bars[-1][0]) <= 0 or lower(bars[-1][0]) <= 0:
            continue
        # Prefer a more recently validated channel, irrespective of direction.
        candidates.append((min(hx2, lx2), upper, lower))
    return max(candidates, key=lambda c: c[0]) if candidates else None


def _display_price(v):
    return float(format(v, ".9g")) if v is not None and math.isfinite(v) else None


def analyze(rows, now):
    """Support, resistance, Fibonacci and two-way channel status from 5d data.

    Return neutral status whenever the five-day candle coverage is insufficient.
    No open candles or future data are considered.
    """
    empty = dict(status="VERİ YOK", direction="—", distance_pct=None,
                 confirmed_at=None, coverage="0/480", support=None, resistance=None,
                 fib={}, fib_band="—")
    if not isinstance(rows, list):
        return empty
    now_ms = int(now * 1000)
    end = now_ms // STEP_MS * STEP_MS
    start = end - COUNT * STEP_MS
    mapped = {}
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        t = _valid(row, start, end, now_ms)
        if t is not None:
            mapped[t] = row
    bars = [((t - start) // STEP_MS, float(r[2]), float(r[3]), float(r[4]), int(r[6]))
            for t, r in sorted(mapped.items())]
    result = dict(empty, coverage=f"{len(bars)}/{COUNT}")
    if (len(bars) < COUNT - MAX_MISSING or not bars
            or end - 1 - bars[-1][4] > 30 * 60_000):
        return result
    price = bars[-1][3]
    lo, hi = min(x[2] for x in bars), max(x[1] for x in bars)
    lows = _swings(bars, 2, False)
    highs = _swings(bars, 1, True)
    lower_levels = [v for _, v in lows] + [lo]
    upper_levels = [v for _, v in highs] + [hi]
    result["support"] = _display_price(max((v for v in lower_levels if v <= price), default=None))
    result["resistance"] = _display_price(min((v for v in upper_levels if v >= price), default=None))
    fib = {name: _display_price(lo + factor * (hi - lo)) for name, factor in
           (("0", 0), ("23.6", .236), ("38.2", .382), ("50", .5),
            ("61.8", .618), ("78.6", .786), ("100", 1))}
    result["fib"] = fib
    levels = [("0", fib["0"]), ("23.6", fib["23.6"]),
              ("38.2", fib["38.2"]), ("50", fib["50"]),
              ("61.8", fib["61.8"]), ("78.6", fib["78.6"]),
              ("100", fib["100"])]
    if price < lo:
        result["fib_band"] = "DİP ALTI"
    elif price > hi:
        result["fib_band"] = "TEPE ÜSTÜ"
    else:
        for i in range(len(levels) - 1):
            if levels[i][1] <= price <= levels[i+1][1]:
                result["fib_band"] = levels[i][0] + "–" + levels[i+1][0]
                break

    channel = _channel(bars)
    if channel is None:
        result["status"] = "KANAL YOK"
        return result
    _, upper, lower = channel
    recent = bars[-5:]
    up = [b[3] > upper(b[0]) * (1 + BUFFER) for b in recent]
    down = [b[3] < lower(b[0]) * (1 - BUFFER) for b in recent]
    up_confirm = [i for i in range(1, len(up)) if up[i-1] and up[i]]
    down_confirm = [i for i in range(1, len(down)) if down[i-1] and down[i]]
    result["distance_pct"] = round((price / upper(bars[-1][0]) - 1) * 100, 2)
    if up_confirm and up[-1] and (not down_confirm or up_confirm[-1] > down_confirm[-1]):
        result.update(status="KIRDI", direction="YUKARI", confirmed_at=recent[up_confirm[-1]][4] / 1000)
    elif down_confirm and down[-1]:
        result.update(status="KIRDI", direction="AŞAĞI", confirmed_at=recent[down_confirm[-1]][4] / 1000)
    elif up_confirm or down_confirm:
        if up_confirm[-1:] > down_confirm[-1:]:
            result.update(status="SAHTE KIRILIM", direction="YUKARI")
        else:
            result.update(status="SAHTE KIRILIM", direction="AŞAĞI")
    elif price > upper(bars[-1][0]):
        result.update(status="KIRILIM BEKLEME", direction="YUKARI")
    elif price < lower(bars[-1][0]):
        result.update(status="KIRILIM BEKLEME", direction="AŞAĞI")
    else:
        du = (upper(bars[-1][0]) / price - 1) * 100
        dl = (price / lower(bars[-1][0]) - 1) * 100
        if min(du, dl) <= 0.5:
            result.update(status="YAKLAŞIYOR", direction="YUKARI" if du <= dl else "AŞAĞI")
        else:
            result.update(status="KIRMADI", direction="—")
    return result
