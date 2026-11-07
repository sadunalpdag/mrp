"""Read-only falling-channel diagnostic from CLOSED cached 15m klines.

No Binance/REST calls, subscriptions, trades or score modifications. A channel
must have descending swing highs AND lows; ambiguous patterns remain unknown.
"""
import math

STEP_MS = 900_000
MIN_BARS = 38
MAX_BARS = 64


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _swings(bars, key, high=True):
    points = []
    for i in range(2, len(bars) - 2):
        value = bars[i][key]
        neighbors = [bars[j][key] for j in (i-2, i-1, i+1, i+2)]
        if (value > max(neighbors)) if high else (value < min(neighbors)):
            points.append((i, value))
    return points


def _descending_pair(points):
    # Latest qualifying pair, not a cherry-picked ancient one.
    for right in range(len(points) - 1, 0, -1):
        i2, v2 = points[right]
        for left in range(right - 1, -1, -1):
            i1, v1 = points[left]
            spacing = i2 - i1
            if spacing > 40:
                break
            if spacing >= 6 and v2 < v1 * 0.997:
                return (i1, v1), (i2, v2)
    return None


def analyze(rows, now):
    """Return status, percent distance to upper line and last confirmation time.

    The last five bars are excluded from channel fitting, so a breakout cannot
    move its own resistance line. All rows must be fully closed and recent.
    """
    unknown = {"status": "VERİ YOK", "distance_pct": None, "confirmed_at": None}
    if not rows:
        return unknown
    now_ms = int(now * 1000)
    indexed = {}
    for row in rows:
        try:
            if len(row) < 7:
                continue
            opened, closed = int(row[0]), int(row[6])
            if closed - opened != STEP_MS - 1 or closed >= now_ms:
                continue
            hi, lo, close = (_number(row[j]) for j in (2, 3, 4))
            if None in (hi, lo, close) or not lo <= close <= hi:
                continue
            if opened % STEP_MS != 0:
                continue
            indexed[opened] = (opened, hi, lo, close, closed)
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
    bars = [indexed[k] for k in sorted(indexed)][-MAX_BARS:]
    if len(bars) < MIN_BARS or not 0 <= now_ms - bars[-1][4] <= 30 * 60_000:
        return unknown
    # Missing close slots could distort pivot geometry.
    if sum((bars[i][0] - bars[i-1][0]) != STEP_MS
           for i in range(1, len(bars))) > 2:
        return unknown

    train = bars[:-5]
    highs = _descending_pair(_swings(train, 1, True))
    lows = _descending_pair(_swings(train, 2, False))
    no_channel = {"status": "KANAL YOK", "distance_pct": None, "confirmed_at": None}
    if highs is None or lows is None:
        return no_channel
    (h1, v1), (h2, v2) = highs
    (l1, p1), (l2, p2) = lows
    upper_slope = (v2 - v1) / (h2 - h1)
    lower_slope = (p2 - p1) / (l2 - l1)
    if upper_slope >= 0 or lower_slope >= 0:
        return no_channel
    # Roughly parallel upper/lower bounds, with visible positive channel width.
    slope_ratio = lower_slope / upper_slope
    if not 0.35 <= slope_ratio <= 2.2:
        return no_channel

    def upper(i):
        return v2 + upper_slope * (i - h2)

    gap1, gap2 = upper(l1) - p1, upper(l2) - p2
    width = (gap1 + gap2) / 2
    if (width <= 0 or width > 0.25 * v2
            or abs(gap1-gap2) > 0.6 * width
            or min(upper(i) for i in range(len(train), len(bars))) <= 0):
        return no_channel

    last = len(bars) - 1
    distance = (bars[last][3] / upper(last) - 1) * 100
    # Two closed 15m bars above the fitted upper line, with 0.15% buffer.
    above = [bars[i][3] > upper(i) * 1.0015
             for i in range(len(train), len(bars))]
    confirm_indices = [i for i in range(1, len(above))
                       if above[i] and above[i-1]]
    confirmed_at = None
    if confirm_indices:
        index = len(train) + confirm_indices[-1]
        confirmed_at = bars[index][4] / 1000
    if confirmed_at and not above[-1] and distance < 0:
        status = "SAHTE KIRILIM"
    elif confirmed_at and above[-1]:
        status = "KIRDI"
    elif distance > 0:
        status = "KIRILIM BEKLEME"
    elif distance >= -0.5:
        status = "YAKLAŞIYOR"
    else:
        status = "KIRMADI"
    return {"status": status, "distance_pct": round(distance, 2),
            "confirmed_at": confirmed_at}
