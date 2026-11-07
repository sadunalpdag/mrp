"""Seven-day PAPER scalp probes. Never places orders, opens WS feeds, or requests REST.

Both signals are 15m proxies from existing CLOSED WS candles. Break & Bounce
requires yesterday's UTC day range already saved locally; 5m entry confirmation
is unavailable and is never represented as confirmed.
"""
import json
import math
import os
import threading
import time
from datetime import datetime, timezone

STEP_MS = 900_000
INTERVAL_S = 300
HORIZON_S = 12 * 3600
MAX_SYMBOLS = 10
MAX_SHEET_SIGNALS = 48
RETENTION_S = 8 * 86400
TABS = {"PULLBACK": "Pullback Test", "BREAK_BOUNCE": "Break Bounce Test"}
HEADERS = ["Coin", "Yön", "Mum zamanı UTC", "Sanal giriş", "TP %1",
           "Risk -%1", "Sonuç", "Sonuç zamanı UTC", "Model", "Gözlem"]
MODEL = "15m mum vekili / gerçek 5m teyidi YOK"


def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _bars(rows, now):
    """Only genuine, correctly timed, already closed 15m OHLCV; no gap filling."""
    by_ts = {}
    for r in rows if isinstance(rows, list) else ():
        try:
            ts, close_ts = int(r[0]), int(r[6])
            o, h, l, c, vol = [_number(r[j]) for j in (1, 2, 3, 4, 5)]
            if (ts % STEP_MS or close_ts != ts + STEP_MS - 1
                    or close_ts >= int(now * 1000)
                    or any(x is None for x in (o, h, l, c, vol))
                    or min(o, h, l, c) <= 0 or vol < 0
                    or l > min(o, c) or h < max(o, c)):
                continue
            by_ts[ts] = (ts, o, h, l, c, vol)
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
    return [by_ts[ts] for ts in sorted(by_ts)[-64:]]


def _ready(bars, now):
    # Up to 4 absent true candles in 56 slots; never invent/fill bars.
    # This is a deliberately labeled proxy, not a strictly uniform SMA sample.
    if len(bars) < 52:
        return False
    newest = bars[-1][0]
    recent = [b for b in bars if newest - 55 * STEP_MS <= b[0] <= newest]
    return (len(recent) >= 52 and
            0 <= now * 1000 - (newest + STEP_MS - 1) <= 1800000)


def pullback(bars):
    """50-SMA trend + engulfing at SMA50: intentionally conservative 15m proxy."""
    if len(bars) < 52:
        return None
    old, prev, last = bars[-3:]
    ma = sum(r[4] for r in bars[-50:]) / 50
    prior_ma = sum(r[4] for r in bars[-51:-1]) / 50
    if (ma > prior_ma and last[4] > ma and
            prev[4] < prev[1] and last[4] > last[1] and
            last[1] <= prev[4] and last[4] >= prev[1] and
            min(prev[3], last[3]) <= ma * 1.005 and
            old[4] > ma * 0.995):
        return "LONG", "MA50 yukarı + bullish engulfing; 15m vekili"
    if (ma < prior_ma and last[4] < ma and
            prev[4] > prev[1] and last[4] < last[1] and
            last[1] >= prev[4] and last[4] <= prev[1] and
            max(prev[2], last[2]) >= ma * 0.995 and
            old[4] < ma * 1.005):
        return "SHORT", "MA50 aşağı + bearish engulfing; 15m vekili"
    return None


def previous_utc_range(rows, now):
    """No claimed daily level when any of yesterday's 96 15m candles is missing."""
    midnight = int(now // 86400) * 86400 * 1000
    prior_start = midnight - 86400000
    clean = _bars_for_range(rows, now, prior_start, midnight)
    if len(clean) != 96:
        return None
    return max(c[2] for c in clean), min(c[3] for c in clean)


def _bars_for_range(rows, now, start_ms, end_ms):
    # The 64-bar WS cache is not enough for a complete daily range.
    # Read the already stored 5-day local history instead, if available.
    seen = {}
    for r in rows if isinstance(rows, list) else ():
        try:
            ts, close_ts = int(r[0]), int(r[6])
            if (ts < start_ms or ts >= end_ms or ts % STEP_MS or
                    close_ts != ts + STEP_MS - 1 or close_ts >= now * 1000):
                continue
            o, h, l, c = [_number(r[j]) for j in (1, 2, 3, 4)]
            if (any(x is None or x <= 0 for x in (o, h, l, c))
                    or l > min(o, c) or h < max(o, c)):
                continue
            seen[ts] = (ts, o, h, l, c)
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
    return [seen[t] for t in sorted(seen)]


def break_bounce(bars, daily_range):
    """Yesterday's range, 15m close beyond it, then 15m retest proxy."""
    if len(bars) < 3 or daily_range is None:
        return None
    prev2, prev, last = bars[-3:]
    high, low = daily_range
    body = abs(prev[4] - prev[1]) / max(prev[2] - prev[3], 1e-15)
    if body < 0.45:
        return None
    if (prev2[4] <= high and prev[1] <= high and prev[4] > high and
            last[3] <= high * 1.003 and last[4] > high and last[4] > last[1]):
        return "LONG", "Dünün üst sınırı kırıldı, 15m retest; 5m teyit yok"
    if (prev2[4] >= low and prev[1] >= low and prev[4] < low and
            last[2] >= low * 0.997 and last[4] < low and last[4] < last[1]):
        return "SHORT", "Dünün alt sınırı kırıldı, 15m retest; 5m teyit yok"
    return None


def _load(path):
    try:
        with open(path, encoding="utf-8") as fd:
            obj = json.load(fd)
        if isinstance(obj, dict) and isinstance(obj.get("signals"), list):
            return obj
    except (OSError, ValueError, TypeError):
        pass
    return {"signals": []}


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as fd:
        json.dump(data, fd, ensure_ascii=False, allow_nan=False)
        fd.flush()
        os.fsync(fd.fileno())
    os.replace(temp, path)


def _dt(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%d.%m %H:%M")


def _settle(signal, bars, now):
    if signal["status"] != "OPEN":
        return
    expiry_ms = signal["ts"] + int(HORIZON_S * 1000)
    later = [r for r in bars if signal["ts"] < r[0] <= expiry_ms]
    for r in later:
        if signal["side"] == "LONG":
            good, bad = r[2] >= signal["tp"], r[3] <= signal["risk"]
        else:
            good, bad = r[3] <= signal["tp"], r[2] >= signal["risk"]
        if good or bad:
            signal["status"] = ("AMBIGUOUS" if good and bad else "TP" if good else "RISK")
            signal["resolved_at"] = r[0] + STEP_MS
            return
    if now * 1000 >= expiry_ms:
        # Do not call an unobserved move a timeout.
        current = bars[-1][0] + STEP_MS if bars else 0
        signal["status"] = "TIMEOUT" if current >= expiry_ms else "DATA_GAP"
        signal["resolved_at"] = int(now * 1000)


def _scan(data_dir, tickers, candles_get, now, obj):
    # Never start additional WS subscriptions or request any REST data.
    ranked = []
    for t in tickers if isinstance(tickers, list) else ():
        if not isinstance(t, dict):
            continue
        symbol = t.get("symbol", "")
        change = _number(t.get("priceChangePercent"))
        volume = _number(t.get("quoteVolume"))
        if (not symbol.endswith("USDT") or not symbol[:-4].isalnum() or
                change is None or volume is None or volume < 0):
            continue
        ranked.append((change, -volume, symbol))
    ranked.sort()
    available = []
    for _, neg_volume, symbol in ranked[:20]:  # same existing Crash candidates
        bars = _bars(candles_get(symbol), now)
        if _ready(bars, now):
            available.append((-neg_volume, symbol, bars))
    available.sort(reverse=True)
    selected = available[:MAX_SYMBOLS]
    known = {(s["strategy"], s["symbol"], s["ts"])
             for s in obj["signals"] if isinstance(s, dict)}
    by_symbol = {s: b for _, s, b in selected}
    # A paper signal might leave the current top 10. Read ONLY the existing
    # local WS cache for up to 24 such symbols; never add WS subscriptions.
    pending = sorted((s for s in obj["signals"] if s.get("status") == "OPEN"),
                     key=lambda item: item.get("ts", 0), reverse=True)
    pending_symbols = []
    for item in pending:
        sym = item.get("symbol")
        if isinstance(sym, str) and sym not in by_symbol and sym not in pending_symbols:
            pending_symbols.append(sym)
        if len(pending_symbols) >= 24:
            break
    for sym in pending_symbols:
        by_symbol[sym] = _bars(candles_get(sym), now)
    for signal in obj["signals"]:
        if signal.get("status") == "OPEN":
            _settle(signal, by_symbol.get(signal.get("symbol"), []), now)
    for _, symbol, bars in selected:
        # Use the same last-closed-bar timestamp as the other V2 indicators.
        entry_ts = bars[-1][0]
        proposals = [("PULLBACK", pullback(bars))]
        yesterday_path = os.path.join(data_dir, "crash_research", "history_cache", symbol + ".json")
        try:
            with open(yesterday_path, encoding="utf-8") as fd:
                daily_rows = json.load(fd)
        except (OSError, ValueError, TypeError):
            daily_rows = []
        proposals.append(("BREAK_BOUNCE", break_bounce(bars, previous_utc_range(daily_rows, now))))
        for strategy, result in proposals:
            if result is None or (strategy, symbol, entry_ts) in known:
                continue
            side, note = result
            entry = bars[-1][4]
            obj["signals"].append({
                "strategy": strategy, "symbol": symbol, "side": side,
                "ts": entry_ts, "entry": entry, "tp": entry * (1.01 if side == "LONG" else 0.99),
                "risk": entry * (0.99 if side == "LONG" else 1.01),
                "status": "OPEN", "resolved_at": None, "note": note,
            })
            known.add((strategy, symbol, entry_ts))
    obj["signals"] = [s for s in obj["signals"]
                      if isinstance(s, dict) and now - s.get("ts", 0) / 1000 < RETENTION_S][-2000:]
    obj["last_scan"] = int(now)
    obj["sampled_coins"] = [s for _, s, _ in selected]
    obj["available_count"] = len(available)
    return obj


def _tab_values(obj, strategy):
    signals = [s for s in obj["signals"] if s.get("strategy") == strategy
               and obj["last_scan"] - s["ts"] / 1000 <= 7 * 86400]
    statuses = {k: sum(s.get("status") == k for s in signals)
                for k in ("OPEN", "TP", "RISK", "TIMEOUT", "AMBIGUOUS", "DATA_GAP")}
    resolved = statuses["TP"] + statuses["RISK"]
    rate = str(round(100 * statuses["TP"] / resolved, 1)) + "%" if resolved else "—"
    rows = [
        ["Yedi günlük sanal test", TABS[strategy]],
        ["Model", MODEL],
        ["Son tarama UTC", _dt(obj["last_scan"] * 1000)],
        ["Takip edilen coin", ", ".join(obj.get("sampled_coins", [])) or "VERİ YETERSİZ"],
        ["Sinyal / açık / TP / risk", "%s / %s / %s / %s" %
         (len(signals), statuses["OPEN"], statuses["TP"], statuses["RISK"])],
        ["Sadece TP ve risk arasında TP oranı", rate],
        ["Belirsiz / zaman aşımı / veri boşluğu",
         "%s / %s / %s" % (statuses["AMBIGUOUS"], statuses["TIMEOUT"], statuses["DATA_GAP"])],
        ["DİKKAT", "5m teyidi YOK; gerçek işlem, komisyon veya kaldıraç dahil değil."],
        HEADERS,
    ]
    for s in sorted(signals, key=lambda x: x["ts"], reverse=True)[:MAX_SHEET_SIGNALS]:
        rows.append([
            s["symbol"], s["side"], _dt(s["ts"]), round(s["entry"], 9),
            round(s["tp"], 9), round(s["risk"], 9), s["status"],
            _dt(s["resolved_at"]) if s.get("resolved_at") else "—",
            "15m PROXY", s.get("note", ""),
        ])
    return rows + [[""] * len(HEADERS) for _ in range(max(0, 9 + MAX_SHEET_SIGNALS - len(rows)))]


def _publish(obj):
    from crash_sheets_dashboard import _credentials, _name
    client, book_id = _credentials()
    if client is None:
        return False
    book = client.open_by_key(book_id)
    if "emafilter" not in _name(book.title):
        raise ValueError("SCALP_WRONG_WORKBOOK")
    tabs = {_name(t.title): t for t in book.worksheets()}
    targets = {}
    for strategy, title in TABS.items():
        key = _name(title)
        if key not in tabs:
            tabs[key] = book.add_worksheet(title=title, rows=80, cols=12)
        targets[strategy] = tabs[key]
    # Writes only own tabs, overwrites fixed ranges (never appends indefinitely).
    data = [{"range": "'" + ws.title.replace("'", "''") + "'!A1:J57",
             "values": _tab_values(obj, strategy)}
            for strategy, ws in targets.items()]
    book.values_batch_update({"valueInputOption": "RAW", "data": data})
    return True


class ShadowWeek:
    def __init__(self, directory, logger, clock=time.time):
        self.directory, self.logger, self.clock = directory, logger, clock
        self.next_due = 0
        self.lock = threading.Lock()
        self.path = os.path.join(directory, "scalp_shadow_week.json")

    def tick(self, tickers, candles_get):
        now = self.clock()
        if now < self.next_due or not self.lock.acquire(False):
            return
        self.next_due = now + INTERVAL_S
        snapshot = [dict(t) for t in tickers if isinstance(t, dict)]
        def work():
            try:
                data = _scan(self.directory, snapshot, candles_get, now, _load(self.path))
                _save(self.path, data)
                try:
                    if _publish(data):
                        self.logger("[SCALP SHADOW] Pullback Test / Break Bounce Test updated; "
                                    "coins=%s; paper_signals=%s; no REST" %
                                    (len(data["sampled_coins"]), len(data["signals"])))
                    else:
                        self.logger("[SCALP SHADOW] local paper results saved; Sheets credentials missing")
                except Exception as exc:
                    self.logger("[SCALP SHADOW SHEETS] skipped: " + type(exc).__name__)
            except Exception as exc:
                self.logger("[SCALP SHADOW] skipped: " + type(exc).__name__)
            finally:
                self.lock.release()
        threading.Thread(target=work, name="scalp-shadow-week", daemon=True).start()
