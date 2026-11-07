"""Seven-day PAPER scalp probes; uses already connected futures WS, never REST/orders.

Pullback observes 24h top-gainer ranks 11-20; Break & Bounce observes 21-30.
Closed 15m WS candles only, with separate bounded subscriptions supplied by
CrashBounceWebSocketCache. Actual 5m entry confirmation remains unavailable.
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
MAX_SYMBOLS = 10  # per strategy, 20 combined
MAX_SHEET_SIGNALS = 48
SELECTION_VERSION = "positive_gainers_11_20_vs_21_30_v1"
RANK_BOUNDS = {"PULLBACK": (10, 20), "BREAK_BOUNCE": (20, 30)}
RETENTION_S = 8 * 86400
TABS = {"PULLBACK": "Pullback Test", "BREAK_BOUNCE": "Break Bounce Test"}
HEADERS = ["Coin", "Yön", "Mum zamanı UTC", "Sanal giriş", "TP %1",
           "Risk -%1", "Sonuç", "Sonuç zamanı UTC", "Model", "Gözlem"]
MODEL = "15m mum vekili / gerçek 5m teyidi YOK"
DIAG_HEADERS = ["Coin", "Veri", "15m mum", "MA50 trend", "Yön",
                "Durum", "Koşullar", "Eksik / beklenen", "Sanal sinyal", "Son mum UTC"]


def gainer_groups(tickers, allowed_market=None):
    """Exact 11-20 and 21-30 ranks from current, positive 24h USDT-M gainers.

    Filter to existing verified USDT perpetual market metadata when provided.
    A missing 15m history never promotes rank 31 into an earlier bucket.
    """
    allowed = set(allowed_market) if allowed_market is not None else None
    by_symbol = {}
    for item in tickers if isinstance(tickers, list) else ():
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        if not isinstance(symbol, str) or not symbol.endswith("USDT"):
            continue
        if len(symbol) > 28 or not symbol[:-4].isascii() or not symbol[:-4].isalnum():
            continue
        if allowed is not None and symbol not in allowed:
            continue
        change = _number(item.get("priceChangePercent"))
        if change is None or change <= 0:
            continue
        by_symbol[symbol] = change
    ranked = [s for s, change in sorted(by_symbol.items(), key=lambda x: (-x[1], x[0]))]
    return {k: ranked[start:end] for k, (start, end) in RANK_BOUNDS.items()}


def open_shadow_symbols(directory, now, max_extra=16):
    """Keep active paper trades subscribed after they leave their gainer rank."""
    state = _load(os.path.join(directory, "scalp_shadow_week.json"))
    live = sorted((r for r in state.get("signals", []) if isinstance(r, dict)
                   and r.get("selection_version") == SELECTION_VERSION
                   and r.get("status") == "OPEN"
                   and isinstance(r.get("ts"), (int, float))
                   and 0 <= now - r["ts"] / 1000 <= HORIZON_S),
                  key=lambda r: r["ts"], reverse=True)
    result = []
    for r in live:
        if r.get("symbol") not in result and isinstance(r.get("symbol"), str):
            result.append(r["symbol"])
        if len(result) >= max_extra:
            break
    return result


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



def _diagnostic(strategy, symbol, bars, data_ready, daily_range=None):
    """Explain every sampled coin, including those without enough 15m data.

    Read-only: diagnostics do not relax or alter paper-trade entry conditions.
    """
    last_time = _dt(bars[-1][0]) if bars else "—"
    observed = (sum(b[0] >= bars[-1][0] - 55 * STEP_MS for b in bars)
                if bars else 0)
    coverage = "%s/56" % observed
    if not data_ready:
        return [symbol, "EKSİK", coverage, "—", "—", "VERİ BEKLENİYOR",
                "0/0", "52/56 gerçek 15m mum ve güncel kapanış gerekli",
                "YOK", last_time]

    old, prev, last = bars[-3:]
    ma = sum(r[4] for r in bars[-50:]) / 50
    prior_ma = sum(r[4] for r in bars[-51:-1]) / 50
    trend = "YUKARI" if ma > prior_ma else "AŞAĞI" if ma < prior_ma else "YATAY"

    if strategy == "PULLBACK":
        if trend == "YATAY":
            return [symbol, "HAZIR", coverage, trend, "—",
                    "TREND BEKLENİYOR", "0/7", "MA50 eğimi yatay",
                    "YOK", last_time]
        if trend == "YUKARI":
            direction = "LONG"
            checks = [
                ("MA50 yükselişi", True),
                ("Fiyat MA50 üstünde", last[4] > ma),
                ("Önceki mum düşüş", prev[4] < prev[1]),
                ("Son mum yükseliş", last[4] > last[1]),
                ("Bullish engulfing", last[1] <= prev[4] and last[4] >= prev[1]),
                ("MA50 yakınında dönüş", min(prev[3], last[3]) <= ma * 1.005),
                ("Önceki yapı MA50 üstü", old[4] > ma * 0.995),
            ]
        else:
            direction = "SHORT"
            checks = [
                ("MA50 düşüşü", True),
                ("Fiyat MA50 altında", last[4] < ma),
                ("Önceki mum yükseliş", prev[4] > prev[1]),
                ("Son mum düşüş", last[4] < last[1]),
                ("Bearish engulfing", last[1] >= prev[4] and last[4] <= prev[1]),
                ("MA50 yakınında dönüş", max(prev[2], last[2]) >= ma * 0.995),
                ("Önceki yapı MA50 altı", old[4] < ma * 1.005),
            ]
        missing = [name for name, ok in checks if not ok]
        confirmed = pullback(bars) is not None
        return [symbol, "HAZIR", coverage, trend, direction,
                "SİNYAL" if confirmed else "ONAY BEKLENİYOR",
                "%s/%s" % (len(checks) - len(missing), len(checks)),
                "; ".join(missing) if missing else "Tüm koşullar tamam",
                "VAR" if confirmed else "YOK", last_time]

    if daily_range is None:
        return [symbol, "HAZIR", coverage, trend, "—", "GÜNLÜK VERİ EKSİK",
                "0/4", "Dünün 96 gerçek 15m mumu gerekli",
                "YOK", last_time]
    high, low = daily_range
    prev2 = bars[-3]
    body = abs(prev[4] - prev[1]) / max(prev[2] - prev[3], 1e-15)
    body_ok = body >= 0.45
    up = prev2[4] <= high and prev[1] <= high and prev[4] > high
    down = prev2[4] >= low and prev[1] >= low and prev[4] < low
    direction = "YUKARI" if up else "AŞAĞI" if down else "—"
    retest = (last[3] <= high * 1.003 and last[4] > high
              and last[4] > last[1]) if up else (
                  last[2] >= low * 0.997 and last[4] < low
                  and last[4] < last[1]) if down else False
    checks = [
        ("Dünün aralığı", True),
        ("Kırılım mumu gövdesi", body_ok),
        ("15m aralık dışı kapanış", up or down),
        ("15m geri test ve dönüş", retest),
    ]
    missing = [name for name, ok in checks if not ok]
    confirmed = break_bounce(bars, daily_range) is not None
    status = ("SİNYAL" if confirmed else "KIRILIM BEKLENİYOR" if not (up or down)
              else "MUM ONAYI BEKLENİYOR" if not body_ok else "RETEST BEKLENİYOR")
    return [symbol, "HAZIR", coverage, trend, direction, status,
            "%s/%s" % (len(checks) - len(missing), len(checks)),
            "; ".join(missing) if missing else "Tüm koşullar tamam",
            "VAR" if confirmed else "YOK", last_time]


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
        # A complete candle-by-candle path is required to claim TIMEOUT.
        required = {signal["ts"] + i * STEP_MS
                    for i in range(1, HORIZON_S * 1000 // STEP_MS + 1)}
        observed = {bar[0] for bar in later}
        signal["status"] = "TIMEOUT" if required.issubset(observed) else "DATA_GAP"
        signal["resolved_at"] = int(now * 1000)


def _scan(data_dir, tickers, candles_get, now, obj, allowed_market=None):
    """Bounded 15m-only prospective scan. Pure research; nothing signed/REST."""
    groups = gainer_groups(tickers, allowed_market)
    ready = {k: [] for k in RANK_BOUNDS}
    recent = {}
    full_candles = {}
    # Read each exact ranked gainer cache once, reuse for yesterday's range.
    for symbol in set(groups["PULLBACK"] + groups["BREAK_BOUNCE"]):
        full_candles[symbol] = candles_get(symbol)
        recent[symbol] = _bars(full_candles[symbol], now)
    for strategy, symbols in groups.items():
        for symbol in symbols:
            bars = recent[symbol]
            if _ready(bars, now):
                ready[strategy].append(symbol)

    # Preserve legacy records locally but NEVER include those trials in
    # the new gainer-universe results/denominators.
    obj["selection_version"] = SELECTION_VERSION
    obj.setdefault("signals", [])
    if not obj.get("gainer_test_started"):
        obj["gainer_test_started"] = int(now)
    known = {(r["strategy"], r["symbol"], r["ts"]) for r in obj["signals"]
             if isinstance(r, dict) and r.get("selection_version") == SELECTION_VERSION
             and all(k in r for k in ("strategy", "symbol", "ts"))}
    # Update pending outcomes even if their symbol rotated off the rank list.
    pending = sorted((r for r in obj["signals"] if isinstance(r, dict)
                      and r.get("selection_version") == SELECTION_VERSION
                      and r.get("status") == "OPEN"),
                     key=lambda r: r.get("ts", 0), reverse=True)
    for r in pending[:200]:
        sym = r.get("symbol")
        if isinstance(sym, str) and sym not in recent:
            recent[sym] = _bars(candles_get(sym), now)
        _settle(r, recent.get(sym, []), now)

    diagnostics = {k: [] for k in RANK_BOUNDS}
    for strategy, symbols in groups.items():
        for symbol in symbols:
            bars = recent[symbol]
            data_ready = _ready(bars, now)
            daily_range = None
            if strategy == "BREAK_BOUNCE" and data_ready:
                # Existing on-disk cache + existing WS; never add REST calls.
                history_path = os.path.join(data_dir, "crash_research", "history_cache",
                                            symbol + ".json")
                try:
                    with open(history_path, encoding="utf-8") as fd:
                        history_rows = json.load(fd)
                except (OSError, ValueError, TypeError):
                    history_rows = []
                daily_range = previous_utc_range(
                    (history_rows if isinstance(history_rows, list) else [])
                    + (full_candles.get(symbol) or []), now)
            diagnostics[strategy].append(
                _diagnostic(strategy, symbol, bars, data_ready, daily_range))
            if not data_ready:
                continue
            entry_ts = bars[-1][0]
            result = (pullback(bars) if strategy == "PULLBACK"
                      else break_bounce(bars, daily_range))
            if result is None or (strategy, symbol, entry_ts) in known:
                continue
            side, note = result
            entry = bars[-1][4]
            obj["signals"].append({
                "selection_version": SELECTION_VERSION,
                "strategy": strategy, "symbol": symbol, "side": side,
                "ts": entry_ts, "entry": entry,
                "tp": entry * (1.01 if side == "LONG" else 0.99),
                "risk": entry * (0.99 if side == "LONG" else 1.01),
                "status": "OPEN", "resolved_at": None, "note": note,
            })
            known.add((strategy, symbol, entry_ts))
    # The 8-day local retention cap also applies to legacy experiments.
    obj["signals"] = [r for r in obj["signals"] if isinstance(r, dict)
                      and isinstance(r.get("ts"), (int, float))
                      and now - r["ts"] / 1000 < RETENTION_S][-2000:]
    obj["last_scan"] = int(now)
    obj["ranked_groups"] = groups
    obj["ready_groups"] = ready
    obj["sampled_coins"] = {k: list(v) for k, v in ready.items()}
    obj["available_count"] = {k: len(v) for k, v in ready.items()}
    obj["diagnostics"] = diagnostics
    return obj

def _tab_values(obj, strategy):
    signals = [s for s in obj["signals"] if s.get("strategy") == strategy
               and s.get("selection_version") == SELECTION_VERSION
               and obj["last_scan"] - s["ts"] / 1000 <= 7 * 86400]
    statuses = {k: sum(s.get("status") == k for s in signals)
                for k in ("OPEN", "TP", "RISK", "TIMEOUT", "AMBIGUOUS", "DATA_GAP")}
    resolved = statuses["TP"] + statuses["RISK"]
    rate = str(round(100 * statuses["TP"] / resolved, 1)) + "%" if resolved else "—"
    rows = [
        ["Yedi günlük sanal test", TABS[strategy]],
        ["Model", MODEL],
        ["Son tarama UTC", _dt(obj["last_scan"] * 1000)],
        ["24s en çok yükselen %s-%s. sıralar / veri hazır %s/%s" %
         (RANK_BOUNDS[strategy][0] + 1, RANK_BOUNDS[strategy][1],
          len(obj.get("ready_groups", {}).get(strategy, [])),
          len(obj.get("ranked_groups", {}).get(strategy, []))),
         ", ".join(obj.get("ranked_groups", {}).get(strategy, [])) or "YETERLİ YÜKSELEN YOK"],
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
    rows += [[""] * len(HEADERS) for _ in range(max(0, 9 + MAX_SHEET_SIGNALS - len(rows)))]
    # Keep original paper-result rows A1:J57 untouched; monitor below them.
    rows.extend([[""] * 10,
                 ["Takip edilen coinler / sinyal çıkmama nedenleri", TABS[strategy]],
                 DIAG_HEADERS])
    rows.extend(obj.get("diagnostics", {}).get(strategy, [])[:MAX_SYMBOLS])
    return rows + [[""] * 10 for _ in range(max(0, 80 - len(rows)))]


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
    data = [{"range": "'" + ws.title.replace("'", "''") + "'!A1:J80",
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

    def tick(self, tickers, candles_get, allowed_market=None):
        now = self.clock()
        if now < self.next_due or not self.lock.acquire(False):
            return
        self.next_due = now + INTERVAL_S
        snapshot = [dict(t) for t in tickers if isinstance(t, dict)]
        market_symbols = set(allowed_market) if allowed_market is not None else None
        def work():
            try:
                data = _scan(self.directory, snapshot, candles_get, now, _load(self.path),
                             market_symbols)
                _save(self.path, data)
                try:
                    if _publish(data):
                        self.logger("[SCALP SHADOW] Pullback Test / Break Bounce Test updated; "
                                    "Pullback_ready=%s/%s; Break_ready=%s/%s; paper_signals=%s; no REST" %
                                    (len(data["ready_groups"]["PULLBACK"]),
                                     len(data["ranked_groups"]["PULLBACK"]),
                                     len(data["ready_groups"]["BREAK_BOUNCE"]),
                                     len(data["ranked_groups"]["BREAK_BOUNCE"]),
                                     sum(r.get("selection_version") == SELECTION_VERSION
                                         for r in data["signals"])))
                    else:
                        self.logger("[SCALP SHADOW] local paper results saved; Sheets credentials missing")
                except Exception as exc:
                    self.logger("[SCALP SHADOW SHEETS] skipped: " + type(exc).__name__)
            except Exception as exc:
                self.logger("[SCALP SHADOW] skipped: " + type(exc).__name__)
            finally:
                self.lock.release()
        threading.Thread(target=work, name="scalp-shadow-week", daemon=True).start()
