"""Read-only BTCUSDT candles from CoinBeacon's Binance-Futures OHLCV cache.

No Binance REST fallback. Uses the same quota-aware MarketClient as the existing
CoinBeacon scanners (preserves their 400-credit reserve). The API's 'to'
parameter pages BACKWARDS by candle open time, inclusive.
"""
import logging
import math
import os
import threading
import time

import requests

LOG = logging.getLogger("btc_coinbeacon_ohlcv")
FIFTEEN = 900_000
HOUR = 3_600_000
PAGE_MAX = 1500
RECENT_REQUIRED = 1000  # 1h ADX and 4h EMA history, plus paper 15m monitoring.
_LOCK = threading.RLock()
_CLIENT = None
_CACHE = {}
_ALERT_LOCK = threading.Lock()
_ALERT_ACTIVE = False
_LAST_ALERT_TS = 0.0
_ALERT_INTERVAL_SECONDS = 3600


def _telegram_status(message):
    """Send only operational status; never include credentials or request URLs."""
    token = os.getenv("BOT_TOKEN", "").strip()
    chat = os.getenv("CHAT_ID", "").strip()
    if not token or not chat:
        LOG.warning("[BTC COINBEACON ALERT] Telegram is not configured")
        return
    try:
        response = requests.post(
            "https://api.telegram.org/bot" + token + "/sendMessage",
            data={"chat_id": chat, "text": message[:1500]}, timeout=10)
        response.raise_for_status()
        if not response.json().get("ok"):
            raise RuntimeError("TELEGRAM_NOT_OK")
    except Exception as exc:
        LOG.warning("[BTC COINBEACON ALERT] send failed type=%s", type(exc).__name__)


def _report_failure(reason):
    global _ALERT_ACTIVE, _LAST_ALERT_TS
    now = time.time()
    with _ALERT_LOCK:
        should_send = not _ALERT_ACTIVE or now - _LAST_ALERT_TS >= _ALERT_INTERVAL_SECONDS
        _ALERT_ACTIVE = True
        if should_send:
            _LAST_ALERT_TS = now
    if should_send:
        _telegram_status(
            "❌ BTC COINBEACON VERİ HATASI\n"
            "Kaynak: CoinBeacon BTCUSDT 15m\n"
            "Neden: " + str(reason)[:100] + "\n"
            "Gemini BTC analizi bu çevrimde güncellenemeyebilir.\n"
            "Crash Bounce işlemleri etkilenmez.\n"
            "Tekrar eden hatalar için uyarı en fazla saatte bir gönderilir.")


def _report_recovered():
    global _ALERT_ACTIVE
    with _ALERT_LOCK:
        was_active = _ALERT_ACTIVE
        _ALERT_ACTIVE = False
    if was_active:
        _telegram_status("✅ BTC COINBEACON BAĞLANTISI DÜZELDİ\n"
                         "BTCUSDT 15m mum verisi yeniden başarıyla alındı.")



class CandleUnavailable(RuntimeError):
    pass


def _client():
    global _CLIENT
    if _CLIENT is None:
        from btc_coinbeacon_client import MarketClient
        _CLIENT = MarketClient(requests)
        if not _CLIENT.key:
            _CLIENT.key = os.getenv("COINBEACON_API_KEY", "").strip()
    if not _CLIENT.key:
        raise CandleUnavailable("COINBEACON_API_KEY_MISSING")
    return _CLIENT


def _validate_bar(raw, before_ms):
    """Only completed, properly time-aligned, finite Binance-futures 15m bars."""
    try:
        opened = int(raw["openTime"])
        close_time = int(raw["closeTime"])
        prices = [float(raw[key]) for key in ("open", "high", "low", "close", "volume")]
        op, hi, lo, close, volume = prices
        if (opened % FIFTEEN != 0 or opened + FIFTEEN - 1 != close_time
                or opened + FIFTEEN > before_ms
                or not all(math.isfinite(x) for x in prices)
                or min(op, hi, lo, close) <= 0 or volume < 0
                or hi < max(op, close) or lo > min(op, close)):
            return None
        return [opened, op, hi, lo, close, volume]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _fetch_closed_15m_unmonitored(end_ms, count, priority_checker=lambda: False):
    """Fetch exactly count consecutive closed candles up to exclusive end_ms.

    If CoinBeacon's historical cache cannot supply enough candles, return an
    explicit failure, never use Binance directly or silently shorten history.
    One 1000-bar fetch normally serves BOTH hourly observer and paper strategy.
    """
    if end_ms % FIFTEEN or not 0 < count <= 2880:
        raise ValueError("INVALID_BTC_CANDLE_WINDOW")
    cache_key = (end_ms, count)
    with _LOCK:
        if cache_key in _CACHE:
            return [list(b) for b in _CACHE[cache_key]]
        if priority_checker():
            raise CandleUnavailable("LIVE_CRASH_PRIORITY")
        client = _client()
        seen = {}
        cursor = end_ms - 1
        pages = 0
        while len(seen) < count:
            if priority_checker():
                raise CandleUnavailable("LIVE_CRASH_PRIORITY")
            pages += 1
            if pages > 3:
                raise CandleUnavailable("COINBEACON_BACKFILL_PAGE_CAP")
            limit = min(PAGE_MAX, count - len(seen))
            data = client.cb(
                "/public/ohlcv/binance-futures/BTCUSDT/15m", 1,
                {"to": cursor, "limit": limit})
            if not isinstance(data, dict):
                raise CandleUnavailable("COINBEACON_CREDIT_OR_HTTP_UNAVAILABLE")
            if (data.get("exchange") != "binance-futures"
                    or data.get("symbol") != "BTCUSDT"
                    or data.get("timeframe") != "15m"):
                raise CandleUnavailable("COINBEACON_WRONG_MARKET")
            rows = data.get("bars")
            if not isinstance(rows, list) or not rows:
                raise CandleUnavailable("COINBEACON_HISTORY_EMPTY")
            valid = []
            for raw in rows:
                bar = _validate_bar(raw, end_ms)
                if bar and bar[0] <= cursor:
                    valid.append(bar)
                    seen[bar[0]] = bar
            if not valid:
                raise CandleUnavailable("COINBEACON_HISTORY_NOT_CLOSED")
            oldest = min(b[0] for b in valid)
            if oldest >= cursor:
                raise CandleUnavailable("COINBEACON_BACKFILL_NO_PROGRESS")
            cursor = oldest - 1  # Inclusive API 'to'; never fetch the same page.
            if len(valid) < limit and len(seen) < count:
                raise CandleUnavailable("COINBEACON_HISTORY_TOO_SHALLOW")
        first = end_ms - count * FIFTEEN
        expected = list(range(first, end_ms, FIFTEEN))
        if any(t not in seen for t in expected):
            raise CandleUnavailable("COINBEACON_15M_GAP_OR_STALE")
        output = [seen[t] for t in expected]
        # Avoid holding 2,880 candles for each hour in memory.
        _CACHE.clear()
        _CACHE[cache_key] = output
        LOG.info("[BTC COINBEACON OHLCV] closed_15m=%d pages=%d last_close=%d",
                 len(output), pages, end_ms)
        return [list(b) for b in output]



def fetch_closed_15m(end_ms, count, priority_checker=lambda: False):
    """Notify Telegram on actual CoinBeacon/data failures, not Crash priority waits."""
    try:
        bars = _fetch_closed_15m_unmonitored(end_ms, count, priority_checker)
    except CandleUnavailable as exc:
        reason = str(exc)
        if reason != "LIVE_CRASH_PRIORITY":
            _report_failure(reason)
        raise
    except Exception as exc:
        _report_failure("UNEXPECTED_" + type(exc).__name__)
        raise
    _report_recovered()
    return bars


def aggregate(bars, timeframe, before_ms):
    """Derive closed 1h/4h OHLCV from CoinBeacon 15m without extra requests."""
    if timeframe not in ("1h", "4h"):
        raise ValueError("BTC_AGGREGATE_UNSUPPORTED_TF")
    interval = HOUR if timeframe == "1h" else 4 * HOUR
    pieces = interval // FIFTEEN
    groups = {}
    for bar in bars:
        t = int(bar[0])
        start = (t // interval) * interval
        if t + FIFTEEN <= before_ms:
            groups.setdefault(start, []).append(bar)
    out = []
    for start, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda b: b[0])
        if (len(rows) != pieces
                or any(b[0] != start + i * FIFTEEN for i, b in enumerate(rows))
                or start + interval > before_ms):
            continue
        out.append([start, rows[0][1], max(b[2] for b in rows),
                    min(b[3] for b in rows), rows[-1][4],
                    sum(b[5] for b in rows)])
    if not out:
        raise CandleUnavailable("COINBEACON_AGGREGATE_NO_COMPLETE_BARS")
    return out
