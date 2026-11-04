"""WS-only, read-only market feed for a future crypto Day Trader adapter.

It accepts ONLY raw Binance Futures CLOSED 15m kline events via submit().
No REST, no account data, no trading or Telegram activity. It never imports
ema.py, so problems here cannot affect V2 entry or TP decisions.
"""
import json
import math
import os
import queue
import re
import threading
import time
from decimal import Decimal, InvalidOperation

_STEP_MS = 900_000
_SYMBOL = re.compile(r"^[A-Z0-9]{2,30}USDT$", re.ASCII)
_SCHEMA = "crash_day_trader_ws_v1"


class DayTraderWsFeed:
    def __init__(self, directory, log=None, clock=time.time, max_bars=960):
        if not 16 <= max_bars <= 3000:
            raise ValueError("invalid max_bars")
        self.directory = directory
        self.log = log or (lambda message: None)
        self.clock = clock
        self.max_bars = max_bars
        self.events = queue.Queue(maxsize=2048)
        self.thread = None
        self._last_overflow_log = 0.0

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        os.makedirs(self.directory, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run, name="day-trader-ws-feed", daemon=True
        )
        self.thread.start()

    def submit(self, symbol, row):
        """Only call from Binance's actual closed kline WebSocket event handler.

        A row is the existing crash WS kline shape: [start_ms, O,H,L,C,V,
        close_ms, ...]. Reject future, stale, unaligned, invalid, or open bars.
        Never fabricate missing timestamps.
        """
        if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
            return False
        try:
            if len(row) < 7:
                return False
            start, finish = int(row[0]), int(row[6])
            now = int(self.clock() * 1000)
            if (start % _STEP_MS or finish != start + _STEP_MS - 1
                    or not 0 <= now - finish <= 30 * 60 * 1000):
                return False
            o, h, l, c, v = [Decimal(str(row[i])) for i in range(1, 6)]
            if (not all(x.is_finite() for x in (o, h, l, c, v))
                    or min(o, h, l, c) <= 0 or v < 0
                    or h < max(o, c) or l > min(o, c) or h < l):
                return False
            bar = {
                "open_time": start,
                "open": str(o),
                "high": str(h),
                "low": str(l),
                "close": str(c),
                "volume": str(v),
                "close_time": finish,
            }
        except (TypeError, ValueError, IndexError, InvalidOperation, OverflowError):
            return False
        try:
            self.events.put_nowait((symbol, bar))
        except queue.Full:
            now_s = self.clock()
            if now_s - self._last_overflow_log > 60:
                self._last_overflow_log = now_s
                self.log("[DAY TRADER WS DROP] queue full; research-only event lost")
            return False
        return True

    def _path(self, symbol):
        # Symbols are validated before reaching this point; no arbitrary paths.
        return os.path.join(self.directory, symbol + ".json")

    def _persist(self, symbol, bar):
        path = self._path(symbol)
        previous = {}
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
            if (doc.get("schema") != _SCHEMA or doc.get("symbol") != symbol
                    or doc.get("source") != "BINANCE_USDT_M_WEBSOCKET"):
                raise ValueError("unrecognized existing feed file")
            for item in doc.get("candles", []):
                previous[int(item["open_time"])] = item
        except FileNotFoundError:
            pass
        previous[bar["open_time"]] = bar
        times = sorted(previous)[-self.max_bars:]
        candles = [previous[t] for t in times]
        missing = sum(max(0, (b - a) // _STEP_MS - 1)
                      for a, b in zip(times, times[1:]))
        doc = {
            "schema": _SCHEMA,
            "symbol": symbol,
            "interval": "15m",
            "source": "BINANCE_USDT_M_WEBSOCKET",
            "closed_only": True,
            "updated_at": self.clock(),
            "candle_count": len(candles),
            "gap_slots": missing,
            "candles": candles,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _run(self):
        while True:
            symbol, bar = self.events.get()
            try:
                self._persist(symbol, bar)
            except Exception as error:
                self.log("[DAY TRADER WS FILE ERR] %s; %s" % (
                    symbol, type(error).__name__))
            finally:
                self.events.task_done()
