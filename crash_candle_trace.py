"""Read-only 15m candle audit: distinguish actual WS closes from cache merges."""
import json
import os
import threading
import time
from datetime import datetime, timezone

from crash_data_sources import valid_rows

STEP = 900_000
WINDOW = 480


def inspect_candles(as_of, ws_rows, history_rows, ws_observation=None):
    """Compare identical fixed slots without fetching candles or changing caches."""
    end = int(as_of * 1000) // STEP * STEP
    start = end - WINDOW * STEP
    latest = end - STEP
    ws = valid_rows(ws_rows, start, end)
    history = valid_rows(history_rows, start, end)
    merged = dict(history)
    merged.update(ws)  # Only timestamp PRESENCE is compared, never market prices.
    last_seen = None
    if isinstance(ws_observation, dict):
        try:
            last_seen = int(ws_observation['open_ms'])
        except (KeyError, TypeError, ValueError):
            pass
    missing = [t for t in range(start, end, STEP) if t not in merged]
    return {
        'last_closed_utc': datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(),
        'ws_cache_latest': latest in ws,
        'history_latest': latest in history,
        'engine_latest': latest in merged,
        'actual_ws_event_latest': None if last_seen is None else last_seen == latest,
        'actual_ws_event_open_ms': last_seen,
        'ws_cache_count': len(ws),
        'history_count': len(history),
        'engine_count': len(merged),
        'recent_4h_count': sum(t in merged for t in range(end - 16 * STEP, end, STEP)),
        'missing_count': len(missing),
        'first_missing_open_ms': missing[0] if missing else None,
    }


class CandleAudit:
    """Five-minute, nonblocking audit of the dashboard's top-20 loser universe."""
    def __init__(self, directory, logger):
        self.directory = directory
        self.logger = logger
        self.next_due = 0.0
        self.lock = threading.Lock()

    def tick(self, tickers, ws_cache, now=None):
        now = time.time() if now is None else float(now)
        # Avoid reporting a just-closed 15m candle as missing before WS delivery.
        if now < self.next_due or (now % 900) < 20:
            return
        if not self.lock.acquire(blocking=False):
            return
        ranked = []
        for ticker in tickers if isinstance(tickers, list) else []:
            try:
                symbol = ticker['symbol']
                if (not isinstance(symbol, str) or not symbol.endswith('USDT')
                        or not symbol.isascii() or not symbol.isalnum()):
                    continue
                change = float(ticker['priceChangePercent'])
                if not __import__('math').isfinite(change):
                    continue
                ranked.append((change, symbol))
            except (TypeError, ValueError, KeyError):
                continue
        ranked.sort()
        symbols = [symbol for _, symbol in ranked[:20]]
        self.next_due = now + 300

        def worker():
            try:
                gaps = []
                source_disagreements = []
                latest_ok = 0
                actual_ws_seen = 0
                for symbol in symbols:
                    path = os.path.join(self.directory, 'history_cache', symbol + '.json')
                    try:
                        with open(path, encoding='utf-8') as handle:
                            saved = json.load(handle)
                    except (OSError, ValueError):
                        saved = []
                    observation = getattr(ws_cache, 'last_ws_observation', lambda s: None)(symbol)
                    status = inspect_candles(now, ws_cache.candles(symbol), saved, observation)
                    latest_ok += int(status['engine_latest'])
                    actual_ws_seen += int(status['actual_ws_event_latest'] is True)
                    if (not status['engine_latest'] or status['engine_count'] < 478
                            or status['recent_4h_count'] < 14):
                        gaps.append((symbol, status))
                    elif status['ws_cache_latest'] != status['history_latest']:
                        source_disagreements.append((symbol, status))
                close_utc = datetime.fromtimestamp((int(now*1000)//STEP*STEP)/1000, timezone.utc).isoformat()
                self.logger(f'[CRASH 15M AUDIT] expected_close_utc={close_utc}; '
                            f'symbols={len(symbols)}; engine_latest_present={latest_ok}; '
                            f'actual_ws_close_seen={actual_ws_seen}; engine_gaps={len(gaps)}; '
                            f'source_disagreements={len(source_disagreements)}; rest_requests=0')
                for category, group, cap in (('GAP', gaps, 10), ('SOURCE', source_disagreements, 5)):
                    for symbol, st in group[:cap]:
                        self.logger(f'[CRASH 15M AUDIT {category}] symbol={symbol}; '
                                    f'last_closed_utc={st["last_closed_utc"]}; '
                                    f'actual_ws_event_latest={st["actual_ws_event_latest"]}; '
                                    f'actual_ws_event_open_ms={st["actual_ws_event_open_ms"]}; '
                                    f'ws_cache_latest={int(st["ws_cache_latest"])}; '
                                    f'history_latest={int(st["history_latest"])}; '
                                    f'engine_latest={int(st["engine_latest"])}; '
                                    f'engine_count={st["engine_count"]}/480; '
                                    f'recent_4h={st["recent_4h_count"]}/16; '
                                    f'missing={st["missing_count"]}; '
                                    f'first_missing_open_ms={st["first_missing_open_ms"]}')
            except Exception as error:
                self.logger(f'[CRASH 15M AUDIT ERROR] {type(error).__name__}')
            finally:
                self.lock.release()
        threading.Thread(target=worker, name='crash-15m-audit', daemon=True).start()
