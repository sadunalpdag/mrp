"""15-minute BTC paper-only Sheets scheduler for existing ema.py worker.

Runs at minute 2, 17, 32, 47 UTC to allow closed CoinBeacon candles.
No broker, Telegram, or TradingView imports. Disabled unless explicitly enabled
by BTC_AI_SCALP_SHEETS_ENABLED=1 in the existing Render worker environment.
"""
import logging
import threading
import time
from datetime import datetime, timezone

LOG = logging.getLogger("btc_ai_scalp_scheduler")
_STARTED = False
_LOCK = threading.Lock()


def _loop():
    from btc_ai_scalp_sheets import run
    while True:
        now = time.time()
        # Every 900 seconds, 120 seconds after the corresponding candle close.
        next_run = (int((now - 120) // 900) + 1) * 900 + 120
        time.sleep(max(1, next_run - now))
        try:
            run()
            LOG.info("BTC AI scalp Sheets cycle completed at %s", datetime.now(timezone.utc).isoformat())
        except Exception:
            LOG.exception("BTC AI scalp Sheets cycle failed; other bots unaffected")


def start():
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        worker = threading.Thread(target=_loop, daemon=True, name="BtcAiScalpSheets")
        worker.start()
        _STARTED = True
