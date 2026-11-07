"""Urgent, read-only OKX USDT linear swap last-closed 15m candle fallback.

Never infer an open bar is closed. Require a validated adjacent Binance
candle at the identical timestamp for a <=0.5% market-parity check.
"""
import math
import time

from crash_data_sources import STEP, tagged, valid_rows
from crash_policy import supported_symbol

MAX_PER_HOUR = 36
MIN_GAP_SECONDS = 5
RETRY_SECONDS = 60
PARITY_LIMIT = 0.005


class OKXLastClosed:
    def __init__(self, log, get=None, clock=time.time):
        self.log, self.get, self.clock = log, get, clock
        self.attempts = []
        self.next_request = 0
        self.retry = {}
        self.cooldown_until = 0
        self.contracts = {}

    def _request(self, path, params):
        get = self.get
        if get is None:
            import requests
            get = requests.get
        response = get("https://www.okx.com" + path, params=params,
                       headers={}, timeout=(4, 7), allow_redirects=False)
        try:
            if response.status_code in (418, 429):
                self.cooldown_until = self.clock() + 300
                return None
            if response.status_code != 200:
                return None
            body = response.json()
            return body if isinstance(body, dict) and body.get("code") == "0" else None
        finally:
            response.close()

    def fill_last(self, symbol, start, end, binance_rows):
        if (not supported_symbol(symbol) or not symbol.endswith("USDT")
                or start % STEP or end % STEP or end-start > 480*STEP
                or end-start < 2*STEP):
            return []
        latest = end-STEP
        anchor = end-2*STEP
        known = valid_rows(binance_rows, start, end)
        base = known.get(anchor)
        if (base is None or latest in known or
                len(base) > 12 and base[12] in
                ("BYBIT_LINEAR", "COINBASE_SPOT", "OKX_SWAP")):
            return []
        now = self.clock()
        self.attempts = [t for t in self.attempts if now-t < 3600]
        if (now < self.retry.get(symbol, 0) or now < self.next_request
                or now < self.cooldown_until or len(self.attempts) >= MAX_PER_HOUR
                or end > int(now*1000)//STEP*STEP):
            return []
        self.attempts.append(now)
        self.next_request = now + MIN_GAP_SECONDS
        self.retry[symbol] = now + RETRY_SECONDS
        instrument = symbol[:-4] + "-USDT-SWAP"
        try:
            if instrument not in self.contracts or now > self.contracts[instrument][1]:
                info = self._request("/api/v5/public/instruments",
                                     {"instType": "SWAP", "instId": instrument})
                data = info.get("data") if info else None
                is_valid = (isinstance(data, list) and len(data) == 1
                            and data[0].get("instId") == instrument
                            and data[0].get("instType") == "SWAP"
                            and data[0].get("settleCcy") == "USDT"
                            and data[0].get("ctType") == "linear"
                            and data[0].get("state") == "live")
                self.contracts[instrument] = (is_valid, now+(3600 if is_valid else 86400))
            if not self.contracts[instrument][0]:
                self.log(f"[CRASH OKX SKIP] {symbol}; reason=NON_MATCHING_SWAP")
                return []
            result = self._request("/api/v5/market/candles",
                                   {"instId": instrument, "bar": "15m", "limit": "5"})
            records = result.get("data") if result else None
            if not isinstance(records, list) or len(records) > 5:
                return []
            converted = []
            for item in records:
                if not isinstance(item, list) or len(item) < 9 or str(item[8]) != "1":
                    continue
                try:
                    t = int(item[0])
                    if t not in (anchor, latest):
                        continue
                    o, h, l, c = map(float, item[1:5])
                    v, q = float(item[6]), float(item[7])
                    if (not all(math.isfinite(x) for x in (o,h,l,c,v,q))
                            or min(o,h,l,c) <= 0 or v < 0 or q < 0
                            or l > min(o,c) or h < max(o,c)):
                        continue
                    converted.append(tagged([
                        t, item[1], item[2], item[3], item[4], item[6],
                        t+STEP-1, item[7], 0, 0, 0, 0], "OKX_SWAP"))
                except (ValueError, TypeError, OverflowError):
                    continue
            found = valid_rows(converted, anchor, end)
            if anchor not in found or latest not in found:
                return []
            if abs(float(found[anchor][4])/float(base[4]) - 1) > PARITY_LIMIT:
                self.log(f"[CRASH OKX SKIP] {symbol}; reason=PRICE_PARITY")
                return []
            self.log(f"[CRASH OKX LAST CLOSED] {symbol}; slot={latest}; "
                     "source=OKX_SWAP; confirmed=1; anchor_parity=1")
            return [found[latest]]
        except Exception as error:
            self.log(f"[CRASH OKX WAIT] {symbol}; reason={type(error).__name__}")
            return []
