"""Coinbase public spot 15m fallback for missing Binance USDT-M candles.

No API key. Only missing closed intervals, with a same-timestamp original
Binance adjacent candle anchor and <=1.5% close deviation. A product symbol
match is NOT sufficient to treat two exchange markets as interchangeable.
Independent persisted request budget and cooldown; never sends orders.
"""
import json
import math
import os
import time
from urllib.parse import quote

from crash_policy import supported_symbol
from crash_data_sources import STEP, tagged, valid_rows

MAX_CANDLES = 350
MAX_REQUESTS_PER_HOUR = 12
MIN_REQUEST_INTERVAL = 120
RETRY_SECONDS = 1800
MISSING_MARKET_RETRY_SECONDS = 24 * 3600
PRICE_MAX_DEVIATION = 0.015
API = "https://api.coinbase.com/api/v3/brokerage/market/products/"


class CoinbaseReference:
    def __init__(self, directory, log, get=None, clock=time.time):
        self.directory, self.log, self.get, self.clock = directory, log, get, clock
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, "coinbase_budget.json")
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.state = json.load(f)
            if not isinstance(self.state, dict) or not isinstance(self.state.get("requests"), list):
                raise ValueError("COINBASE_BUDGET_INVALID")
        else:
            self.state = {"requests": [], "next_http": 0, "pause_until": 0,
                          "next_symbol": {}, "unavailable": {}}

    def _save(self):
        path = self.path + ".tmp"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path, self.path)

    @staticmethod
    def _pair(symbol):
        if not supported_symbol(symbol) or not symbol.endswith("USDT"):
            return None
        base = symbol[:-4]
        if not base:
            return None
        # Do NOT translate 1000PEPEUSDT to PEPE-USD: contracts can
        # represent 1000x spot and falsely match the wrong instrument.
        return base + "-USD"

    def fill_missing(self, symbol, start, end, missing, binance_rows):
        pair = self._pair(symbol)
        if (not pair or start % STEP or end % STEP
                or end-start > 480*STEP or not isinstance(missing, list) or not missing):
            return []
        existing = {t: r for t, r in valid_rows(binance_rows, start, end).items()
                    if len(r) <= 12 or r[12] not in ("BYBIT_LINEAR", "COINBASE_SPOT")}
        wanted = sorted({t for t in missing if t in range(start,end,STEP) and t not in existing})
        if not wanted or not existing:
            return []  # No unanchored full-history replacement.
        first=wanted[0]
        left=first-STEP if first-STEP in existing else None
        # Limit request to <=350 candles including a Binance overlap.
        max_missing=MAX_CANDLES - (1 if left is not None else 0)
        stop=first+STEP
        while stop in wanted and (stop-first)//STEP < max_missing:
            stop += STEP
        right=stop if stop in existing and (stop-first)//STEP < MAX_CANDLES else None
        if left is None and right is None:
            self.log(f"[CRASH COINBASE SKIP] {symbol}; NO_ADJACENT_BINANCE_ANCHOR")
            return []
        request_start=left if left is not None else first
        request_end=stop + (STEP if right is not None else 0)
        if (request_end-request_start)//STEP > MAX_CANDLES:
            return []
        now=self.clock()
        self.state.setdefault("requests",[])
        self.state["requests"]=[t for t in self.state["requests"] if now-t<3600]
        self.state.setdefault("next_symbol",{})
        self.state.setdefault("unavailable",{})
        if (now < self.state.get("next_http",0)
                or now < self.state.get("pause_until",0)
                or now < self.state["next_symbol"].get(symbol,0)
                or now < self.state["unavailable"].get(pair,0)
                or len(self.state["requests"]) >= MAX_REQUESTS_PER_HOUR):
            return []
        self.state["requests"].append(now)
        self.state["next_http"]=now+MIN_REQUEST_INTERVAL
        self.state["next_symbol"][symbol]=now+RETRY_SECONDS
        self._save()  # Reserve before HTTP so redeploy does not reset limit.
        get=self.get
        if get is None:
            import requests
            get=requests.get
        try:
            response=get(API+quote(pair,safe="-"),
                params={"start":str(request_start//1000),
                        "end":str(request_end//1000),
                        "granularity":"FIFTEEN_MINUTE",
                        "limit":(request_end-request_start)//STEP},
                headers={}, timeout=(5,10), allow_redirects=False)
            try:
                code=response.status_code
                if code==404:
                    self.state["unavailable"][pair]=now+MISSING_MARKET_RETRY_SECONDS
                    self.log(f"[CRASH COINBASE MISSING] {symbol}; pair={pair}; cached_24h=1")
                    return []
                if code in (429,403):
                    retry=900
                    try:
                        retry=max(retry, min(86400,float(response.headers.get("Retry-After",0))))
                    except (ValueError,TypeError,AttributeError):
                        pass
                    self.state["pause_until"]=now+retry
                    self.log(f"[CRASH COINBASE THROTTLED] {symbol}; http={code}; wait_sn={int(retry)}")
                    return []
                if code!=200:
                    raise ValueError("HTTP_"+str(code))
                payload=response.json()
            finally:
                response.close()
            if not isinstance(payload,dict) or not isinstance(payload.get("candles"),list):
                raise ValueError("INVALID_COINBASE_RESPONSE")
            raw=payload["candles"]
            if len(raw) > (request_end-request_start)//STEP:
                raise ValueError("COINBASE_UNEXPECTED_COUNT")
            converted=[]
            for item in raw:
                if not isinstance(item,dict):
                    continue
                try:
                    t=int(item["start"])*1000
                    o,h,l,c,v=(float(item[k]) for k in ("open","high","low","close","volume"))
                    if (t % STEP or not request_start <= t < request_end
                            or any(not math.isfinite(x) for x in (o,h,l,c,v))
                            or min(o,h,l,c)<=0 or v<0 or l>min(o,c) or h<max(o,c)):
                        continue
                    # Public Coinbase volume is in base asset; quote turnover
                    # below is an ESTIMATE (base volume * close), not actual taker CVD.
                    converted.append(tagged(
                        [t,str(o),str(h),str(l),str(c),str(v),t+STEP-1,
                         str(v*c),0,0,0,0], "COINBASE_SPOT"))
                except (KeyError,ValueError,TypeError,OverflowError):
                    continue
            found=valid_rows(converted,request_start,request_end)
            anchors=[t for t in (left,right) if t is not None and t in found]
            if not anchors:
                raise ValueError("BINANCE_ANCHOR_MISSING")
            for t in anchors:
                bc=float(existing[t][4])
                cc=float(found[t][4])
                if bc<=0 or abs(cc/bc-1)>PRICE_MAX_DEVIATION:
                    raise ValueError("CROSS_EXCHANGE_PRICE_MISMATCH")
            output=[found[t] for t in wanted if first<=t<stop and t in found]
            self.state["next_symbol"][symbol]=now+60 if output else now+RETRY_SECONDS
            self.log(f"[CRASH COINBASE FILL] {symbol}; missing={len(wanted)}; "
                     f"accepted={len(output)}; anchor={len(anchors)}; source=COINBASE_SPOT")
            return output
        except Exception as error:
            self.log(f"[CRASH COINBASE WAIT] {symbol}; type={type(error).__name__}; "
                     f"code={str(error)[:48] if isinstance(error,ValueError) else 'TRANSPORT'}")
            return []
        finally:
            self._save()
