"""Paced Coinbase Exchange spot 15m candles as a last-resort gap source.

No credentials, account endpoints, or order operations. Coinbase BTC-USD is
NOT Binance BTCUSDT perpetual: every accepted bar is tagged COINBASE_SPOT
and requires a nearby Binance price anchor within 3%.
"""
import math
import time
from datetime import datetime, timezone
from crash_policy import supported_symbol
from crash_data_sources import STEP, valid_rows, tagged

MAX_REQUESTS_PER_HOUR = 12
MIN_REQUEST_GAP_SECONDS = 180
PER_SYMBOL_RETRY_SECONDS = 1800
UNSUPPORTED_RETRY_SECONDS = 86400
MAX_BARS = 299


class CoinbaseReference:
    def __init__(self, directory, log, get=None, clock=time.time):
        self.directory, self.log, self.get, self.clock = directory, log, get, clock
        self.next_symbol = {}
        self.requests = []
        self.next_request = 0
        self.global_cooldown = 0

    def fill_missing(self, symbol, start, end, missing, binance_rows):
        if (not supported_symbol(symbol) or not symbol.endswith("USDT")
                or start % STEP or end % STEP or not missing
                or end-start > 480*STEP):
            return []
        existing = {t:r for t,r in valid_rows(binance_rows,start,end).items()
                    if len(r)<=12 or r[12] not in ("BYBIT_LINEAR","COINBASE_SPOT")}
        wanted = sorted(t for t in set(missing)
                        if start<=t<end and t%STEP==0 and t not in existing)
        if not wanted or not existing:
            return []
        first = wanted[0]
        stop = first+STEP
        while stop in wanted and stop-first < (MAX_BARS-1)*STEP:
            stop+=STEP
        left = first-STEP if first-STEP in existing else None
        right = stop if stop in existing else None
        if left is None and right is None:
            self.log(f"[CRASH COINBASE SKIP] {symbol}; NO_ADJACENT_BINANCE_ANCHOR")
            return []
        request_start = left if left is not None else first
        request_end = stop+(STEP if right is not None else 0)
        if (request_end-request_start)//STEP > MAX_BARS:
            return []
        now=self.clock()
        self.requests=[t for t in self.requests if now-t<3600]
        if (now<self.next_symbol.get(symbol,0) or now<self.next_request
                or now<self.global_cooldown or len(self.requests)>=MAX_REQUESTS_PER_HOUR):
            return []
        self.requests.append(now)
        self.next_request=now+MIN_REQUEST_GAP_SECONDS
        self.next_symbol[symbol]=now+PER_SYMBOL_RETRY_SECONDS
        product=symbol[:-4]+"-USD"
        get=self.get
        if get is None:
            import requests
            get=requests.get
        try:
            params={
                "start":datetime.fromtimestamp(request_start/1000,timezone.utc).isoformat(),
                "end":datetime.fromtimestamp(request_end/1000,timezone.utc).isoformat(),
                "granularity":900}
            response=get("https://api.exchange.coinbase.com/products/"+product+"/candles",
                params=params,headers={"Accept":"application/json"},
                timeout=(5,10),allow_redirects=False)
            try:
                status=response.status_code
                if status in (404,400):
                    self.next_symbol[symbol]=now+UNSUPPORTED_RETRY_SECONDS
                    self.log(f"[CRASH COINBASE UNAVAILABLE] {symbol}; HTTP_{status}")
                    return []
                if status==429:
                    retry=response.headers.get("Retry-After","300")
                    try: delay=max(300,min(3600,int(float(retry))))
                    except (TypeError,ValueError): delay=600
                    self.global_cooldown=now+delay
                    self.log(f"[CRASH COINBASE LIMIT] HTTP_429; retry_seconds={delay}")
                    return []
                if status!=200:
                    raise ValueError("HTTP_"+str(status))
                raw=response.json()
            finally:
                response.close()
            if not isinstance(raw,list) or len(raw)>MAX_BARS:
                raise ValueError("INVALID_COINBASE_CANDLES")
            converted=[]
            for item in raw:
                if not isinstance(item,list) or len(item)<6:
                    continue
                try:
                    t=int(float(item[0])*1000)
                    low,high,opening,closing,base_volume=map(float,item[1:6])
                    if (t%STEP or not request_start<=t<request_end
                            or not all(math.isfinite(x) for x in
                                       (low,high,opening,closing,base_volume))
                            or min(low,high,opening,closing)<=0 or base_volume<0
                            or low>min(opening,closing) or high<max(opening,closing)):
                        continue
                    quote_volume=base_volume*closing
                    converted.append(tagged([
                        t,str(opening),str(high),str(low),str(closing),
                        str(base_volume),t+STEP-1,str(quote_volume),0,0,0,0],
                        "COINBASE_SPOT"))
                except (TypeError,ValueError,OverflowError):
                    continue
            found=valid_rows(converted,request_start,request_end)
            overlap=[t for t in (left,right) if t is not None and t in found]
            if not overlap:
                raise ValueError("BINANCE_ANCHOR_MISSING")
            for t in overlap:
                bc=float(existing[t][4]);cc=float(found[t][4])
                if bc<=0 or abs(cc/bc-1)>0.03:
                    raise ValueError("CROSS_EXCHANGE_PRICE_MISMATCH")
            output=[found[t] for t in wanted if first<=t<stop and t in found]
            self.log(f"[CRASH COINBASE FILL] {symbol}; missing={len(wanted)}; "
                     f"filled={len(output)}; anchor={len(overlap)}; SPOT_RESEARCH_ONLY")
            return output
        except Exception as error:
            self.log(f"[CRASH COINBASE WAIT] {symbol}; type={type(error).__name__}; "
                     f"code={str(error)[:40] if isinstance(error,ValueError) else 'TRANSPORT'}")
            return []
