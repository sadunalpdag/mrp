"""Closed USDT-M 15m: WS/cache -> Binance archive -> Bybit -> Coinbase -> Binance REST.
Vendor calls are serialized by HistoryCollector. Alternate venues are limited
to missing timestamps and source-tagged; original Binance bars have priority.
"""
import csv
import hashlib
import io
import json
import math
import os
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from crash_policy import supported_symbol

STEP = 900000
DAY = 86400000
REST_MAX_CANDLES = 192  # Never request all 480 candles through Binance REST.
BINANCE_DATA_REST_MAX_HOUR = 12  # Research only, other Binance calls need their own guard.
BINANCE_DATA_REST_GAP = 300      # >=5m between historical-candle requests.


def valid_rows(rows, start, end):
    result = {}
    for row in rows if isinstance(rows, list) else []:
        try:
            t = int(row[0]); o,h,l,c,v = map(float,row[1:6])
            if (start <= t < end and t % STEP == 0 and int(row[6]) == t+STEP-1
                    and all(math.isfinite(x) for x in (o,h,l,c,v))
                    and min(o,h,l,c)>0 and v>=0 and l<=min(o,c)<=max(o,c)<=h):
                result[t] = list(row)
        except (ValueError, TypeError, IndexError, OverflowError):
            continue
    return result


def tagged(row, source):
    r = list(row)
    while len(r) < 12:
        r.append(None)
    return r[:12]+[source]


def source_counts(rows):
    return dict(Counter(r[12] if len(r)>12 else 'BINANCE_CACHE_WS' for r in rows))


class DataSources:
    def __init__(self, directory, rest, blocked, log, get=None, clock=time.time,
                 bybit=None, coinbase=None):
        self.directory,self.rest,self.blocked,self.log=directory,rest,blocked,log
        self.get,self.clock=get,clock
        self.retry={}
        self.binance_recheck={}
        self.bybit,self.coinbase=bybit,coinbase
        self.next_http=0
        os.makedirs(directory,exist_ok=True)
        self.budget_path=os.path.join(directory,"binance_data_rest_budget.json")
        self.binance_budget=self._read(self.budget_path,
            {"requests":[],"next_rest":0})
        if (not isinstance(self.binance_budget,dict)
                or not isinstance(self.binance_budget.get("requests"),list)):
            raise ValueError("BINANCE_HISTORY_BUDGET_INVALID")

    @staticmethod
    def _read(path, default):
        try:
            with open(path,encoding='utf-8') as f:return json.load(f)
        except (OSError,ValueError):return default

    @staticmethod
    def _write(path, value):
        with open(path+'.tmp','w',encoding='utf-8') as f:
            json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(path+'.tmp',path)

    def _download(self,url):
        delay=self.next_http-self.clock()
        if delay>0:time.sleep(min(delay,1))
        self.next_http=self.clock()+1
        get=self.get
        if get is None:
            import requests
            get=requests.get
        r=get(url,headers={},
              timeout=(5,10),allow_redirects=False,stream=True)
        try:
            if r.status_code!=200:
                raise ValueError('HTTP_'+str(r.status_code))
            data=bytearray()
            for part in r.iter_content(65536):
                data.extend(part)
                if len(data)>4*1024*1024:raise ValueError('RESPONSE_TOO_LARGE')
            return bytes(data)
        finally:r.close()

    def archive(self,symbol,day):
        date=datetime.fromtimestamp(day/1000,timezone.utc).date().isoformat()
        name=f'{symbol}-15m-{date}'
        path=os.path.join(self.directory,name+'.json')
        saved=self._read(path,None)
        if saved is not None:return saved
        if self.clock()<self.retry.get(name,0):return []
        self.retry[name]=self.clock()+3600
        url=f'https://data.binance.vision/data/futures/um/daily/klines/{symbol}/15m/{name}.zip'
        try:
            content=self._download(url)
            checksum=self._download(url+'.CHECKSUM').decode('ascii').split()[0]
            if len(checksum)!=64 or hashlib.sha256(content).hexdigest()!=checksum:
                raise ValueError('ARCHIVE_CHECKSUM_MISMATCH')
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                members=z.infolist()
                if len(members)!=1 or members[0].file_size>4*1024*1024:
                    raise ValueError('INVALID_ARCHIVE_SIZE')
                reader=csv.reader(io.StringIO(z.read(members[0]).decode('utf-8-sig')))
                rows=[]
                for row in reader:
                    if row and row[0] in ('open_time','open time'):continue
                    rows.append(row)
            rows=[tagged(r,'BINANCE_ARCHIVE') for r in valid_rows(rows,day,day+DAY).values()]
            if not rows:raise ValueError('EMPTY_ARCHIVE')
            self._write(path,rows)
            return rows
        except Exception as error:
            # Never log URLs, headers or arbitrary remote response text.
            code=str(error) if isinstance(error,ValueError) else type(error).__name__
            self.log(f'[CRASH DATA ARCHIVE WAIT] {symbol}; day={date}; code={code[:80]}')
            return []

    def _reserve_binance_rest(self):
        """Persist the historical data-only REST budget before each request.

        Does not replace Binance's shared public/private-IP limiter in ema.py.
        """
        now=self.clock()
        requests=[t for t in self.binance_budget["requests"] if now-t<3600]
        self.binance_budget["requests"]=requests
        if (now<self.binance_budget.get("next_rest",0)
                or len(requests)>=BINANCE_DATA_REST_MAX_HOUR):
            return False
        requests.append(now)
        self.binance_budget["next_rest"]=now+BINANCE_DATA_REST_GAP
        self._write(self.budget_path,self.binance_budget)
        return True

    def resolve(self,symbol,start,end,cached):
        if not supported_symbol(symbol) or end-start>480*STEP or start%STEP or end%STEP:
            raise ValueError('INVALID_CANDLE_REQUEST')
        end=min(end,int(self.clock()*1000)//STEP*STEP)
        found=valid_rows(cached,start,end)
        expected=range(start,end,STEP)
        foreign=("BYBIT_LINEAR","COINBASE_SPOT")

        def source(row):
            return row[12] if len(row)>12 else "BINANCE_CACHE_WS"

        def missing():
            return [t for t in expected if t not in found]

        def foreign_times():
            return [t for t in expected if t in found and source(found[t]) in foreign]

        def add(rows,src=None):
            added=0
            for t,row in valid_rows(rows,start,end).items():
                incoming=tagged(row,src) if src else row
                prior=found.get(t)
                # Binance WS/cache/archive/REST always wins. Never replace a
                # genuine Binance candle with Bybit or Coinbase spot.
                if prior is None or (source(prior) in foreign
                                     and source(incoming) not in foreign):
                    found[t]=incoming
                    added+=1
            return added

        # Archive is public static ZIP+checksum, not Binance trading REST.
        today=int(self.clock()*1000)//DAY*DAY
        needed_days=missing()+foreign_times()
        for day in sorted({t//DAY*DAY for t in needed_days if t//DAY*DAY<today}):
            add(self.archive(symbol,day))

        # Exactly one worker handles a historical snapshot at a time. Each
        # adapter also has an independent provider-specific throttle.
        if missing() and self.bybit is not None:
            added=add(self.bybit.fill_missing(symbol,start,end,missing(),list(found.values())))
            if added:
                self.log(f"[CRASH DATA ROUTE] {symbol}; BYBIT_LINEAR added={added}")

        if missing() and self.coinbase is not None:
            added=add(self.coinbase.fill_missing(symbol,start,end,missing(),list(found.values())))
            if added:
                self.log(f"[CRASH DATA ROUTE] {symbol}; COINBASE_SPOT added={added}")

        # Only after BOTH outside sources have been offered missing timestamps
        # may Binance historical REST repair them. This also handles symbols
        # not listed by either outside exchange. Once per hour, prefer
        # authoritative Binance candles to previously substituted foreign ones.
        gaps=missing()
        old_foreign=foreign_times()
        recheck=bool(not gaps and old_foreign and
                     self.clock()>=self.binance_recheck.get(symbol,0))
        work=gaps if gaps else (old_foreign if recheck else [])
        blocked=self.blocked()
        if work and not blocked:
            if not self._reserve_binance_rest():
                self.log(f"[CRASH DATA BINANCE DEFERRED] {symbol}; "
                         "HISTORY_BUDGET_12_PER_HOUR_5MIN_GAP")
            else:
                first=work[0]
                finish=first+STEP
                while (finish<end
                       and (finish not in found or source(found[finish]) in foreign)
                       and (finish-first)//STEP<REST_MAX_CANDLES):
                    finish+=STEP
                count=(finish-first)//STEP
                try:
                    # Don't repeatedly poll substituted venues for tiny gaps.
                    self.binance_recheck[symbol]=self.clock()+3600
                    added=add(self.rest("/fapi/v1/klines",{
                        "symbol":symbol,"interval":"15m",
                        "startTime":first,"endTime":finish-1,
                        "limit":count}),"BINANCE_REST")
                    self.log(f"[CRASH DATA BINANCE REST] {symbol}; "
                             f"requested={count}; added={added}; alternate_first=1")
                except Exception as error:
                    self.log(f"[CRASH DATA REST WAIT] {symbol}; "
                             f"type={type(error).__name__}; retry_later=1")
        elif work and blocked:
            self.log(f"[CRASH DATA REST WAIT] {symbol}; REST_BEKLEMESI; "
                     "Bybit_Coinbase_archive_only=1")

        rows=[found[t] for t in sorted(found)]
        self.log(f"[CRASH DATA SOURCES] {symbol}; candles={len(rows)}/{len(expected)}; "
                 f"missing={len(missing())}; sources={source_counts(rows)}")
        return rows
