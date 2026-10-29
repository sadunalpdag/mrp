"""Closed Binance USD-M 15m: local cache/WS -> daily archive -> REST gaps -> Bybit.
All fallbacks are source-tagged; Binance is preferred whenever available.
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
    def __init__(self, directory, rest, blocked, log, get=None, clock=time.time, bybit=None):
        self.directory,self.rest,self.blocked,self.log=directory,rest,blocked,log
        self.get,self.clock=get,clock
        self.retry={}
        self.binance_recheck={}
        self.bybit=bybit
        self.next_http=0
        os.makedirs(directory,exist_ok=True)

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

    def resolve(self,symbol,start,end,cached):
        if not supported_symbol(symbol) or end-start>480*STEP or start%STEP or end%STEP:
            raise ValueError('INVALID_CANDLE_REQUEST')
        end=min(end,int(self.clock()*1000)//STEP*STEP)
        found=valid_rows(cached,start,end)
        expected=range(start,end,STEP)

        def source(row):
            return row[12] if len(row)>12 else 'BINANCE_CACHE_WS'

        def missing():
            return [t for t in expected if t not in found]

        def bybit_times():
            return [t for t in expected if t in found and source(found[t])=='BYBIT_LINEAR']

        def add(rows,src=None):
            added=0
            for t,row in valid_rows(rows,start,end).items():
                if (t not in found or
                        (source(found[t])=='BYBIT_LINEAR' and
                         (src is not None or source(row)!='BYBIT_LINEAR'))):
                    found[t]=tagged(row,src) if src else row
                    added+=1
            return added

        # First use cached closed Binance/WebSocket candles, then published
        # Binance daily archives. The archive host has a separate request
        # budget and cached ZIPs are not downloaded again.
        today=int(self.clock()*1000)//DAY*DAY
        needed_days=missing()+bybit_times()
        for day in sorted({t//DAY*DAY for t in needed_days if t//DAY*DAY<today}):
            add(self.archive(symbol,day))

        # Only query the gaps left AFTER cache, WS and archive. A single REST
        # call is bounded to at most 192 consecutive 15m candles (not 480).
        # Keep a 30m backoff for later Binance replacement of Bybit candles.
        gaps=missing()
        old_bybit=bybit_times()
        blocked=self.blocked()
        recheck=bool(old_bybit and self.clock()>=self.binance_recheck.get(symbol,0))
        work=gaps if gaps else (old_bybit if recheck else [])
        rest_added=0
        if work and not blocked:
            first=work[0]
            finish=first+STEP
            while (finish<end and (finish not in found or source(found[finish])=='BYBIT_LINEAR')
                   and (finish-first)//STEP<REST_MAX_CANDLES):
                finish+=STEP
            request_candles=(finish-first)//STEP
            try:
                if not gaps:
                    self.binance_recheck[symbol]=self.clock()+1800
                rest_added=add(self.rest('/fapi/v1/klines', {
                    'symbol':symbol, 'interval':'15m',
                    'startTime':first, 'endTime':finish-1,
                    'limit':request_candles}), 'BINANCE_REST')
                self.log(f'[CRASH DATA REST GAPS] {symbol}; '
                         f'istenen={request_candles}; alinan={rest_added}; '
                         f'arsiv_oncelikli=1')
            except Exception as error:
                self.log(f'[CRASH DATA REST WAIT] {symbol}; '
                         f'type={type(error).__name__}; yedek kaynaklara geciliyor')
        elif gaps and blocked:
            self.log(f'[CRASH DATA REST WAIT] {symbol}; REST_BEKLEMESI; yedek kaynaklar aktif')

        # If REST is working, allow later queue passes to fetch any additional
        # gaps without immediately loading Bybit. If REST is blocked or could
        # not supply missing rows, Bybit may fill ONLY those missing timestamps.
        if missing() and self.bybit is not None and (blocked or rest_added==0):
            add(self.bybit.fill_missing(symbol,start,end,missing(),list(found.values())))

        rows=[found[t] for t in sorted(found)]
        self.log(f'[CRASH DATA SOURCES] {symbol}; candles={len(rows)}/{len(expected)}; '
                 f'missing={len(missing())}; sources={source_counts(rows)}')
        return rows
