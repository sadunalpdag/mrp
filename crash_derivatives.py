"""Prospective, non-blocking OI/CVD shadow snapshots; never gates V2 or orders."""
import csv
import io
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

STEP = 900000
ADJUSTMENT = {'DELEVERAGING':8, 'SPOT_CONFIRMED_RECOVERY':10,
              'POSSIBLE_ABSORPTION':3, 'NEUTRAL':0, 'OI_RISING_SELL_PRESSURE':-15}
FIELDS = ('oi_change_1h','oi_change_4h','spot_cvd_1h','spot_cvd_4h',
          'perp_cvd_1h','perp_cvd_4h','funding_rate','price_change_4h')


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError, OverflowError):
        return None


def classify_derivatives_regime(price_change_4h, oi_change_4h, spot_cvd_4h, perp_cvd_4h):
    p, oi, spot, perp = map(number, (price_change_4h, oi_change_4h, spot_cvd_4h, perp_cvd_4h))
    # Missing required inputs are not neutral evidence. Ordered rules match user specification.
    if any(v is None for v in (p, oi, spot, perp)):
        return 'UNKNOWN'
    if p < 0 and oi <= -2:
        return 'DELEVERAGING'
    if p < 0 and oi >= 2 and perp < 0:
        return 'OI_RISING_SELL_PRESSURE'
    if p < 0 and oi >= 2 and spot > 0:
        return 'POSSIBLE_ABSORPTION'
    if spot > 0 and perp > 0:
        return 'SPOT_CONFIRMED_RECOVERY'
    return 'NEUTRAL'


def finalize(data):
    data['derivatives_regime'] = classify_derivatives_regime(
        *(data.get(k) for k in ('price_change_4h','oi_change_4h','spot_cvd_4h','perp_cvd_4h')))
    data['regime_adjustment'] = ADJUSTMENT.get(data['derivatives_regime'], 0)
    base = number(data.get('base_score'))
    data['shadow_score'] = None if base is None else max(0, min(100, base+data['regime_adjustment']))
    return data


def empty_snapshot(record):
    return finalize(dict.fromkeys(FIELDS) | {
        'schema_version':1, 'signal_id':record['signal_id'], 'symbol':record['symbol'],
        'as_of':record['created_at'], 'window_end_ms':int(record['created_at']*1000)//STEP*STEP,
        'base_score':(record.get('v2_features', {}).get('strength_score') or {}).get('score'),
        'status':'PENDING', 'errors':{}, 'sources':{}, 'signal_gate_changed':False})


def candles(rows, end, count=16):
    valid = {}
    for r in rows or []:
        try:
            t = int(r[0])
            if len(r)>12 and r[12] not in ('BINANCE_CACHE_WS','BINANCE_REST','BINANCE_ARCHIVE'):
                continue  # Never label another exchange's or synthetic taker volume as Binance CVD.
            vals = [number(r[j]) for j in (1,4,7,10)]
            if any(v is None for v in vals):
                continue
            o,c,total,buy = vals
            if (end-count*STEP <= t < end and t%STEP == 0 and int(r[6]) == t+STEP-1
                    and o > 0 and c > 0 and 0 <= buy <= total):
                valid[t] = (o,c,2*buy-total)
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
    return valid


def cvd(rows, end, count):
    values = candles(rows, end, count)
    times = list(range(end-count*STEP, end, STEP))
    return sum(values[t][2] for t in times) if all(t in values for t in times) else None


def oi_changes(rows, end):
    values = {}
    for r in rows or []:
        try:
            t, v = int(r['timestamp']), number(r['sumOpenInterest'])
            if end-4*3600000 <= t <= end and v is not None and v > 0:
                values[t] = v
        except (KeyError, TypeError, ValueError):
            continue
    # All comparisons use exactly the same closed-window endpoints, no future or stale substitutions.
    return {f'oi_change_{h}h':100*(values[end]/values[end-h*3600000]-1)
            if end in values and end-h*3600000 in values else None for h in (1,4)}


def format_shadow(d):
    def fmt(key, suffix='', multiplier=1):
        v = number(d.get(key))
        return 'bilinmiyor' if v is None else f'{v*multiplier:+,.2f}{suffix}'
    regime = d.get('derivatives_regime', 'UNKNOWN')
    warning = '\n⚠️ Fiyat düşerken OI ve vadeli satış baskısı artıyor.' if regime == 'OI_RISING_SELL_PRESSURE' else ''
    return (f"📊 Türev rejimi: {regime} ({d.get('status','UNKNOWN')})\n"
            f"OI 1s: {fmt('oi_change_1h','%')} | 4s: {fmt('oi_change_4h','%')}\n"
            f"Spot CVD 1s: {fmt('spot_cvd_1h',' USDT')} | 4s: {fmt('spot_cvd_4h',' USDT')}\n"
            f"Perp CVD 1s: {fmt('perp_cvd_1h',' USDT')} | 4s: {fmt('perp_cvd_4h',' USDT')}\n"
            f"Son ödenen funding: {fmt('funding_rate','%',100)}\n"
            f"Temel V2: {fmt('base_score')} | Gölge puan: {fmt('shadow_score')}"
            f"{warning}\nKapalı 15m pencereler; yalnız araştırma, sinyal/temel puan değişmedi.\n")


class ShadowCollector:
    def __init__(self, directory, fetch, blocked, cached, log, clock=time.time):
        self.root = Path(directory); self.root.mkdir(parents=True, exist_ok=True)
        self.fetch, self.blocked, self.cached, self.log, self.clock = fetch, blocked, cached, log, clock
        self.jobs = queue.Queue(maxsize=100)
        self.results = queue.Queue(maxsize=100)
        self.seen = set()
        self.next_job = 0
        self.thread = None

    def submit(self, record):
        sid = record['signal_id']
        if sid in self.seen:
            return
        self.jobs.put_nowait(empty_snapshot(record))
        self.seen.add(sid)

    def collect(self, d):
        symbol, end = d['symbol'], d['window_end_ms']
        params = dict(symbol=symbol, interval='15m', startTime=end-16*STEP, endTime=end-1, limit=16)
        def get(label, path, args):
            if self.blocked():
                d['errors'][label] = 'REST_COOLDOWN'
                return []
            try:
                result = self.fetch(path, args)
                if not isinstance(result, list):
                    raise ValueError('INVALID_RESPONSE')
                return result
            except Exception as error:
                d['errors'][label] = type(error).__name__
                return []
        perp = self.cached(symbol)
        if cvd(perp, end, 16) is None:
            perp = get('perp', '/fapi/v1/klines', params)
        spot = get('spot', '/api/v3/klines', params)
        oi = get('oi', '/futures/data/openInterestHist',
                 dict(symbol=symbol, period='15m', startTime=end-16*STEP, endTime=end, limit=17))
        funding = get('funding', '/fapi/v1/fundingRate',
                      dict(symbol=symbol, startTime=int(d['as_of']*1000)-86400000,
                           endTime=int(d['as_of']*1000), limit=100))
        for market, rows in (('spot',spot),('perp',perp)):
            for h in (1,4):
                d[f'{market}_cvd_{h}h'] = cvd(rows,end,h*4)
            d['sources'][market] = f'BINANCE_{market.upper()}_KLINE_TAKER_QUOTE'
        values = candles(perp,end)
        if len(values)==16:
            d['price_change_4h'] = 100*(values[end-STEP][1]/values[end-16*STEP][0]-1)
        d.update(oi_changes(oi,end))
        d['oi_endpoints_ms'] = [end-h*3600000 for h in (4,1,0)]
        d['cvd_method'] = 'sum(2*taker_buy_quote-total_quote); closed_15m; USDT'
        d['complete_candles'] = {'spot':len(candles(spot,end)), 'perp':len(candles(perp,end))}
        eligible = [r for r in funding if isinstance(r,dict) and number(r.get('fundingTime')) is not None
                    and d['as_of']*1000-86400000 <= float(r['fundingTime']) <= d['as_of']*1000
                    and number(r.get('fundingRate')) is not None]
        if eligible:
            last = max(eligible,key=lambda r:float(r['fundingTime']))
            d.update(funding_rate=number(last['fundingRate']), funding_time_ms=int(last['fundingTime']))
        d['sources'].update(oi='BINANCE_SUM_OPEN_INTEREST_BASE_UNITS', funding='BINANCE_LAST_SETTLED')
        d.update(status='COMPLETE' if all(d.get(k) is not None for k in FIELDS) else 'PARTIAL', fetched_at=self.clock())
        return finalize(d)

    def step(self):
        if self.clock() < self.next_job or self.jobs.empty() or self.results.full():
            return
        d = self.jobs.get_nowait()
        self.next_job = self.clock()+60  # at most one signal job/min, never scan all markets
        try:
            d = self.collect(d)
        except Exception as error:
            d.update(status='ERROR', fetched_at=self.clock())
            d['errors']['collector'] = type(error).__name__
            finalize(d)
        # Raw source arrays are represented by timestamp-bound aggregated features and provenance.
        with (self.root/'snapshots.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')
        self.log(f"[CRASH DERIVATIVES] {d['symbol']}; regime={d['derivatives_regime']}; status={d['status']}; errors={d['errors']}")
        self.results.put_nowait(d)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def run():
            while True:
                try:
                    self.step()
                except Exception as error:
                    self.log(f'[CRASH DERIVATIVES ERR] {type(error).__name__}')
                time.sleep(1)
        self.thread = threading.Thread(target=run,name='derivatives-shadow',daemon=True)
        self.thread.start()


def export_csv(records):
    out = io.StringIO()
    fields = ['signal_id','symbol','as_of','status','derivatives_regime','base_score','shadow_score',
              'regime_adjustment',*FIELDS,'tracking_status','entered_at','finished_at',
              'shadow_new_low_observed','shadow_mae_pct','coverage_gaps']
    writer = csv.DictWriter(out,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
    for r in records:
        d = r.get('derivatives_shadow')
        if d:
            writer.writerow(d | {k:r.get(k) for k in fields[-6:]})
    return out.getvalue()


def summarize(records):
    groups = {}
    for r in records:
        d = r.get('derivatives_shadow')
        if not d or not r.get('v2_sent'):
            continue
        g = groups.setdefault(d['derivatives_regime'], {'signals':0,'entered':0,'tp':0,'timeout':0,
                            'open':0,'new_low_observed':0,'new_low_observed_count':0,'mae_observations':[], 'with_gaps':0})
        g['signals'] += 1
        g['entered'] += int(bool(r.get('entered_at')))
        status = r.get('tracking_status')
        g['tp'] += int(status=='TP')
        g['timeout'] += int(status=='TIMEOUT' and bool(r.get('entered_at')))
        g['open'] += int(status=='OPEN')
        g['new_low_observed'] += int(r.get('shadow_new_low_observed') is True)
        g['new_low_observed_count'] += int(type(r.get('shadow_new_low_observed')) is bool)
        g['with_gaps'] += int(bool(r.get('coverage_gaps')))
        if number(r.get('shadow_mae_pct')) is not None:
            g['mae_observations'].append(r['shadow_mae_pct'])
    for g in groups.values():
        n = g['tp']+g['timeout']
        g['tp_rate_resolved_pct'] = 100*g['tp']/n if n else None
        g['new_low_observed_pct'] = 100*g['new_low_observed']/g['new_low_observed_count'] if g['new_low_observed_count'] else None
        values = g.pop('mae_observations')
        g['mae_sample_count'] = len(values)
        g['mean_observed_mae_pct'] = sum(values)/len(values) if values else None
    return {'groups':groups, 'minimum_signals_for_review':50, 'automatic_promotion':False,
            'notes':'Observed virtual results only; open/waiting are not failures. MAE is negative, trade-print based, gaps may miss lows. No net profit claim; fees, funding and slippage not included. Regime hierarchy follows supplied rule order.'}
