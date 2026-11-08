"""Asynchronous, public-data-only five-day research snapshots (one worker)."""
import json
import math
import os
import threading
import time

STEP = 900000
COUNT = 480
DOWNLOAD_INTERVAL = 180
SIGNAL_URGENT_REPAIR_WINDOW_SECONDS = 280


class HistoryPacingWait(Exception):
    pass


from crash_policy import supported_symbol

class HistoryCollector:
    def __init__(self, directory, fetch, emit, blocked, log, resolver=None, ws_rows=None, on_rows=None, bybit=None, urgent_repair=None):
        self.directory, self.fetch, self.emit = directory, fetch, emit
        self.blocked, self.log = blocked, log
        self.resolver, self.ws_rows, self.on_rows = resolver, ws_rows, on_rows
        self.bybit = bybit  # Compatibility for existing caller tests; resolver owns fallback
        self.urgent_repair = urgent_repair  # Public one-bar recovery, separately budgeted.
        self.lock = threading.RLock()
        self.thread = None
        self.attempts = []
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, 'pending.json')
        self.pacing_path = os.path.join(directory, 'download_schedule.json')
        self.next_download = self._read(self.pacing_path, {}).get('next_download', 0)
        self.jobs = self._read(self.path, {})
        discarded = [key for key,job in self.jobs.items()
                     if not isinstance(job, dict) or not supported_symbol(job.get('symbol'))]
        for key in discarded:
            self.jobs.pop(key)
        if discarded:
            self._write(self.path, self.jobs)
            self.log(f'[CRASH HISTORY SKIP] removed {len(discarded)} unsupported persisted jobs')

    def _read(self, path, default):
        if not os.path.exists(path):
            return default
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)

    def _write(self, path, value):
        with open(path + '.tmp', 'w', encoding='utf-8') as handle:
            json.dump(value, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(path + '.tmp', path)

    def submit(self, record, phase, at):
        symbol = record['symbol']
        if not supported_symbol(symbol):
            raise ValueError(f'Invalid research symbol: {symbol!r}; expected ASCII A-Z, 0-9, underscore, 1-40 characters')
        if phase not in ('signal', 'entry', 'warmup'):
            raise ValueError(f'Invalid history phase: {phase!r}')
        if isinstance(at, bool) or not isinstance(at, (float, int)) or not math.isfinite(at) or at <= 0:
            raise ValueError(f'Invalid history timestamp: {at!r}')
        key = record['signal_id'] + ':' + phase
        anchor = record.get('created_at', at)
        if (isinstance(anchor, bool) or not isinstance(anchor, (int, float))
                or not math.isfinite(anchor) or anchor <= 0):
            anchor = at
        with self.lock:
            updated = dict(self.jobs)
            if phase == 'warmup' and key in updated:
                updated[key] = dict(updated[key], as_of=max(at,updated[key]['as_of']), created_at=at)
            else:
                updated.setdefault(key, dict(signal_id=record['signal_id'], symbol=symbol,
                                             phase=phase, as_of=at, created_at=anchor, retry_after=0))
            # A failed serialization must not poison the in-memory queue.
            self._write(self.path, updated)
            self.jobs = updated

    def prune_expired(self, now):
        """Do not consume download budget on jobs for >24h-old candidates."""
        from crash_policy import TRACKING_SECONDS
        with self.lock:
            expired = [key for key, job in self.jobs.items()
                       if now - job.get('created_at', job.get('as_of', 0)) >= TRACKING_SECONDS]
            if expired:
                remaining = {k: v for k, v in self.jobs.items() if k not in expired}
                self._write(self.path, remaining)
                self.jobs = remaining
        if expired:
            self.log(f'[CRASH HISTORY EXPIRED] removed={len(expired)}; horizon=24h')
        return len(expired)

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._run, name='crash-history', daemon=True)
            self.thread.start()

    @staticmethod
    def validated(rows):
        if not isinstance(rows, list):
            raise ValueError('Invalid history response')
        result = {}
        for row in rows:
            if len(row) < 7:
                raise ValueError('Invalid candle')
            t = int(row[0])
            vals = [float(row[i]) for i in (1, 2, 3, 4, 5)]
            if (t % STEP or int(row[6]) != t + STEP - 1
                    or any(not math.isfinite(v) for v in vals)
                    or min(vals[:4]) <= 0 or vals[4] < 0):
                raise ValueError('Invalid candle values')
            result[t] = row
        return result

    def collect(self, job, scheduled_at=None):
        if not supported_symbol(job.get('symbol')):
            raise ValueError('Unsupported history symbol')
        # End at the last fully closed candle at the ORIGINAL event time.
        end = int(job['as_of'] * 1000) // STEP * STEP
        start = end - COUNT * STEP
        path = os.path.join(self.directory, job['symbol'] + '.json')
        cached = self.validated(self._read(path, []))
        if self.ws_rows:
            from crash_data_sources import valid_rows
            for t, row in valid_rows(self.ws_rows(job['symbol']), start, end).items():
                old = cached.get(t)
                if (old is None or len(row) <= 12 or row[12] not in ('BYBIT_LINEAR','COINBASE_SPOT','OKX_SWAP')
                        or (len(old) > 12 and old[12] in ('BYBIT_LINEAR','COINBASE_SPOT','OKX_SWAP'))):
                    cached[t] = row
        expected = list(range(start, end, STEP))
        missing = [t for t in expected if t not in cached]
        # Fresh signals can miss just the last CLOSED 15m slot even when all
        # 5-day history is ready. Bypass the 180s research-download queue for
        # this one gap, using separately rate-capped public venue endpoints.
        # An adjacent Binance candle is compulsory for price parity.
        if (job['phase'] == 'signal' and callable(self.urgent_repair)
                and scheduled_at is not None
                and 0 <= scheduled_at-job['as_of'] <= SIGNAL_URGENT_REPAIR_WINDOW_SECONDS
                # Repair the final closed slot even if two other historical
                # slots are missing: 477/480 can become valid 478/480.
                and end-STEP in missing and len(missing) <= 3
                and end-2*STEP in cached
                and sum(t in cached for t in range(end-16*STEP,end,STEP)) >= 14):
            try:
                found = self.validated(self.urgent_repair(
                    job['symbol'], start, end, list(cached.values())))
                recovered = found.get(end-STEP)
                if recovered is not None:
                    cached[end-STEP] = recovered
                    missing = [t for t in expected if t not in cached]
                    self.log(f"[CRASH HISTORY URGENT LAST] {job['symbol']}; "
                             f"source={recovered[12] if len(recovered)>12 else 'BINANCE'}; "
                             "closed_slot_verified=1")
            except Exception as error:
                self.log(f"[CRASH HISTORY URGENT WAIT] {job['symbol']}; "
                         f"reason={type(error).__name__}")
        alternate_cached = any(len(r)>12 and r[12] in ('BYBIT_LINEAR','COINBASE_SPOT','OKX_SWAP')
                           for t,r in cached.items() if start <= t < end)
        # A signal with 478-480 genuine closed candles is decision-ready only
        # when the latest closed slot and >=14 of the last 16 slots exist.
        # Do not delay that signal behind a paced *research* download to fill
        # optional old gaps or replace reference-venue candles.
        signal_cache_ready = (job['phase'] == 'signal' and len(missing) <= 2
            and end-STEP in cached
            and sum(t in cached for t in range(end-16*STEP, end, STEP)) >= 14)
        if signal_cache_ready and (missing or alternate_cached):
            self.log(f"[CRASH HISTORY CACHE READY] {job['symbol']}; "
                     f"real_candles={COUNT-len(missing)}/480; latest_closed=1; "
                     "paced_download_bypassed=1")
        if (missing or (alternate_cached and self.resolver is not None)) and not signal_cache_ready:
            if scheduled_at is not None:
                # Hourly limit applies ONLY to network work. Cache-only signal
                # snapshots must remain available even when this budget is full.
                if len(self.attempts) >= 18:
                    self.next_download = max(self.next_download,
                                             min(self.attempts) + 3600)
                    raise HistoryPacingWait()
                if scheduled_at < self.next_download:
                    raise HistoryPacingWait()
                self.next_download = scheduled_at + DOWNLOAD_INTERVAL
                self._write(self.pacing_path, {'next_download': self.next_download})
                self.attempts.append(scheduled_at)
            if self.resolver:
                fresh = self.validated(self.resolver.resolve(job['symbol'],start,end,list(cached.values())))
                cached.update({t:r for t,r in fresh.items() if start <= t < end})
            else:
                # Download only the first contiguous missing range; retain partial progress.
                missing_end = missing[0] + STEP
                while missing_end < end and missing_end not in cached:
                    missing_end += STEP
                self.log(f"[CRASH HISTORY DOWNLOAD] {job['symbol']}; "
                         f"cached={COUNT-len(missing)}/480; missing={len(missing)}; "
                         f"request_candles={(missing_end-missing[0])//STEP}; next_request_after=180s")
                rows = self.fetch('/fapi/v1/klines', dict(symbol=job['symbol'], interval='15m',
                                  startTime=missing[0], endTime=missing_end-1, limit=499))
                fresh = self.validated(rows)
                cached.update({t:r for t,r in fresh.items() if start <= t < end})
        snapshot = [cached[t] for t in expected if t in cached]
        # Retain ten days locally to support delayed entry and queued snapshots.
        latest = max(cached, default=end)
        retained = [cached[t] for t in sorted(cached) if t >= latest - 960 * STEP]
        self._write(path, retained)
        if self.on_rows:
            self.on_rows(job['symbol'], retained)
        from crash_data_sources import source_counts
        event = dict(data_sources=source_counts(snapshot), type='history_5d', signal_id=job['signal_id'], symbol=job['symbol'],
                    phase=job['phase'], as_of=job['as_of'], fetched_at=time.time(),
                    interval='15m', expected_candles=COUNT, candle_count=len(snapshot),
                    complete=len(snapshot)==COUNT, missing_candles=COUNT-len(snapshot),
                    closed_15m_ohlcv=snapshot)
        if (event['data_sources'].get('BYBIT_LINEAR') or
                event['data_sources'].get('COINBASE_SPOT')):
            self.log(f'[CRASH HISTORY MIXED] {job["symbol"]}; '
                     f'bybit={event["data_sources"].get("BYBIT_LINEAR",0)}; '
                     f'coinbase={event["data_sources"].get("COINBASE_SPOT",0)}; '
                     'Binance=oncelikli; karma_veri_ile_sinyal_hesaplanabilir')
        return event

    def step(self, now=None):
        now = time.time() if now is None else now
        self.prune_expired(now)
        if self.blocked() and self.resolver is None:
            return
        self.attempts = [t for t in self.attempts if now-t < 3600]
        with self.lock:
            ready = [(k,dict(v)) for k,v in self.jobs.items() if v['retry_after']<=now]
            # Current signal checks precede entry research and expired snapshots.
            ready.sort(key=lambda item: (not (item[1]['phase']=='signal' and
                0 <= now-item[1]['as_of'] <= SIGNAL_URGENT_REPAIR_WINDOW_SECONDS), item[1]['phase'] != 'warmup', -item[1]['as_of']))
            item = ready[0] if ready else None
        if item is None:
            return
        key, job = item
        try:
            event = self.collect(job, scheduled_at=now)
            # Use fetch day so the upcoming ZIP includes late/retried downloads.
            if job['phase'] != 'warmup':
                self.emit(event, time.time())
            with self.lock:
                if event['complete'] and not any(event['data_sources'].get(src,0) for src in ('BYBIT_LINEAR','COINBASE_SPOT','OKX_SWAP')):
                    self.jobs.pop(key, None)
                else:
                    # Incomplete OLD history may wait an hour, but a fresh
                    # signal missing the *last closed bar* must not be delayed
                    # beyond its 280s decision window. REST pacing/budgets
                    # remain enforced by collect() on the next attempt.
                    near_complete = COUNT-2 <= event['candle_count'] < COUNT
                    end = int(job['as_of'] * 1000) // STEP * STEP
                    last_closed_present = any(
                        int(row[0]) == end-STEP
                        for row in event['closed_15m_ohlcv'])
                    urgent_last_gap = (near_complete and job['phase'] == 'signal'
                        and 0 <= now-job['as_of'] <= SIGNAL_URGENT_REPAIR_WINDOW_SECONDS
                        and not last_closed_present)
                    self.jobs[key]['retry_after'] = now + (
                        30 if urgent_last_gap else
                        3600 if near_complete else
                        1800 if event['complete'] else
                        300 if self.resolver else 1800)
                self._write(self.path, self.jobs)
            if COUNT-2 <= event['candle_count'] < COUNT:
                with self.lock:
                    retry_after = self.jobs.get(key, {}).get('retry_after', now)
                self.log(f"[CRASH HISTORY NEAR_COMPLETE] {job['symbol']}; "
                         f"real_candles={event['candle_count']}/480; "
                         f"next_retry_s={max(0, int(retry_after-now))}")
            self.log(f"[CRASH HISTORY] {job['symbol']} {job['phase']}: {event['candle_count']}/480")
        except HistoryPacingWait:
            with self.lock:
                self.jobs[key]['retry_after'] = self.next_download
                self._write(self.path, self.jobs)
            self.log(f"[CRASH HISTORY SCHEDULE] {job['symbol']}; "
                     f"queue={len(self.jobs)}; wait_seconds={max(0,int(self.next_download-now))}")
        except Exception as error:
            with self.lock:
                self.jobs[key]['retry_after'] = now + (300 if self.resolver else 1800)
                self._write(self.path, self.jobs)
            self.log(f"[CRASH HISTORY DEFERRED] {job['symbol']}: {type(error).__name__}: {str(error)[:200]}")

    def _run(self):
        while True:
            try:
                self.step()
            except Exception as error:
                self.log(f'[CRASH HISTORY ERR] {type(error).__name__}: {str(error)[:200]}')
            # A 24-job signal burst would otherwise wait 24 x 15 = 360s,
            # outlasting the 280s candidate deadline. Drain eligible urgent
            # cache checks promptly; all actual network work remains protected
            # by the unchanged 180s pacing and per-venue API budgets.
            now = time.time()
            with self.lock:
                urgent_ready = any(
                    job.get('phase') == 'signal'
                    and job.get('retry_after', 0) <= now
                    and 0 <= now-job.get('as_of', 0) <= SIGNAL_URGENT_REPAIR_WINDOW_SECONDS
                    for job in self.jobs.values())
            time.sleep(2 if urgent_ready else 15)


