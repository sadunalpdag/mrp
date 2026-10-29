"""Asynchronous, public-data-only five-day research snapshots (one worker)."""
import json
import math
import os
import threading
import time

STEP = 900000
COUNT = 480
DOWNLOAD_INTERVAL = 60


class HistoryPacingWait(Exception):
    pass


from crash_policy import supported_symbol

class HistoryCollector:
    def __init__(self, directory, fetch, emit, blocked, log):
        self.directory, self.fetch, self.emit = directory, fetch, emit
        self.blocked, self.log = blocked, log
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
        if phase not in ('signal', 'entry'):
            raise ValueError(f'Invalid history phase: {phase!r}')
        if isinstance(at, bool) or not isinstance(at, (float, int)) or not math.isfinite(at) or at <= 0:
            raise ValueError(f'Invalid history timestamp: {at!r}')
        key = record['signal_id'] + ':' + phase
        with self.lock:
            updated = dict(self.jobs)
            updated.setdefault(key, dict(signal_id=record['signal_id'], symbol=symbol,
                                         phase=phase, as_of=at, retry_after=0))
            # A failed serialization must not poison the in-memory queue.
            self._write(self.path, updated)
            self.jobs = updated

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
        expected = list(range(start, end, STEP))
        missing = [t for t in expected if t not in cached]
        if missing:
            if scheduled_at is not None:
                if scheduled_at < self.next_download:
                    raise HistoryPacingWait()
                self.next_download = scheduled_at + DOWNLOAD_INTERVAL
                self._write(self.pacing_path, {'next_download': self.next_download})
                self.attempts.append(scheduled_at)
            # Download only the first contiguous missing range; retain partial progress.
            missing_end = missing[0] + STEP
            while missing_end < end and missing_end not in cached:
                missing_end += STEP
            self.log(f"[CRASH HISTORY DOWNLOAD] {job['symbol']}; "
                     f"cached={COUNT-len(missing)}/480; missing={len(missing)}; "
                     f"request_candles={(missing_end-missing[0])//STEP}; next_request_after=60s")
            rows = self.fetch('/fapi/v1/klines', dict(symbol=job['symbol'], interval='15m',
                              startTime=missing[0], endTime=missing_end-1, limit=499))
            fresh = self.validated(rows)
            cached.update({t:r for t,r in fresh.items() if start <= t < end})
        snapshot = [cached[t] for t in expected if t in cached]
        # Retain ten days locally to support delayed entry and queued snapshots.
        latest = max(cached, default=end)
        retained = [cached[t] for t in sorted(cached) if t >= latest - 960 * STEP]
        self._write(path, retained)
        return dict(type='history_5d', signal_id=job['signal_id'], symbol=job['symbol'],
                    phase=job['phase'], as_of=job['as_of'], fetched_at=time.time(),
                    interval='15m', expected_candles=COUNT, candle_count=len(snapshot),
                    complete=len(snapshot)==COUNT, missing_candles=COUNT-len(snapshot),
                    closed_15m_ohlcv=snapshot)

    def step(self, now=None):
        now = time.time() if now is None else now
        if self.blocked():
            return
        self.attempts = [t for t in self.attempts if now-t < 3600]
        if len(self.attempts) >= 24:
            return
        with self.lock:
            ready = [(k,dict(v)) for k,v in self.jobs.items() if v['retry_after']<=now]
            # Current signal checks precede entry research and expired snapshots.
            ready.sort(key=lambda item: (not (item[1]['phase']=='signal' and
                0 <= now-item[1]['as_of'] <= 90), -item[1]['as_of']))
            item = ready[0] if ready else None
        if item is None:
            return
        key, job = item
        try:
            event = self.collect(job, scheduled_at=now)
            # Use fetch day so the upcoming ZIP includes late/retried downloads.
            self.emit(event, time.time())
            with self.lock:
                if event['complete']:
                    self.jobs.pop(key, None)
                else:
                    self.jobs[key]['retry_after'] = now + 1800
                self._write(self.path, self.jobs)
            self.log(f"[CRASH HISTORY] {job['symbol']} {job['phase']}: {event['candle_count']}/480")
        except HistoryPacingWait:
            with self.lock:
                self.jobs[key]['retry_after'] = self.next_download
                self._write(self.path, self.jobs)
            self.log(f"[CRASH HISTORY SCHEDULE] {job['symbol']}; "
                     f"queue={len(self.jobs)}; wait_seconds={max(0,int(self.next_download-now))}")
        except Exception as error:
            with self.lock:
                self.jobs[key]['retry_after'] = now + 1800
                self._write(self.path, self.jobs)
            self.log(f"[CRASH HISTORY DEFERRED] {job['symbol']}: {type(error).__name__}: {str(error)[:200]}")

    def _run(self):
        while True:
            try:
                self.step()
            except Exception as error:
                self.log(f'[CRASH HISTORY ERR] {type(error).__name__}: {str(error)[:200]}')
            time.sleep(15)


