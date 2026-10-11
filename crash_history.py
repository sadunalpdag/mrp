"""Asynchronous, public-data-only five-day research snapshots (one worker)."""
import json
import math
import os
import re
import threading
import time

STEP = 900000
COUNT = 480

class HistoryCollector:
    def __init__(self, directory, fetch, emit, blocked, log):
        self.directory, self.fetch, self.emit = directory, fetch, emit
        self.blocked, self.log = blocked, log
        self.lock = threading.RLock()
        self.thread = None
        self.attempts = []
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, 'pending.json')
        self.jobs = self._read(self.path, {})

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
        if not re.fullmatch(r'[A-Z0-9_]{1,40}', symbol):
            raise ValueError('Invalid research symbol')
        key = record['signal_id'] + ':' + phase
        with self.lock:
            self.jobs.setdefault(key, dict(signal_id=record['signal_id'], symbol=symbol,
                                          phase=phase, as_of=at, retry_after=0))
            self._write(self.path, self.jobs)

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

    def collect(self, job):
        # End at the last fully closed candle at the ORIGINAL event time.
        end = int(job['as_of'] * 1000) // STEP * STEP
        start = end - COUNT * STEP
        path = os.path.join(self.directory, job['symbol'] + '.json')
        cached = self.validated(self._read(path, []))
        expected = list(range(start, end, STEP))
        missing = [t for t in expected if t not in cached]
        if missing:
            # An incremental tail in normal operation; also repairs internal gaps.
            rows = self.fetch('/fapi/v1/klines', dict(symbol=job['symbol'], interval='15m',
                              startTime=missing[0], endTime=end-1, limit=499))
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
            item = next(((k,dict(v)) for k,v in self.jobs.items() if v['retry_after']<=now), None)
        if item is None:
            return
        key, job = item
        self.attempts.append(now)
        try:
            event = self.collect(job)
            # Use fetch day so the upcoming ZIP includes late/retried downloads.
            self.emit(event, time.time())
            with self.lock:
                self.jobs.pop(key, None)
                self._write(self.path, self.jobs)
            self.log(f"[CRASH HISTORY] {job['symbol']} {job['phase']}: {event['candle_count']}/480")
        except Exception as error:
            with self.lock:
                self.jobs[key]['retry_after'] = now + 1800
                self._write(self.path, self.jobs)
            self.log(f"[CRASH HISTORY DEFERRED] {job['symbol']}: {type(error).__name__}")

    def _run(self):
        while True:
            try:
                self.step()
            except Exception as error:
                self.log(f'[CRASH HISTORY ERR] {type(error).__name__}')
            time.sleep(15)
