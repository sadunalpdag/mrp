"""Bybit USDT perpetual 15m: validated gap-only fallback with source tags."""
import json
import math
import os
import time
from crash_policy import supported_symbol
from crash_data_sources import STEP, valid_rows, tagged
from crash_v2 import assess_entry

COUNT = 480
PER_SYMBOL_RETRY_SECONDS = 1800
MAX_REQUESTS_PER_HOUR = 24


class BybitReference:
    def __init__(self, directory, log, get=None, clock=time.time):
        self.directory, self.log, self.get, self.clock = directory, log, get, clock
        self.next_symbol = {}
        self.requests = []
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _read(path):
        try:
            with open(path, encoding='utf-8') as handle:
                obj = json.load(handle)
            return obj if isinstance(obj, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write(path, value):
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


    def fill_missing(self, symbol, start, end, missing, binance_rows):
        """Fetch ONLY the first missing contiguous 15m span, never a whole 5d series.

        Compare Bybit close with a same-timestamp Binance anchor at the edge
        before accepting different-exchange candles. Fail closed without an
        anchor or if markets diverge more than 3%.
        """
        if (not supported_symbol(symbol) or start % STEP or end % STEP
                or not isinstance(missing, list) or not missing
                or end - start > COUNT * STEP):
            return []
        existing = valid_rows(binance_rows, start, end)
        wanted = sorted(t for t in set(missing)
                        if start <= t < end and t % STEP == 0 and t not in existing)
        if not wanted or not existing:
            return []  # Do not substitute an entire Bybit-only five-day history.
        first = wanted[0]
        stop = first + STEP
        while stop in wanted and stop - first < COUNT * STEP:
            stop += STEP
        # Include a 15-minute Binance overlap at the nearest available edge,
        # so the symbols cannot silently be treated as identical markets.
        left = first - STEP if first - STEP in existing else None
        right = stop if stop in existing else None
        if left is None and right is None:
            # A longer gap may have an anchor further away; a single request
            # must contain the directly adjacent anchor to verify parity.
            self.log(f'[CRASH BYBIT SKIP] {symbol}; reason=NO_ADJACENT_BINANCE_ANCHOR')
            return []
        request_start = left if left is not None else first
        request_end = stop + (STEP if right is not None else 0)
        now = self.clock()
        self.requests = [t for t in self.requests if now - t < 3600]
        if (now < self.next_symbol.get(symbol, 0)
                or len(self.requests) >= MAX_REQUESTS_PER_HOUR):
            return []
        self.requests.append(now)
        self.next_symbol[symbol] = now + PER_SYMBOL_RETRY_SECONDS
        get = self.get
        if get is None:
            import requests
            get = requests.get
        try:
            response = get('https://api.bybit.com/v5/market/kline',
                params={'category':'linear', 'symbol':symbol, 'interval':'15',
                        'start':request_start, 'end':request_end-1,
                        'limit':int((request_end-request_start)//STEP)},
                headers={}, timeout=(5, 10), allow_redirects=False)
            try:
                if response.status_code != 200:
                    raise ValueError('HTTP_' + str(response.status_code))
                payload = response.json()
            finally:
                response.close()
            if not isinstance(payload, dict) or payload.get('retCode') != 0:
                raise ValueError('BYBIT_CODE_' + str(payload.get('retCode', 'INVALID'))[:24]
                                 if isinstance(payload, dict) else 'INVALID_RESPONSE')
            result = payload.get('result')
            if (not isinstance(result, dict) or result.get('symbol') != symbol
                    or result.get('category') != 'linear'):
                raise ValueError('INVALID_BYBIT_SYMBOL')
            raw = result.get('list')
            if not isinstance(raw, list) or len(raw) > (request_end-request_start)//STEP:
                raise ValueError('INVALID_BYBIT_CANDLES')
            converted = []
            for item in raw:
                if not isinstance(item, list) or len(item) < 7:
                    continue
                try:
                    t = int(item[0])
                    numbers = [float(x) for x in item[1:7]]
                    o,h,l,c,v,q = numbers
                    if (t % STEP or not request_start <= t < request_end
                            or not all(math.isfinite(x) for x in numbers)
                            or min(o,h,l,c) <= 0 or v < 0 or q < 0
                            or l > min(o,c) or h < max(o,c)):
                        continue
                    converted.append(tagged(
                        [t,item[1],item[2],item[3],item[4],item[5],
                         t+STEP-1,item[6],0,0,0,0], 'BYBIT_LINEAR'))
                except (ValueError, TypeError, OverflowError):
                    continue
            found = valid_rows(converted, request_start, request_end)
            overlap = [t for t in (left, right) if t is not None and t in found]
            if not overlap:
                raise ValueError('BINANCE_ANCHOR_MISSING')
            for t in overlap:
                bc = float(existing[t][4])
                yc = float(found[t][4])
                if bc <= 0 or abs(yc/bc - 1) > 0.03:
                    raise ValueError('CROSS_EXCHANGE_PRICE_MISMATCH')
            output = [found[t] for t in wanted if first <= t < stop and t in found]
            self.next_symbol[symbol] = now + 60 if output else now + PER_SYMBOL_RETRY_SECONDS
            self.log(f'[CRASH BYBIT FILL] {symbol}; '
                     f'eksik_istendi={len(wanted)}; gelen={len(output)}; '
                     f'kontrol={len(overlap)}; sadece_eksik_mum=1')
            return output
        except Exception as error:
            self.log(f'[CRASH BYBIT WAIT] {symbol}; '
                     f'reason={type(error).__name__}; '
                     f'code={str(error)[:48] if isinstance(error, ValueError) else "TRANSPORT"}; '
                     f'tekrar_deneme_sn={PER_SYMBOL_RETRY_SECONDS}')
            return []

    def inspect(self, symbol, end):
        """Return a separately labeled, closed-candle V2 reference or None if paced.

        'end' is an exclusive 15m UTC boundary in milliseconds. Never retrieve
        the unfinished candle or mix this result into Binance OHLCV.
        """
        if not supported_symbol(symbol) or end % STEP:
            return None
        start = end - COUNT * STEP
        path = os.path.join(self.directory, symbol + '.json')
        saved = self._read(path)
        if (saved.get('end_ms') == end and
                saved.get('source') == 'BYBIT_LINEAR' and
                saved.get('complete') is True and
                len(saved.get('closed_15m_ohlcv', [])) == COUNT):
            return saved.get('reference')
        now = self.clock()
        self.requests = [t for t in self.requests if now - t < 3600]
        if now < self.next_symbol.get(symbol, 0) or len(self.requests) >= MAX_REQUESTS_PER_HOUR:
            return None
        self.next_symbol[symbol] = now + PER_SYMBOL_RETRY_SECONDS
        self.requests.append(now)
        get = self.get
        if get is None:
            import requests
            get = requests.get
        try:
            response = get('https://api.bybit.com/v5/market/kline',
                params={'category':'linear', 'symbol':symbol, 'interval':'15',
                        'start':start, 'end':end-1, 'limit':COUNT},
                headers={}, timeout=(5, 10), allow_redirects=False)
            try:
                if response.status_code != 200:
                    raise ValueError('HTTP_' + str(response.status_code))
                payload = response.json()
            finally:
                response.close()
            if not isinstance(payload, dict) or payload.get('retCode') != 0:
                code = payload.get('retCode', 'INVALID') if isinstance(payload, dict) else 'INVALID'
                raise ValueError('BYBIT_CODE_' + str(code)[:24])
            result = payload.get('result')
            if not isinstance(result, dict) or result.get('symbol') != symbol or result.get('category') != 'linear':
                raise ValueError('INVALID_BYBIT_SYMBOL')
            raw = result.get('list')
            if not isinstance(raw, list) or len(raw) > COUNT:
                raise ValueError('INVALID_BYBIT_CANDLES')
            converted = []
            for item in raw:
                if not isinstance(item, list) or len(item) < 7:
                    continue
                try:
                    t = int(item[0])
                    o,h,l,c,v,q = [float(x) for x in item[1:7]]
                    if (t % STEP or not start <= t < end or
                            not all(math.isfinite(x) for x in (o,h,l,c,v,q)) or
                            min(o,h,l,c) <= 0 or v < 0 or q < 0 or
                            l > min(o,c) or h < max(o,c)):
                        continue
                    converted.append(tagged(
                        [t, item[1], item[2], item[3], item[4], item[5],
                         t+STEP-1, item[6], '0', '0', '0', '0'], 'BYBIT_LINEAR'))
                except (ValueError, TypeError, OverflowError):
                    continue
            valid = valid_rows(converted, start, end)
            ordered = [valid[t] for t in sorted(valid)]
            assessment = assess_entry(ordered, [], end/1000)
            complete = len(ordered) == COUNT and assessment.get('candles') == COUNT
            reference = {
                'source': 'BYBIT_LINEAR', 'reference_only': True,
                'symbol': symbol, 'as_of': end/1000,
                'expected_candles': COUNT, 'candle_count': len(ordered),
                'complete': complete, 'v2_reference_status': assessment['status'],
                'rsi14': assessment.get('rsi14'),
                'recovery_4h_pct': assessment.get('recovery_4h_pct'),
                'return_5d_pct': assessment.get('return_5d_pct'),
                'strength_score': assessment.get('strength_score'),
            }
            self._write(path, {
                'source':'BYBIT_LINEAR', 'symbol':symbol, 'end_ms':end,
                'fetched_at':now, 'complete':complete,
                'reference':reference, 'closed_15m_ohlcv':ordered,
            })
            self.log(f'[CRASH BYBIT REFERENCE] {symbol}; '
                f'candles={len(ordered)}/{COUNT}; v2={assessment["status"]}; '
                'kaynak=BYBIT_LINEAR; sadece_arastirma=1; binance_mumlari_degismedi')
            return reference
        except Exception as error:
            # Do not print untrusted HTTP response bodies or credentials.
            self.log(f'[CRASH BYBIT WAIT] {symbol}; '
                     f'reason={type(error).__name__}; '
                     f'code={str(error)[:40] if isinstance(error, ValueError) else "TRANSPORT"}; '
                     f'tekrar_deneme_sn={PER_SYMBOL_RETRY_SECONDS}')
            return None
