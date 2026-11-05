"""Opt-in night executor. Importing this module never sends an order.

The caller supplies an exchange adapter and explicitly enables live operation.
Only fresh, Telegram-acknowledged candidates can be submitted.
"""
import fcntl
import hashlib
import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from zoneinfo import ZoneInfo

ZONE = ZoneInfo('Europe/Paris')
D = Decimal
ENTRY_TTL_SECONDS = 8 * 3600
ORDER_POLL_SECONDS = 5 * 60
TP_POLL_SECONDS = 5 * 60
MANUAL_TP_POLL_SECONDS = 5 * 60
LOCK_AUDIT_SECONDS = 30 * 60
ENTRY_EXPIRY_SCAN_SECONDS = 15 * 60
TERMINAL_PHASES = ('DONE', 'NO_FILL', 'CLOSED_EXTERNALLY')


def night(now):
    hour = datetime.fromtimestamp(now, ZONE).hour
    return hour >= 20 or hour < 8


def day(now):
    return datetime.fromtimestamp(now, ZONE).date().isoformat()


def rounded(value, step, up=False):
    return (D(str(value))/D(str(step))).to_integral_value(
        rounding=ROUND_UP if up else ROUND_DOWN)*D(str(step))


def plan(record, info, position_usdt='50'):
    if info.get('status') != 'TRADING' or info.get('contractType') != 'PERPETUAL':
        raise ValueError('SYMBOL_NOT_TRADING_PERPETUAL')
    filters = {f['filterType']: f for f in info['filters']}
    pf, lot = filters['PRICE_FILTER'], filters['LOT_SIZE']
    entry = rounded(record['entry'], pf['tickSize'])
    tp = rounded(max(D(str(record['tp'])), entry*D('1.01')), pf['tickSize'], True)
    if entry <= 0 or tp <= entry:
        raise ValueError('INVALID_PRICE')
    for price in (entry, tp):
        if price < D(pf['minPrice']) or price > D(pf['maxPrice']):
            raise ValueError('PRICE_FILTER')
    budget = D(str(position_usdt))
    if not budget.is_finite() or not 0 < budget <= 1000:
        raise ValueError('INVALID_POSITION_USDT')
    qty = rounded(budget/entry, lot['stepSize'])
    if qty < D(lot['minQty']) or qty > D(lot['maxQty']) or qty <= 0:
        raise ValueError('LOT_SIZE')
    minimum = D(filters.get('MIN_NOTIONAL', {}).get('notional', '0'))
    if qty*entry < minimum:
        raise ValueError('MIN_NOTIONAL')
    return {'price': str(entry), 'tp': str(tp), 'quantity': str(qty)}


ERROR_REASONS = {
    '-1021': 'Sunucu saati ile istek zamanı uyuşmuyor.',
    '-1022': 'API imzası geçersiz; anahtar eşleşmesini kontrol edin.',
    '-2014': 'API anahtar biçimi geçersiz.',
    '-2015': 'API anahtarı, IP izni veya Futures yetkisi uygun değil.',
    '-2013': 'Emir sorgusunda kayıt bulunamadı.',
    '-2019': 'Kullanılabilir marjin yetersiz.',
    '-2022': 'Reduce-only emir reddedildi; pozisyon miktarı ve mevcut kapatma emirlerini kontrol edin.',
    '-1111': 'Fiyat veya miktar hassasiyeti desteklenmiyor.',
    '-1013': 'Emir borsa filtresine uymuyor.',
    '-1015': 'Emir hız sınırı aşıldı.',
    '-1003': 'API istek sınırı aşıldı.',
    '-4014': 'Fiyat tickSize adımına uymuyor.',
    '-4023': 'Miktar stepSize adımına uymuyor.',
    '-4061': 'Emir yönü ile hesap pozisyon modu uyuşmuyor.',
    '-4164': 'Emir minimum tutar koşuluna uymuyor.',
    '-5022': 'Maker TP hemen gerçekleşeceği için reddedildi; hedefe ulaşılmış olabilir.',
}


class BrokerError(Exception):
    """Only a numeric API code, never raw API messages or signed URLs."""
    def __init__(self, code, uncertain=False, deferred_reason=None, deferred_seconds=None, not_sent=False):
        raw = str(code).removeprefix('BINANCE_')
        self.code = raw if raw.lstrip('-').isdigit() else 'UNKNOWN'
        self.uncertain = uncertain or self.code == 'UNKNOWN'
        # Set ONLY when the local scheduler refused a request before HTTP.
        # Such a deferral must not be reported as an exchange -1003 response.
        self.deferred_reason = deferred_reason
        self.deferred_seconds = deferred_seconds
        self.not_sent = bool(not_sent)  # Raised before HTTP, unlike an uncertain exchange response.
        super().__init__('BINANCE_'+self.code)


def error_details(error):
    if isinstance(error, BrokerError):
        if error.deferred_reason:
            return ('Kod: LOCAL_IP_BUDGET | Neden: Binance isteği gönderilmedi; '
                    f'yerel koruma {error.deferred_reason}, '
                    f'yaklaşık {max(1, int(float(error.deferred_seconds or 0) + 0.999))} sn bekleme.')
        reason = ERROR_REASONS.get(error.code, 'Binance yanıtı doğrulanamadı.' if error.uncertain else 'Binance isteği reddetti; hata kodunu kontrol edin.')
        return f'Kod: {error.code} | Neden: {reason}'
    local = {'TP_NOT_ACTIVE': 'TP emri iptal edilmiş, süresi dolmuş veya reddedilmiş.',
             'TP_PROPERTIES_NOT_CONFIRMED': 'TP yönü, fiyatı, miktarı veya reduce-only özelliği uyuşmuyor.',
             'ENTRY_RESULT_NOT_FINAL': 'Giriş emrinin nihai sonucu henüz doğrulanamadı.'}
    if isinstance(error, ValueError) and str(error) in local:
        return 'Kod: '+str(error)+' | Neden: '+local[str(error)]
    return 'Kod: LOCAL_CHECK | Neden: Yerel kontrol başarısız ('+type(error).__name__+').' 


class BinanceBroker:
    def __init__(self, key, secret):
        if not key or not secret:
            raise ValueError('MISSING_BINANCE_CREDENTIALS')
        self.key, self.secret = key, secret
        self.cooldown_until = 0
        self.rate_failures = 0

    def request(self, method, path, params=None, signed=True, budget_priority=None):
        import requests
        import hmac
        from urllib.parse import urlencode
        if time.time() < self.cooldown_until:
            raise BrokerError('-1003', not_sent=True)
        from crash_rest_budget import BUDGET, RateDeferred, estimated_weight, priority_for
        params = dict(params or {})
        # Reserve before signing or sending. Local deferrals NEVER issue a
        # Binance request and NEVER masquerade as a successful BUY/TP.
        if budget_priority not in (None, 'protect'):
            raise ValueError('INVALID_BUDGET_PRIORITY')
        # Protection overrides are restricted to verified position/order queries.
        # New BUY and preflight requests cannot consume TP-reserved capacity.
        if budget_priority == 'protect' and not (method == 'GET' and path in (
                '/fapi/v3/positionRisk', '/fapi/v1/openOrders',
                '/fapi/v1/openAlgoOrders', '/fapi/v1/exchangeInfo',
                '/fapi/v1/ticker/bookTicker')):
            raise ValueError('UNSAFE_PROTECTION_PRIORITY')
        try:
            BUDGET.reserve(budget_priority or priority_for(method, path, params),
                           estimated_weight(path, params, method=method))
        except RateDeferred as deferred:
            raise BrokerError('-1003', deferred_reason=deferred.reason,
                              deferred_seconds=deferred.seconds, not_sent=True) from None
        if signed:
            params.update(timestamp=int(time.time()*1000), recvWindow=5000)
        query = urlencode(params)
        if signed:
            query += '&signature='+hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        try:
            response = requests.request(method, 'https://fapi.binance.com'+path+'?'+query,
                headers={'X-MBX-APIKEY': self.key}, timeout=(5, 10))
            try:
                body = response.json()
            except ValueError:
                body = {}
        except Exception:
            raise BrokerError('UNKNOWN', uncertain=True) from None
        BUDGET.observe(response.headers, response.status_code,
                       response.headers.get("Retry-After", 0))
        api_code = str(body.get('code', '')) if isinstance(body, dict) else ''
        if response.status_code in (429, 418) or api_code == '-1003':
            self.rate_failures += 1
            delay = min(3600, 120 * 2**min(self.rate_failures-1, 5))
            if response.status_code == 418:
                delay = max(delay, 900)
            try:
                delay = max(delay, float(response.headers.get('Retry-After', 0)))
            except (ValueError, TypeError):
                pass
            ban = re.search(r'ban(?:ned)? until (\d{13})', str(body.get('msg', '')), re.I) if isinstance(body, dict) else None
            if ban:
                delay = max(delay, int(ban.group(1))/1000-time.time()+5)
            self.cooldown_until = time.time()+delay
            raise BrokerError('-1003')
        if response.status_code >= 400:
            code = body.get('code', 'UNKNOWN') if isinstance(body, dict) else 'UNKNOWN'
            raise BrokerError(code, uncertain=response.status_code >= 500 or str(code) in ('-1006', '-1007'))
        return body

    def positions(self):
        # Only used by TP-repair scans. Always read LIVE positions.
        result = self.request('GET', '/fapi/v3/positionRisk',
                              budget_priority='protect')
        if not isinstance(result, list):
            raise ValueError('INVALID_POSITION_RESPONSE')
        return result

    def closing_orders(self, symbol):
        normal = self.request('GET', '/fapi/v1/openOrders', {'symbol': symbol},
                              budget_priority='protect')
        algo = self.request('GET', '/fapi/v1/openAlgoOrders', {'symbol': symbol},
                            budget_priority='protect')
        if not isinstance(normal, list) or not isinstance(algo, list):
            raise ValueError('INVALID_ORDER_RESPONSE')
        return normal + algo

    def best_bid(self, symbol):
        result = self.request('GET', '/fapi/v1/ticker/bookTicker', {'symbol': symbol},
                              signed=False, budget_priority='protect')
        return D(result['bidPrice'])

    def repair_tp(self, symbol, qty, price, client_id, position_side="BOTH"):
        # Maker-only: never deliberately sell immediately after a missed target.
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='SELL', type='LIMIT',
            timeInForce='GTX', quantity=qty, price=price, newClientOrderId=client_id,
            **self.close_params(position_side)))

    def protected_info(self, symbol):
        # Only use from an existing-position TP-repair flow.
        return self.info(symbol, budget_priority='protect')

    def info(self, symbol, budget_priority=None):
        # Exchange filters rarely change. Reuse the validated snapshot for
        # 30 minutes instead of downloading ALL symbols for every candidate.
        # Never cache balances, positions, open orders or TP verification.
        now = time.time()
        cached = getattr(self, '_exchange_info_cache', None)
        if cached is None or now >= cached[0]:
            data = self.request('GET', '/fapi/v1/exchangeInfo',
                                signed=False, budget_priority=budget_priority)
            symbols = data.get('symbols') if isinstance(data, dict) else None
            if not isinstance(symbols, list) or not symbols:
                raise ValueError('INVALID_EXCHANGE_INFO')
            lookup = {item['symbol']: item for item in symbols
                      if isinstance(item, dict) and isinstance(item.get('symbol'), str)}
            if not lookup:
                raise ValueError('INVALID_EXCHANGE_INFO')
            self._exchange_info_cache = (now + 1800, lookup)
            cached = self._exchange_info_cache
        return cached[1][symbol]

    @staticmethod
    def close_params(position_side):
        if position_side == 'LONG':
            return {'positionSide': 'LONG'}
        if position_side == 'BOTH':
            return {'positionSide': 'BOTH', 'reduceOnly': 'true'}
        raise ValueError('INVALID_POSITION_SIDE')

    @staticmethod
    def is_closing_order(order):
        # Hedge closes are identified by positionSide + direction, not reduceOnly.
        if order.get('reduceOnly') in (True, 'true') or order.get('closePosition') in (True, 'true'):
            return True
        return ((order.get('positionSide') == 'LONG' and order.get('side') == 'SELL')
                or (order.get('positionSide') == 'SHORT' and order.get('side') == 'BUY'))

    def check_slots(self, active, orders):
        closing = [o for o in orders if self.is_closing_order(o)]
        pending = {o['symbol'] for o in orders if not self.is_closing_order(o)}
        total = len(active | pending)
        limit = getattr(self, 'max_symbols', 5)
        if total >= limit:
            raise ValueError(f'MAX_{limit}_WITH_PENDING_ORDERS | açık={len(active)} giriş={len(pending-active)} toplam={total} kapatma_hariç={len(closing)}')

    def preflight(self, symbol):
        dual = self.request('GET', '/fapi/v1/positionSide/dual').get('dualSidePosition')
        if type(dual) is not bool:
            raise ValueError('INVALID_POSITION_MODE')
        position_side = 'LONG' if dual else 'BOTH'
        positions = self.request('GET', '/fapi/v3/positionRisk')
        active = {p['symbol'] for p in positions if D(p['positionAmt']) != 0}
        if symbol in active or len(active) >= getattr(self, 'max_symbols', 5):
            raise ValueError(f"POSITION_EXISTS_OR_MAX_{getattr(self, 'max_symbols', 5)}")
        orders = self.request('GET', '/fapi/v1/openOrders')
        if any(o['symbol'] == symbol for o in orders):
            raise ValueError('EXISTING_SYMBOL_ORDER')
        # Pending orders on other symbols can also occupy future position slots.
        self.check_slots(active, orders)
        algo_orders = self.request('GET', '/fapi/v1/openAlgoOrders')
        if not isinstance(algo_orders, list):
            raise ValueError('INVALID_ALGO_ORDER_RESPONSE')
        if any(o['symbol'] == symbol for o in algo_orders):
            raise ValueError('EXISTING_SYMBOL_ALGO_ORDER')
        self.check_slots(active, orders+algo_orders)
        # Operator visibility without another REST call or any cancellation.
        # Only explicitly tagged bot BUY orders count as the bot's own pending
        # entries; manual orders and closing/TP orders are never modified.
        bot_pending = [o for o in orders if
                       o.get('side') == 'BUY' and o.get('type') == 'LIMIT' and
                       re.fullmatch(r'cbn[0-9a-f]{24}e',
                                    str(o.get('clientOrderId', '')))]
        print(f'[CRASH ENTRY PREFLIGHT] {symbol}; active_symbols={len(active)}; '
              f'normal_open={len(orders)}; algo_open={len(algo_orders)}; '
              f'bot_pending_buy={len(bot_pending)}; '
              f'symbol_limit={getattr(self, "max_symbols", 5)}; '
              'no orders cancelled', flush=True)
        result = self.request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': 10})
        if result.get('leverage') != 10:
            raise ValueError('LEVERAGE_NOT_CONFIRMED')
        return position_side

    def order(self, symbol, client_id):
        return self.request('GET', '/fapi/v1/order', {'symbol': symbol, 'origClientOrderId': client_id})

    def open_entry_orders(self):
        result = self.request('GET', '/fapi/v1/openOrders')
        if not isinstance(result, list):
            raise ValueError('INVALID_ORDER_RESPONSE')
        return result

    def cancel_entry(self, symbol, client_id):
        return self.request('DELETE', '/fapi/v1/order', {'symbol': symbol, 'origClientOrderId': client_id})

    def cancel_owned_tp(self, symbol, client_id):
        # Never cancel an operator's order. The exact bot TP client ID is
        # checked again against persisted trade state before calling this.
        if not re.fullmatch(r'cbn[0-9a-f]{24}t', str(client_id)):
            raise ValueError('INVALID_OWNED_TP_ID')
        return self.request('DELETE', '/fapi/v1/order',
                            {'symbol': symbol, 'origClientOrderId': client_id})

    def buy(self, symbol, p, client_id):
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='BUY', type='LIMIT',
            timeInForce='GTC', quantity=p['quantity'], price=p['price'], newClientOrderId=client_id,
            newOrderRespType='RESULT', positionSide=p.get('position_side', 'BOTH')))

    def tp(self, symbol, qty, price, client_id, position_side="BOTH"):
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='SELL', type='LIMIT',
            timeInForce='GTC', quantity=qty, price=price, newClientOrderId=client_id,
            **self.close_params(position_side)))


class NightExecutor:
    def __init__(self, directory, broker, deliver, log, clock=time.time, enabled=False, max_daily=20, manual_tp=False):
        self.broker, self.deliver, self.log, self.clock = broker, deliver, log, clock
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, 'crash_night_state.json')
        self.guard = open(self.path+'.lock', 'a')
        try:
            fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = {'orders': {}, 'outbox': [], 'halt': None}
            marker = self.path+'.initialized'
            if os.path.exists(self.path):
                with open(self.path, encoding='utf-8') as handle:
                    self.state = json.load(handle)
                if not isinstance(self.state.get('orders'), dict) or not isinstance(self.state.get('outbox'), list):
                    raise ValueError('INVALID_EXECUTOR_STATE')
            elif os.path.exists(marker):
                raise ValueError('EXECUTOR_STATE_MISSING')
            self.state.pop('unverified_day', None)
            self.state.setdefault('symbol_quarantine', {})
            if not isinstance(self.state['symbol_quarantine'], dict):
                raise ValueError('INVALID_SYMBOL_QUARANTINE')
            # Apply new defaults exactly once; later Telegram changes must survive redeploys.
            if not self.state.get('trade_settings_defaults_20261101'):
                self.state['trade_settings'] = {'max_daily': max_daily, 'max_symbols': 20,
                                                'position_usdt': '500'}
                self.state['trade_settings_defaults_20261101'] = True
                self.state['paused'] = not enabled
            else:
                # Python's off switch always wins; preserve explicit Telegram /otomatik_kapat.
                self.state['paused'] = not enabled or bool(self.state.get('paused', False))
            self.save()
            with open(marker, 'a') as handle:
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.guard.close()
            raise
        settings = self.state.get('trade_settings', {})
        self.max_daily = int(settings.get('max_daily', max_daily))
        self.max_symbols = int(settings.get('max_symbols', 20))
        self.broker.max_symbols = self.max_symbols
        self.position_usdt = D(str(settings.get('position_usdt', '500')))
        if not 1 <= self.max_symbols <= 20 or not 1 <= self.max_daily <= 20 or not self.position_usdt.is_finite() or not 1 <= self.position_usdt <= 1000:
            self.guard.close()
            raise ValueError('INVALID_TRADE_SETTINGS')
        self.manual_tp_enabled = manual_tp
        self.manual_guard = None
        self.started = clock()
        self.queue = queue.Queue(maxsize=100)
        # Short, in-memory retries ONLY for locally deferred, pre-HTTP
        # preflight. Never retry a BUY whose HTTP request may have reached
        # Binance. On restart an unsent candidate may be safely discarded.
        self.preflight_retries = {}
        self.thread = None
        self.next_poll = 0
        self.next_expiry_scan = 0
        self.next_manual_scan = 0
        self.next_halt_recheck = 0
        self.next_quarantine_scan = 0
        self.next_manual_close_scan = 0
        self.next_lock_audit = 0
        self.skip_notice_after = {}
        self.protected_poll = {}
        self.enabled = threading.Event()
        self.armed_at = clock()
        if enabled and not self.state['paused']:
            self.enabled.set()
        self.controls = queue.Queue(maxsize=10)
        self.account_stream = None
        self.account_events = queue.Queue(maxsize=2000)
        self.account_resync = threading.Event()
        self.account_versions = {}
        self.account_resync_after = 0
        self.ws_manual_symbols = set()
        self.ws_manual_after = 0

    def save(self):
        with open(self.path+'.tmp', 'w', encoding='utf-8') as handle:
            json.dump(self.state, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self.path+'.tmp', self.path)

    def daily_count(self):
        today = day(self.clock())
        raw = sum(o['day'] == today and not o.get('entry_never_sent', False)
                  for o in list(self.state['orders'].values()))
        adjustment = self.state.get('daily_counter_override', {})
        if adjustment.get('day') == today:
            return max(0, adjustment['count'] + raw - adjustment['raw_at_set'])
        return raw

    def summary(self):
        count = self.daily_count()
        stream = self.account_stream.summary() if self.account_stream else 'Hesap WS: baslatilmadi / REST yedek'
        pending_check = sum(o.get('phase') == 'ENTRY_UNKNOWN'
                            for o in self.state['orders'].values())
        quarantined = ','.join(sorted(self.state['symbol_quarantine'])) or 'yok'
        return f"{'DURDU: '+self.state['halt'] if self.state['halt'] else ('AÇIK' if self.enabled.is_set() else 'KAPALI')} | bugün {count}/{self.max_daily} | sembol sınırı {self.max_symbols} | pozisyon {self.position_usdt} USDT | 10x | 24 saat | giriş mutabakatı: {pending_check} (yerel kayıt) | karantina: {quarantined} | {stream}"

    def apply_control(self, action):
        if action.startswith(('limit:', 'amount:', 'symbols:')):
            kind, raw = action.split(':', 1)
            try:
                value = D(raw)
                ceiling = {'amount':1000, 'symbols':20, 'limit':20}[kind]
                if not value.is_finite() or not 1 <= value <= ceiling or (kind != 'amount' and value != value.to_integral_value()):
                    raise ValueError('range')
            except (ValueError, ArithmeticError):
                return '⚠️ Kullanım: /sembol 1–20, /limit 1–20 veya /tutar 1–1000 (USDT toplam pozisyon, teminat değil).'
            settings = dict(self.state.get('trade_settings', {}))
            key = {'limit':'max_daily', 'amount':'position_usdt', 'symbols':'max_symbols'}[kind]
            settings[key] = str(value) if kind == 'amount' else int(value)
            self.state['trade_settings'] = settings
            self.save()
            if kind == 'limit':
                self.max_daily = int(value)
            elif kind == 'symbols':
                self.max_symbols = int(value)
                self.broker.max_symbols = self.max_symbols
            else:
                self.position_usdt = value
            return (f'✅ Yeni giriş ayarı kaydedildi. Kullanılan sayaç, mevcut pozisyonlar ve bekleyen emirler değişmedi. Aynı anda sembol sınırı: {self.max_symbols}.\n'+self.summary())
        if action.startswith('counter:'):
            value = action.split(':', 1)[1]
            if not value.isascii() or not value.isdigit() or not 0 <= int(value) <= self.max_daily:
                return f'⚠️ Kullanım: /sayac 0–{self.max_daily}. Bu değer bugünkü kullanılan giriş sayısıdır.'
            today = day(self.clock())
            before = self.daily_count()
            self.state['daily_counter_override'] = dict(day=today, count=int(value),
                raw_at_set=sum(o['day'] == today and not o.get('entry_never_sent', False)
                               for o in self.state['orders'].values()))
            self.save()
            return (f'✅ Günlük kullanılan giriş sayısı: {before} → {int(value)} / {self.max_daily}\n'
                    f'Paris 00:00 itibarıyla yeni gün sayacı 0 olur. Açık/bekleyen emirler değişmedi; aynı anda sembol sınırı: {self.max_symbols}.')
        if action == 'on':
            if self.state['halt']:
                return '⚠️ Açılamadı: '+self.state['halt']+'. Binance pozisyonları ve TP emirleri kontrol edilmeli. Komut hata kilidini kaldırmaz.'
            self.enabled.clear()
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.state['paused'] = False
            self.save()
            self.armed_at = self.clock()
            self.enabled.set()
            return ('✅ Otomatik girişler AÇIK. Yalnız yeni giriş geçişleri işlenecek.\n'
                    f'24 saat | 10x | en fazla {self.position_usdt} USDT pozisyon / yaklaşık {self.position_usdt/10} USDT teminat\n'
                    f'Günde en fazla {self.max_daily} giriş denemesi, aynı anda en fazla {self.max_symbols} sembol. TP +%1; SL yok.\n'+self.summary())
        if action == 'off':
            self.enabled.clear()
            self.state['paused'] = True
            self.save()
            return '⏸️ Yeni girişler KAPALI. Mevcut pozisyonlar kapatılmadı; TP takibi sürüyor. Daha önce gönderilmiş giriş emri sonuçlanabilir.'
        return self.summary()

    def request_control(self, action):
        if action == 'status':
            return self.summary()
        if action == 'off':
            # Block further preflight/POST checks without waiting for REST reconciliation.
            self.enabled.clear()
        done = threading.Event()
        result = {}
        try:
            self.controls.put_nowait((action, done, result))
        except queue.Full:
            return '⚠️ Komut kuyruğu dolu. /otomatik_durum ile kontrol edin.'
        if not done.wait(20):
            return '⏳ Komut işlenmeyi bekliyor. /otomatik_durum ile sonucu kontrol edin.'
        return result['message']

    def alert(self, text):
        self.state['outbox'].append({'text': text, 'retry_at': 0})
        self.save()

    def halt(self, reason, symbol):
        self.state['halt'] = reason
        self.alert(f'⚠️ OTOMATİK İŞLEM | {symbol}\n{reason}\nYeni girişler durdu. Mevcut pozisyon otomatik kapatılmadı; Binance üzerinden kontrol edin.')

    def recover_mixed_halt(self):
        """Resolve simultaneous TP and BUY uncertainty only from live Binance proof.

        Two legacy recovery methods individually reject the other's unknown
        phase. Verify BOTH atomically to avoid a permanent manual-close lock.
        Never send or cancel orders, and never infer flatness from WS deltas.
        """
        reason = self.state.get('halt')
        if not isinstance(reason, str) or not reason.startswith(
                ('TP DOĞRULANAMADI', 'GİRİŞ SONUCU BELİRSİZ')):
            return False
        orders = list(self.state['orders'].values())
        unknown = [o for o in orders if o.get('phase') == 'ENTRY_UNKNOWN']
        tp_affected = [o for o in orders if o.get('warned') and
                       o.get('phase') in ('TP_UNKNOWN', 'PROTECTED', 'DONE')]
        if not unknown or not tp_affected:
            return False
        if (any(not o.get('warned') or not o.get('symbol') or
                not o.get('entry_id') or not o.get('quantity') for o in unknown)
                or any(o.get('phase') == 'TP_UNKNOWN' and not o.get('warned')
                       for o in orders)
                or any(not o.get('symbol') for o in tp_affected)):
            return False

        symbols = {o['symbol'] for o in unknown + tp_affected}
        resolved = []
        try:
            positions = self.broker.positions()
            if not isinstance(positions, list):
                raise ValueError('INVALID_POSITION_RESPONSE')
            for position in positions:
                if position.get('symbol') in symbols and D(str(position['positionAmt'])) != 0:
                    return False
            for symbol in sorted(symbols):
                # Includes regular AND conditional/algo orders.
                open_orders = self.broker.closing_orders(symbol)
                if not isinstance(open_orders, list):
                    raise ValueError('INVALID_ORDER_RESPONSE')
                if open_orders:
                    return False

            for order in unknown:
                try:
                    entry = self.broker.order(order['symbol'], order['entry_id'])
                except BrokerError as error:
                    if error.code == '-2013' and not error.uncertain:
                        resolved.append((order, 'NO_FILL', 'ORDER_NOT_FOUND_FLAT'))
                        continue
                    raise
                if (not isinstance(entry, dict)
                        or entry.get('symbol') != order['symbol']
                        or entry.get('clientOrderId') != order['entry_id']
                        or entry.get('side') != 'BUY'):
                    raise ValueError('ENTRY_IDENTITY_NOT_CONFIRMED')
                status = entry.get('status')
                filled = D(str(entry['executedQty']))
                if filled < 0 or filled > D(str(order['quantity'])):
                    raise ValueError('INVALID_FILLED_QUANTITY')
                if status in ('CANCELED', 'EXPIRED', 'EXPIRED_IN_MATCH', 'REJECTED') and filled == 0:
                    resolved.append((order, 'NO_FILL', 'ORDER_FINAL_ZERO_FILL'))
                elif status in ('FILLED', 'CANCELED', 'EXPIRED', 'EXPIRED_IN_MATCH') and filled > 0:
                    resolved.append((order, 'CLOSED_EXTERNALLY', 'LIVE_FLAT_AFTER_FILL'))
                else:
                    return False
        except Exception as error:
            code = error.code if isinstance(error, BrokerError) else type(error).__name__
            self.log(f'[CRASH MIXED HALT REVIEW WAIT] code={code}; halt retained')
            return False

        now = self.clock()
        for order, phase, proof in resolved:
            order['phase'] = phase
            order['entry_reconcile_proof'] = proof
            order['flat_confirmed_at'] = now
        for order in tp_affected:
            if order['phase'] != 'DONE':
                order['phase'] = 'CLOSED_EXTERNALLY'
                order['flat_confirmed_at'] = now
                order['close_verification'] = 'LIVE_POSITION_ZERO_NO_OPEN_ORDERS'
        self.state['halt'] = None
        self.armed_at = now
        self.preflight_retries.clear()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.save()
        labels = ', '.join(sorted(symbols))
        self.log(f'[CRASH MIXED HALT CLEARED] {labels}; live flat, no normal/algo '
                 'orders and original BUY client IDs reconciled; old signals discarded')
        self.alert(f'✅ KARMA İŞLEM KİLİDİ TEMİZLENDİ | {labels}\n'
                   'Binance canlı kontrolünde açık pozisyon/emir bulunmadı; '
                   'belirsiz alışlar orijinal emir kimliğiyle doğrulandı. '
                   'TP dolumu varsayılmadı. Otomatik ayar korunuyor, '
                   'yalnız yeni sinyaller işlenebilir.')
        return True

    @staticmethod
    def _flat_from_snapshot(positions, symbols):
        """Fresh, signed exchange position state; any hedge exposure blocks close."""
        if not isinstance(positions, list):
            raise ValueError('INVALID_POSITION_RESPONSE')
        for position in positions:
            if not isinstance(position, dict):
                raise ValueError('INVALID_POSITION_RESPONSE')
            if position.get('symbol') in symbols:
                amount = D(str(position['positionAmt']))
                if not amount.is_finite():
                    raise ValueError('INVALID_POSITION_AMOUNT')
                if amount != 0:
                    return False
        return True

    def _finish_flat_symbol(self, symbol):
        """Finish an operator close without any persistent symbol quarantine.

        Cancel only an EXACT tracked bot TP, confirmed as a reduce-only LIMIT
        SELL and only after a live-flat position snapshot. Operator-created
        orders are never altered; ordinary entry preflight blocks a second
        BUY on their symbol. Any unresolved bot BUY or API ambiguity fails
        closed and is retried at the normal five-minute cadence.
        """
        try:
            if any(o.get('symbol') == symbol and o.get('phase') == 'ENTRY_UNKNOWN'
                   for o in self.state['orders'].values()):
                self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; ENTRY_UNKNOWN')
                return None
            if not self._flat_from_snapshot(self.broker.positions(), {symbol}):
                self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; POSITION_OPEN')
                return None
            opened = self.broker.closing_orders(symbol)
            if not isinstance(opened, list):
                raise ValueError('INVALID_ORDER_RESPONSE')
            tracked = {
                o.get('tp_id'): o for o in self.state['orders'].values()
                if o.get('symbol') == symbol and o.get('tp_id')
            }
            own_tps = []
            for live in opened:
                if (not isinstance(live, dict) or live.get('symbol') != symbol
                        or live.get('side') not in ('BUY', 'SELL')):
                    self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; INVALID_LIVE_ORDER')
                    return None
                cid = str(live.get('clientOrderId', ''))
                if live['side'] == 'BUY' and re.fullmatch(r'cbn[0-9a-f]{24}e', cid):
                    self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; BOT_BUY_STILL_OPEN')
                    return None
                tracked_tp = tracked.get(cid)
                if tracked_tp is not None:
                    position_side = tracked_tp.get('position_side', 'BOTH')
                    if (live.get('side') != 'SELL'
                            or live.get('type') != 'LIMIT'
                            or live.get('status') not in ('NEW', 'PARTIALLY_FILLED')
                            or live.get('positionSide', 'BOTH') != position_side
                            or (position_side == 'BOTH'
                                and live.get('reduceOnly') not in (True, 'true'))
                            or not re.fullmatch(r'cbn[0-9a-f]{24}t', cid)
                            or D(str(live.get('price', 'NaN'))) != D(str(tracked_tp.get('tp', 'NaN')))
                            or D(str(live.get('origQty', 'NaN'))) != D(str(tracked_tp.get('filled_qty', 'NaN')))):
                        self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; '
                                 'TRACKED_TP_PROPERTIES_MISMATCH')
                        return None
                    own_tps.append(cid)
            if own_tps:
                # Recheck before touching ONLY the exact orphaned bot TP.
                if not self._flat_from_snapshot(self.broker.positions(), {symbol}):
                    self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; POSITION_CHANGED')
                    return None
                for cid in own_tps:
                    try:
                        self.broker.cancel_owned_tp(symbol, cid)
                    except Exception as error:
                        # The DELETE result could be lost; never assume it
                        # succeeded. A fresh open-order query below decides.
                        code = error.code if isinstance(error, BrokerError) else type(error).__name__
                        self.log(f'[CRASH MANUAL TP CANCEL CHECK] {symbol}; code={code}')
                opened = self.broker.closing_orders(symbol)
                if not isinstance(opened, list):
                    raise ValueError('INVALID_ORDER_RESPONSE')
                if any(not isinstance(o, dict) or o.get('symbol') != symbol
                       or o.get('side') not in ('BUY', 'SELL') for o in opened):
                    raise ValueError('INVALID_ORDER_RESPONSE')
                if any(o.get('clientOrderId') in own_tps for o in opened):
                    self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; BOT_TP_STILL_OPEN')
                    return None
                # Do not permit a newly surfaced bot BUY to be overlooked.
                if any(o.get('side') == 'BUY' and re.fullmatch(
                        r'cbn[0-9a-f]{24}e', str(o.get('clientOrderId', '')))
                       for o in opened):
                    self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; BOT_BUY_STILL_OPEN')
                    return None
            if not self._flat_from_snapshot(self.broker.positions(), {symbol}):
                self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; POSITION_CHANGED')
                return None
            remaining = len(opened)
            if own_tps:
                self.log(f'[CRASH MANUAL TP CLEANUP] {symbol}; '
                         f'cancel_verified={len(own_tps)}; other_orders={remaining}')
            if remaining:
                self.log(f'[CRASH MANUAL ORDERS KEPT] {symbol}; count={remaining}; '
                         'operator orders untouched; ordinary preflight blocks duplicate BUY')
            return remaining
        except Exception as error:
            code = error.code if isinstance(error, BrokerError) else type(error).__name__
            self.log(f'[CRASH MANUAL CLOSE WAIT] {symbol}; VERIFY_FAILED={code}')
            return None

    def review_quarantined_symbols(self):
        """Migrate old quarantine records to normal exchange preflight rules."""
        for symbol in list(self.state['symbol_quarantine']):
            remaining = self._finish_flat_symbol(symbol)
            if remaining is None:
                continue
            del self.state['symbol_quarantine'][symbol]
            for order in self.state['orders'].values():
                if (order.get('symbol') == symbol
                        and order.get('phase') == 'CLOSED_EXTERNALLY'
                        and order.get('close_verification') == 'LIVE_POSITION_ZERO_QUARANTINED'):
                    order['close_verification'] = (
                        'LIVE_POSITION_ZERO_MANUAL_ORDERS_KEPT'
                        if remaining else 'LIVE_POSITION_ZERO_NO_OPEN_ORDERS')
            self.save()
            self.log(f'[CRASH LEGACY QUARANTINE CLEARED] {symbol}; '
                     f'flat confirmed; manual_open_orders={remaining}; no new quarantine')
            message = (f'✅ ESKİ COIN KARANTİNASI KALDIRILDI | {symbol}\n'
                       'Pozisyon kapalı doğrulandı. Botun eski TP emri varsa '
                       'yalnız kendi emri temizlendi. Manuel emirler korundu; '
                       'aynı coinde yeni alış Binance ön kontrolüne tabidir.')
            threading.Thread(target=self._deliver_skip_notice,
                             args=(message,), name='crash-legacy-quarantine-clear',
                             daemon=True).start()

    def recover_closed_tp_halt(self):
        """Resolve a stale TP-only global halt without creating quarantines.

        Only a Binance-confirmed flat affected symbol is settled. Any open,
        unverified, or partially closed position and any bot BUY uncertainty
        preserve the safety stop; manually placed orders are NOT cancelled.
        """
        reason = self.state.get('halt')
        if not isinstance(reason, str) or not reason.startswith('TP DOĞRULANAMADI'):
            return False
        orders = list(self.state['orders'].values())
        if any(o.get('phase') == 'ENTRY_UNKNOWN' for o in orders):
            self.log('[CRASH TP HALT BLOCK] ENTRY_UNKNOWN; original BUY not reconciled')
            return False
        affected = [o for o in orders if o.get('warned') and
                    (o.get('phase') in ('TP_UNKNOWN', 'PROTECTED', 'DONE')
                     or (o.get('phase') == 'CLOSED_EXTERNALLY' and
                         o.get('close_verification') in (
                             'LIVE_POSITION_ZERO_NO_OPEN_ORDERS',
                             'LIVE_POSITION_ZERO_QUARANTINED',
                             'LIVE_POSITION_ZERO_MANUAL_ORDERS_KEPT'))) and
                    isinstance(o.get('symbol'), str) and o['symbol']]
        if not affected:
            self.log('[CRASH TP HALT BLOCK] NO_VERIFIABLE_AFFECTED_ORDER')
            return False
        if any(o.get('phase') == 'TP_UNKNOWN' and not o.get('warned') for o in orders):
            self.log('[CRASH TP HALT BLOCK] UNWARNED_TP_UNKNOWN')
            return False
        remaining_by_symbol = {}
        symbols = sorted({o['symbol'] for o in affected})
        for symbol in symbols:
            remaining = self._finish_flat_symbol(symbol)
            if remaining is None:
                self.log(f'[CRASH TP HALT BLOCK] {symbol}; CLOSE_NOT_VERIFIED')
                return False
            remaining_by_symbol[symbol] = remaining

        now = self.clock()
        for order in affected:
            if order['phase'] != 'DONE':
                order['phase'] = 'CLOSED_EXTERNALLY'
                order['flat_confirmed_at'] = now
                order['close_verification'] = (
                    'LIVE_POSITION_ZERO_MANUAL_ORDERS_KEPT'
                    if remaining_by_symbol[order['symbol']]
                    else 'LIVE_POSITION_ZERO_NO_OPEN_ORDERS')
        for symbol in symbols:
            self.state['symbol_quarantine'].pop(symbol, None)
        self.state['halt'] = None
        self.armed_at = now
        self.preflight_retries.clear()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.save()
        self.log(f'[CRASH TP HALT CLEARED] manual-close verified: '
                 f'{",".join(symbols)}; no quarantine; '
                 f'manual_orders={sum(remaining_by_symbol.values())}; '
                 'old signals discarded; executor on/off setting unchanged')
        note = (f'✅ MANUEL KAPANIŞ TAMAMLANDI | {",".join(symbols)}\n'
                'Binance pozisyonları sıfır doğrulandı. Botun kalan TP emri varsa '
                'temizlendi; manuel emirler korunuyor. Karantina oluşturulmadı. '
                'Otomatik ayar ve yeni sinyal kuralları değişmedi.')
        threading.Thread(target=self._deliver_skip_notice, args=(note,),
                         name='crash-manual-close-verified', daemon=True).start()
        return True

    def recover_unknown_entry_halt(self):
        """Resolve a previous ENTRY_UNKNOWN hold only from live Binance proof.

        No orders are submitted or cancelled here. A missing account response,
        a pending entry, or unverified fills must leave the safety halt intact.
        """
        reason = self.state.get('halt')
        if not isinstance(reason, str) or not reason.startswith('GİRİŞ SONUCU BELİRSİZ'):
            return False
        records = list(self.state['orders'].values())
        unknown = [o for o in records if o.get('phase') == 'ENTRY_UNKNOWN']
        if (not unknown or any(not o.get('warned') or not o.get('symbol') or
                               not o.get('entry_id') for o in unknown)
                or any(o.get('phase') == 'TP_UNKNOWN' for o in records)):
            return False
        symbols = {o['symbol'] for o in unknown}
        try:
            positions = self.broker.positions()
            if not isinstance(positions, list):
                raise ValueError('INVALID_POSITION_RESPONSE')
            for p in positions:
                if p.get('symbol') in symbols and D(str(p['positionAmt'])) != 0:
                    return False
            for symbol in sorted(symbols):
                opened = self.broker.closing_orders(symbol)
                if not isinstance(opened, list):
                    raise ValueError('INVALID_ORDER_RESPONSE')
                if opened:  # Includes regular BUY and algo orders.
                    return False
            resolved = []
            for order in unknown:
                try:
                    entry = self.broker.order(order['symbol'], order['entry_id'])
                except BrokerError as error:
                    if error.code == '-2013' and not error.uncertain:
                        # The original entry ID is absent; account is flat
                        # and there are no open orders for its symbol.
                        resolved.append((order, 'NO_FILL', 'ORDER_NOT_FOUND_FLAT'))
                        continue
                    raise
                if (not isinstance(entry, dict)
                        or entry.get('symbol') != order['symbol']
                        or entry.get('clientOrderId') != order['entry_id']
                        or entry.get('side') != 'BUY'):
                    raise ValueError('ENTRY_IDENTITY_NOT_CONFIRMED')
                status = entry.get('status')
                filled = D(str(entry['executedQty']))
                if filled < 0 or filled > D(str(order['quantity'])):
                    raise ValueError('INVALID_FILLED_QUANTITY')
                if status in ('CANCELED', 'EXPIRED', 'EXPIRED_IN_MATCH', 'REJECTED') and filled == 0:
                    resolved.append((order, 'NO_FILL', 'ORDER_FINAL_ZERO_FILL'))
                elif status in ('FILLED', 'CANCELED', 'EXPIRED', 'EXPIRED_IN_MATCH') and filled > 0:
                    # Only the account is confirmed flat, not a winning TP.
                    resolved.append((order, 'CLOSED_EXTERNALLY', 'LIVE_FLAT_AFTER_FILL'))
                else:
                    return False
        except Exception as error:
            code = error.code if isinstance(error, BrokerError) else type(error).__name__
            self.log(f'[CRASH ENTRY HALT REVIEW WAIT] code={code}; safety hold retained')
            return False
        now = self.clock()
        for order, phase, proof in resolved:
            order['phase'] = phase
            order['entry_reconcile_proof'] = proof
            order['flat_confirmed_at'] = now
        self.state['halt'] = None
        self.armed_at = now
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.save()
        labels = ', '.join(sorted(symbols))
        self.log(f'[CRASH ENTRY HALT CLEARED] {labels}; '
                 'live flat+no orders+client-ID resolution; auto setting unchanged')
        self.alert(f'✅ BELİRSİZ ALIŞ KİLİDİ TEMİZLENDİ | {labels}\n'
                   'Binance pozisyon ve açık emir sorguları doğrulandı. '
                   'Eski sinyaller tekrar gönderilmeyecek. '
                   'Otomatik işlem açık/kapalı ayarı korunuyor.')
        return True

    def deliver_pending(self):
        if not self.state['outbox']:
            return
        item = self.state['outbox'][0]
        if self.clock() < item['retry_at']:
            return
        item['retry_at'] = self.clock()+300
        self.save()
        receipt = self.deliver(item['text'])
        if receipt.get('ok'):
            self.state['outbox'].pop(0)
            self.save()
        # An unacknowledged alert blocks new entries until delivery succeeds.

    def _deliver_skip_notice(self, message):
        """Informational only: never persist as an entry-blocking safety alert."""
        try:
            receipt = self.deliver(message)
            if not receipt.get('ok'):
                self.log('[CRASH ENTRY SKIP NOTICE] Telegram unavailable; '
                         'safety outbox unchanged')
        except Exception as error:
            self.log(f'[CRASH ENTRY SKIP NOTICE ERR] {type(error).__name__}')

    def entry_skip(self, symbol, reason):
        """Always log rejection; send optional deduplicated, nonblocking notice."""
        self.log(f'[CRASH ENTRY SKIP] {symbol}; reason={reason}; '
                 f'daily={self.daily_count()}/{self.max_daily}; '
                 f'halt={bool(self.state.get("halt"))}; outbox={len(self.state["outbox"])}')
        notices = {
            'DAILY_LIMIT': 'Günlük giriş sınırı doldu.',
            'EXISTING_ORDER': 'Aynı coin için mevcut giriş/pozisyon kaydı var.',
            'SIGNAL_EXPIRED': 'Girişin 120 saniyelik geçerlilik süresi doldu.',
            'SIGNAL_EXPIRED_AFTER_PREFLIGHT': 'Ön kontrol sırasında sinyalin süresi doldu.',
            'QUEUE_FULL': 'Otomatik giriş kuyruğu dolu.',
            'SYMBOL_QUARANTINED': 'Coin karantinada; kalan emirler kontrol ediliyor.',
        }
        # A safety halt already has its own durable warning. Do NOT put
        # optional skip notices into the blocking outbox: a failed Telegram
        # delivery must not block tomorrow's otherwise valid new entries.
        if (reason in notices and not self.state['outbox']
                and self.clock() >= self.skip_notice_after.get(reason, 0)):
            self.skip_notice_after[reason] = self.clock() + LOCK_AUDIT_SECONDS
            message = (f'⏸️ OTOMATİK ALIŞ ENGELLENDİ | {symbol}\n'
                       f'Neden: {notices[reason]} '
                       'Mevcut emirler ve TP koruması değiştirilmedi.')
            threading.Thread(target=self._deliver_skip_notice,
                             args=(message,), name='crash-entry-skip-notice',
                             daemon=True).start()

    def submit(self, record, observed_at):
        symbol = record.get('symbol', 'UNKNOWN')
        if not self.enabled.is_set():
            self.entry_skip(symbol, 'AUTO_OFF')
            return
        if observed_at < self.armed_at:
            self.entry_skip(symbol, 'SIGNAL_BEFORE_ARM')
            return
        try:
            self.queue.put_nowait((dict(record), observed_at))
        except queue.Full:
            self.entry_skip(symbol, 'QUEUE_FULL')

    def enter(self, record, observed_at):
        from crash_policy import supported_symbol
        from crash_failure_shadow import entry_block_reason, position_fraction, SIZING_VERSION
        now = self.clock()
        sid, symbol = record.get('signal_id'), record.get('symbol')
        reason = (
            'AUTO_OFF' if not self.enabled.is_set() else
            'SIGNAL_BEFORE_ARM' if observed_at < self.armed_at else
            'INVALID_SIGNAL' if not sid or not supported_symbol(symbol) else
            'HALTED' if self.state['halt'] else
            'SYMBOL_QUARANTINED' if symbol in self.state['symbol_quarantine'] else
            'ALERT_PENDING' if self.state['outbox'] else
            'SIGNAL_EXPIRED' if not 0 <= now-observed_at <= 120 or observed_at < self.started else
            'V2_NOT_APPROVED' if record.get('v2_status') != 'STRONG_V2'
                or record.get('v2_sent') is not True else
            'TRACKING_INACTIVE' if record.get('tracking_status') not in ('WAIT_ENTRY', 'OPEN') else
            'DUPLICATE_SIGNAL' if sid in self.state['orders'] else None)
        if reason:
            self.entry_skip(symbol, reason)
            return
        # Three concurrent risk flags or incomplete assessment fail closed.
        # Zero/one flags use the configured notional; two flags use half.
        # This is an entry-only decision. Existing orders/TPs are untouched.
        risk_block = entry_block_reason(record)
        fraction = position_fraction(record)
        if risk_block or fraction is None or fraction <= 0:
            self.log(f'[CRASH RISK ENTRY SKIP] {symbol}: {risk_block or "INVALID_RISK_FRACTION"}')
            self.entry_skip(symbol, 'RISK_BLOCKED')
            return
        configured_usdt = self.position_usdt
        target_usdt = configured_usdt * D(str(fraction))
        flags = record['failure_shadow']['flags']
        risk_count = sum(flags.values())
        if self.daily_count() >= self.max_daily:
            self.entry_skip(symbol, 'DAILY_LIMIT')
            return
        if any(o['phase'] not in TERMINAL_PHASES and o['symbol'] == symbol for o in self.state['orders'].values()):
            self.entry_skip(symbol, 'EXISTING_ORDER')
            return
        try:
            p = plan(record, self.broker.info(symbol), target_usdt)
            position_side = self.broker.preflight(symbol)
            p['position_side'] = position_side if position_side in ('BOTH', 'LONG') else 'BOTH'
        except Exception as error:
            if (isinstance(error, BrokerError) and error.not_sent
                    and error.deferred_reason == 'IP_WEIGHT_SOFT_CEILING'):
                retry = self.preflight_retries.get(sid, {})
                attempts = int(retry.get('attempts', 0))
                retry_at = self.clock() + max(1.0, float(error.deferred_seconds or 1)) + 1
                # Keep the same original real trade-cross timestamp; do not
                # manufacture a new signal or exceed the existing 120s window.
                if (attempts < 2 and retry_at < observed_at + 120
                        and len(self.preflight_retries) < 25):
                    self.preflight_retries[sid] = dict(
                        record=dict(record), observed_at=observed_at,
                        retry_at=retry_at, attempts=attempts+1)
                    self.log(f'[CRASH NIGHT PREFLIGHT DEFERRED] {symbol}; '
                             f'local_IP_weight; attempt={attempts+1}/2; '
                             f'retry_after_s={int(retry_at-self.clock())}; '
                             'no BUY sent; TP checks remain prioritized')
                    return
            self.preflight_retries.pop(sid, None)
            reason = ('LOCAL_IP_BUDGET' if isinstance(error, BrokerError)
                      and error.not_sent and error.deferred_reason == 'IP_WEIGHT_SOFT_CEILING'
                      else (str(error)[:240] if isinstance(error, (ValueError, BrokerError))
                            else 'CHECK_FAILED'))
            self.alert(f'⚠️ OTOMATİK GİRİŞ YAPILMADI | {symbol}\n'
                       f'TP/giriş ön kontrolü başarısız: {type(error).__name__}: {reason}')
            return
        # Network preflight can take seconds. Recheck window and freshness.
        now = self.clock()
        if (not self.enabled.is_set() or not 0 <= now-observed_at <= 120
                or entry_block_reason(record) is not None
                or position_fraction(record) != fraction):
            self.entry_skip(symbol, 'SIGNAL_EXPIRED_AFTER_PREFLIGHT'
                            if not 0 <= now-observed_at <= 120 else
                            'PRECHECK_CHANGED')
            return
        cid = 'cbn'+hashlib.sha256(sid.encode()).hexdigest()[:24]
        order = dict(symbol=symbol, day=day(now), phase='ENTRY_UNKNOWN',
                     entry_kind='GTC', submitted_at=now, entry_id=cid+'e',
                     tp_id=cid+'t', risk_sizing_version=SIZING_VERSION,
                     risk_flags={k: flags[k] for k in ("weak_volume", "falling_rsi_ema", "repeat_lower_entry")},
                     risk_flag_count=risk_count, risk_position_fraction=fraction,
                     risk_configured_usdt=str(configured_usdt),
                     risk_target_usdt=str(target_usdt),
                     risk_effective_notional_usdt=str(D(p['quantity'])*D(p['price'])), **p)
        self.state['orders'][sid] = order
        self.save()  # Persist before POST; unknown responses are never blindly retried.
        self.log(f'[CRASH RISK SIZING] {symbol}; flags={risk_count}/3; '
                 f'fraction={fraction}; base_usdt={configured_usdt}; '
                 f'target_usdt={target_usdt}; quantity={p["quantity"]}; '
                 f'estimated_notional_usdt={order["risk_effective_notional_usdt"]}')
        if not self.enabled.is_set():
            order['phase'] = 'NO_FILL'
            self.save()
            return
        try:
            self.broker.buy(symbol, p, order['entry_id'])
        except BrokerError as error:
            if error.not_sent:
                # No POST was sent: NEVER query a nonexistent client order,
                # invent a fill, or permanently halt automatic trading.
                order['phase'] = 'NO_FILL'
                order['entry_never_sent'] = True
                order['entry_submit_error'] = error_details(error)
                self.save()
                self.log(f'[CRASH NIGHT ENTRY NOT SENT] {symbol}; '
                         f'reason={error.deferred_reason or "EXCHANGE_COOLDOWN"}; '
                         'BUY HTTP not attempted; old signal not retried')
                self.alert(f'⏳ OTOMATİK ALIŞ GÖNDERİLMEDİ | {symbol}\n'
                           'Binance REST koruması alış isteğini göndermeden erteledi. '
                           'Bu sinyalde alım yapılmadı ve günlük giriş hakkı tüketilmedi. '
                           'Yeni uygun sinyaller izlenmeye devam ediyor.')
                return
            # A POST may have reached Binance; never issue a second BUY.
            self.log(f'[CRASH NIGHT BUY RESPONSE] {symbol}; '
                     f'code={error.code}; uncertain={error.uncertain}; '
                     'original client ID will be reconciled')
        except Exception as error:
            self.log(f'[CRASH NIGHT BUY RESPONSE] {symbol}; '
                     f'type={type(error).__name__}; original client ID will be reconciled')
        self.reconcile(order)

    def _confirm_external_close(self, order):
        """Recognize manual close as a terminal trade, never a new quarantine."""
        symbol = order.get('symbol')
        if (order.get('phase') not in ('TP_UNKNOWN', 'PROTECTED')
                or not symbol or
                any(o is not order and o.get('symbol') == symbol and
                    o.get('phase') == 'ENTRY_UNKNOWN'
                    for o in self.state['orders'].values())):
            return False
        remaining = self._finish_flat_symbol(symbol)
        if remaining is None:
            return False
        now = self.clock()
        order['phase'] = 'CLOSED_EXTERNALLY'
        order['flat_confirmed_at'] = now
        order['close_verification'] = (
            'LIVE_POSITION_ZERO_MANUAL_ORDERS_KEPT'
            if remaining else 'LIVE_POSITION_ZERO_NO_OPEN_ORDERS')
        order['external_close_reason'] = 'TP_NOT_ACTIVE_POSITION_FLAT'
        self.state['symbol_quarantine'].pop(symbol, None)
        self.save()
        self.log(f'[CRASH EXTERNAL CLOSE CONFIRMED] {symbol}; '
                 f'flat, remaining_manual_orders={remaining}; '
                 'no quarantine; TP fill NOT assumed')
        message = (f'ℹ️ POZİSYON ELLE KAPATILDI | {symbol}\n'
                   f'Bot kaydı kapatıldı. Manuel emir: {remaining}; '
                   'elle verilmiş emirler silinmedi. Bu coin için bir sonraki '
                   'alış, Binance açık emir/pozisyon kontrolünden geçecek.')
        threading.Thread(target=self._deliver_skip_notice, args=(message,),
                         name='crash-external-close-notice', daemon=True).start()
        return True

    def review_protected_manual_closes(self):
        """Find manual exits even when the old bot TP still reports NEW.

        One account snapshot per five-minute sweep. Only symbols that are
        already flat proceed to signed TP-ID and order/position verification.
        An unconfirmed API result never closes a trade or cancels an order.
        """
        active = [o for o in self.state['orders'].values()
                  if o.get('phase') in ('TP_UNKNOWN', 'PROTECTED')
                  and o.get('symbol') and o.get('tp_id')]
        if not active or not callable(getattr(self.broker, 'positions', None)):
            return
        try:
            positions = self.broker.positions()
            if not isinstance(positions, list):
                raise ValueError('INVALID_POSITION_RESPONSE')
        except Exception as error:
            code = error.code if isinstance(error, BrokerError) else type(error).__name__
            self.log(f'[CRASH MANUAL CLOSE SCAN WAIT] code={code}')
            return
        for order in active:
            symbol = order['symbol']
            try:
                if not self._flat_from_snapshot(positions, {symbol}):
                    continue
                tp = None
                try:
                    tp = self.broker.order(symbol, order['tp_id'])
                except BrokerError as error:
                    if error.code != '-2013' or error.uncertain:
                        raise
                if tp is not None:
                    if (not isinstance(tp, dict)
                            or tp.get('symbol') != symbol
                            or tp.get('clientOrderId') != order['tp_id']
                            or tp.get('side') != 'SELL'
                            or tp.get('status') not in (
                                'NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED',
                                'EXPIRED', 'EXPIRED_IN_MATCH', 'REJECTED')):
                        self.log(f'[CRASH MANUAL CLOSE SCAN WAIT] {symbol}; '
                                 'TP_IDENTITY_OR_STATUS_INVALID')
                        continue
                    if tp['status'] == 'FILLED':
                        # Keep an actual TP execution classified as DONE, never
                        # claim a manual exit just because exposure is zero.
                        self.reconcile(order, check_expiry=False, ws_order=tp)
                        continue
                self._confirm_external_close(order)
            except Exception as error:
                code = error.code if isinstance(error, BrokerError) else type(error).__name__
                self.log(f'[CRASH MANUAL CLOSE SCAN WAIT] {symbol}; code={code}')

    def reconcile(self, order, check_expiry=True, ws_order=None):
        symbol = order['symbol']
        try:
            if order['phase'] == 'ENTRY_UNKNOWN':
                entry = ws_order if ws_order and ws_order['clientOrderId'] == order['entry_id'] else self.broker.order(symbol, order['entry_id'])
                status = entry.get('status')
                qty = D(entry['executedQty'])
                if order.get('entry_kind') == 'GTC' and status in ('NEW', 'PARTIALLY_FILLED'):
                    created = D(str(entry.get('time', 0))) / 1000 or D(str(order.get('submitted_at', self.clock())))
                    expired = check_expiry and D(str(self.clock())) - created >= ENTRY_TTL_SECONDS
                    if qty == 0 and not expired:
                        if not order.get('pending_notified'):
                            order['pending_notified'] = True
                            self.alert(f'⏳ ALIŞ LİMİTİ BEKLİYOR | {symbol}\nGTC limit: {order["price"]} | Miktar: {order["quantity"]}\nFiyat limite veya altına gelince gerçekleşebilir. Henüz dolum yok.')
                        return
                    # Stop the unfilled remainder before protecting the final filled quantity.
                    # Query after cancellation even when the DELETE response is uncertain.
                    order['expiry_cancel_requested'] = bool(expired)
                    order['partial_cancel_requested'] = qty > 0
                    self.save()
                    try:
                        self.broker.cancel_entry(symbol, order['entry_id'])
                    except Exception:
                        pass
                    entry = self.broker.order(symbol, order['entry_id'])
                    status = entry.get('status')
                    qty = D(entry['executedQty'])
                if status not in ('FILLED', 'EXPIRED', 'CANCELED', 'EXPIRED_IN_MATCH', 'REJECTED'):
                    raise ValueError('ENTRY_RESULT_NOT_FINAL')
                if qty == 0:
                    order['phase'] = 'NO_FILL'
                    self.save()
                    reason = '8 saat doldu; bekleyen alış iptal edildi.' if order.get('expiry_cancel_requested') else 'Giriş emri dolmadan kapandı.'
                    self.alert(f'ℹ️ OTOMATİK İŞLEM | {symbol}\n{reason} Açık giriş emri bırakılmadı. Günlük hak kullanıldı.')
                    return
                if qty < 0 or qty > D(order['quantity']):
                    raise ValueError('INVALID_FILLED_QUANTITY')
                order['filled_qty'] = str(qty)
                order['phase'] = 'TP_UNKNOWN'
                self.save()  # On restart query this client ID; never duplicate TP.
                try:
                    kwargs = {'position_side': 'LONG'} if order.get('position_side') == 'LONG' else {}
                    self.broker.tp(symbol, order['filled_qty'], order['tp'], order['tp_id'], **kwargs)
                except Exception as error:
                    order['tp_submit_error'] = error_details(error)
                    self.save()
            if order['phase'] in ('TP_UNKNOWN', 'PROTECTED'):
                tp = ws_order if ws_order and ws_order['clientOrderId'] == order['tp_id'] else self.broker.order(symbol, order['tp_id'])
                expected_side = order.get('position_side', 'BOTH')
                if (tp.get('side') != 'SELL' or tp.get('positionSide', 'BOTH') != expected_side
                        or (expected_side == 'BOTH' and tp.get('reduceOnly') not in (True, 'true'))
                        or D(tp['origQty']) != D(order['filled_qty']) or D(tp['price']) != D(order['tp'])):
                    raise ValueError('TP_PROPERTIES_NOT_CONFIRMED')
                if tp.get('status') == 'FILLED':
                    order['phase'] = 'DONE'
                    self.save()
                    self.alert(f'✅ OTOMATİK TP DOLDU | {symbol}\nTP: {order["tp"]}')
                elif tp.get('status') in ('NEW', 'PARTIALLY_FILLED'):
                    first = order['phase'] != 'PROTECTED'
                    order['phase'] = 'PROTECTED'
                    self.save()
                    if first:
                        self.alert(f'✅ OTOMATİK GİRİŞ + TP DOĞRULANDI | {symbol}\nMiktar: {order["filled_qty"]} | Giriş limit: {order["price"]}\nTP: {order["tp"]} | 10x | SL yok')
                else:
                    raise ValueError('TP_NOT_ACTIVE')
        except Exception as error:
            if isinstance(error, BrokerError) and error.code == '-1003':
                self.log(f'[CRASH API COOLDOWN] {symbol}; sorgulama ertelendi')
                return
            # Missing/cancelled TP after an operator closes a position is a
            # normal terminal event, IF the affected symbol is proven flat
            # and no normal/algo orders remain. Do not relax unresolved BUYs.
            missing_tp = (
                (isinstance(error, BrokerError) and error.code == '-2013'
                 and not error.uncertain)
                or (isinstance(error, ValueError)
                    and str(error) == 'TP_NOT_ACTIVE'))
            if missing_tp and order['phase'] in ('TP_UNKNOWN', 'PROTECTED'):
                if self._confirm_external_close(order):
                    return
            reason = ('GİRİŞ SONUCU BELİRSİZ' if order['phase'] == 'ENTRY_UNKNOWN' else 'TP DOĞRULANAMADI')
            if not order.get('warned'):
                order['warned'] = True
                self.halt(reason+'\n'+order.get('tp_submit_error', error_details(error)), symbol)
            code = error.code if isinstance(error, BrokerError) else type(error).__name__
            self.log(f'[CRASH NIGHT RECONCILE] {symbol} {order["phase"]}; '
                     f'code={code}; deferred={getattr(error, "deferred_reason", None)}')

    def expire_untracked_entries(self):
        # After ephemeral storage loss, identify only this bot's BUY entry IDs.
        fetch = getattr(self.broker, 'open_entry_orders', None)
        if not callable(fetch):
            return
        tracked = {o['entry_id'] for o in self.state['orders'].values() if 'entry_id' in o}
        for entry in fetch():
            cid = entry.get('clientOrderId', '')
            if (cid in tracked or not re.fullmatch(r'cbn[0-9a-f]{24}e', cid)
                    or entry.get('side') != 'BUY' or entry.get('type') != 'LIMIT'
                    or entry.get('positionSide', 'BOTH') not in ('BOTH', 'LONG')):
                continue
            created = D(str(entry.get('time', 0))) / 1000
            if created <= 0 or D(str(self.clock())) - created < ENTRY_TTL_SECONDS:
                continue
            symbol = entry['symbol']
            try:
                self.broker.cancel_entry(symbol, cid)
            except Exception:
                pass
            result = self.broker.order(symbol, cid)
            if result.get('status') not in ('FILLED', 'CANCELED', 'EXPIRED', 'EXPIRED_IN_MATCH', 'REJECTED'):
                raise ValueError('ENTRY_EXPIRY_CANCEL_UNCONFIRMED')
            self.alert(f'ℹ️ 8 SAATLİK ALIŞ TAKİBİ | {symbol}\nBot giriş emri artık açık değil. Durum: {result.get("status")} | Dolan miktar: {result.get("executedQty")}\nAçık pozisyon ve TP emirleri iptal edilmedi.')

    def on_account_event(self, event):
        # Socket callbacks never mutate trading state or place orders.
        if event.get('e') == 'ACCOUNT_STREAM_RESYNC':
            self.account_resync.set()
            return
        try:
            self.account_events.put_nowait(event)
        except queue.Full:
            if not self.account_resync.is_set():
                self.log('[CRASH ACCOUNT WS OVERFLOW] REST mutabakati gerekli')
            self.account_resync.set()

    def process_account_events(self):
        from crash_user_stream import order_update
        now = self.clock()
        if self.account_resync.is_set() and now >= self.account_resync_after:
            self.account_resync.clear()
            self.account_resync_after = now+60
            self.next_poll = self.next_manual_scan = 0
            self.protected_poll.clear()
        for _ in range(200):
            if self.clock() < getattr(self.broker, 'cooldown_until', 0):
                break
            try:
                event = self.account_events.get_nowait()
            except queue.Empty:
                break
            try:
                stamp = int(event['E'])
                # Stale/replayed events cause authoritative reconciliation, not a late TP.
                if not -5 <= now-stamp/1000 <= 30:
                    self.account_resync.set()
                    continue
                if event.get('e') == 'ORDER_TRADE_UPDATE':
                    update = order_update(event)
                    key = ('order', update['symbol'], update['clientOrderId'])
                    version = (stamp, Decimal(update['executedQty']),
                               update['status'] in ('FILLED','CANCELED','EXPIRED','EXPIRED_IN_MATCH','REJECTED'))
                    if version <= self.account_versions.get(key, (-1, Decimal(0), False)):
                        continue
                    self.account_versions[key] = version
                    matched = False
                    for order in list(self.state['orders'].values()):
                        if order['symbol'] != update['symbol']:
                            continue
                        is_entry = update['clientOrderId'] == order['entry_id']
                        is_tp = update['clientOrderId'] == order['tp_id']
                        if not (is_entry or is_tp):
                            continue
                        matched = True
                        if (is_entry and order['phase'] != 'ENTRY_UNKNOWN') or (is_tp and order['phase'] not in ('TP_UNKNOWN','PROTECTED')):
                            break
                        expected = 'BUY' if is_entry else 'SELL'
                        if (update['side'] != expected or update['type'] != 'LIMIT'
                                or update['positionSide'] != order.get('position_side','BOTH')
                                or (is_entry and (Decimal(update['origQty']) != Decimal(order['quantity'])
                                    or Decimal(update['price']) != Decimal(order['price'])))):
                            self.account_resync.set()
                            break
                        self.log(f"[CRASH ACCOUNT ORDER] {update['symbol']}; status={update['status']}; filled={update['executedQty']}")
                        self.reconcile(order, check_expiry=False, ws_order=update)
                        self.protected_poll[order['entry_id']] = now+TP_POLL_SECONDS
                        break
                    if not matched:
                        self.ws_manual_symbols.add(update['symbol'])
                elif event.get('e') == 'ACCOUNT_UPDATE':
                    # This is a delta: omitted symbols are NOT closed positions.
                    tracked = {o['symbol'] for o in self.state['orders'].values() if o['phase'] not in TERMINAL_PHASES}
                    for p in event.get('a', {}).get('P', []):
                        key = ('position', p['s'], p.get('ps','BOTH'))
                        if stamp <= self.account_versions.get(key, -1):
                            continue
                        self.account_versions[key] = stamp
                        if p.get('ps','BOTH') in ('BOTH','LONG') and p['s'] not in tracked:
                            self.ws_manual_symbols.add(p['s'])
                        self.log(f"[CRASH ACCOUNT POSITION] {p['s']}; side={p.get('ps','BOTH')}; degisiklik alindi")
                if len(self.account_versions) > 10000:
                    self.account_versions.clear()
                    self.account_resync.set()
            except Exception as error:
                self.account_resync.set()
                self.log(f'[CRASH ACCOUNT EVENT ERR] {type(error).__name__}')
        if (self.manual_tp_enabled and self.ws_manual_symbols and now >= self.ws_manual_after
                and self.clock() >= getattr(self.broker, 'cooldown_until', 0)):
            if self.manual_guard is None:
                from crash_manual_tp import ManualTPGuard
                self.manual_guard = ManualTPGuard(self)
            symbols = set(self.ws_manual_symbols)
            self.ws_manual_symbols.clear()
            self.ws_manual_after = now+30
            # REST safety check is scoped to changed symbols; never treat a delta as a snapshot.
            self.manual_guard.scan(symbols=symbols)

    def step(self):
        while True:
            try:
                action, done, result = self.controls.get_nowait()
            except queue.Empty:
                break
            try:
                result['message'] = self.apply_control(action)
            except Exception:
                self.enabled.clear()
                result['message'] = '⚠️ Komut kaydedilemedi; yeni girişler kapalı.'
                raise
            finally:
                done.set()
        # A rate-limit pause suppresses exchange calls, not Telegram delivery/controls.
        if self.clock() < getattr(self.broker, 'cooldown_until', 0):
            self.deliver_pending()
            return
        self.process_account_events()
        if self.state.get('halt') and self.clock() >= self.next_halt_recheck:
            self.next_halt_recheck = self.clock() + MANUAL_TP_POLL_SECONDS
            self.recover_mixed_halt()
            if self.state.get('halt'):
                self.recover_closed_tp_halt()
            if self.state.get('halt'):
                self.recover_unknown_entry_halt()
        if (self.state['symbol_quarantine']
                and self.clock() >= self.next_quarantine_scan):
            self.next_quarantine_scan = self.clock() + MANUAL_TP_POLL_SECONDS
            self.review_quarantined_symbols()
        if self.clock() >= self.next_lock_audit:
            self.next_lock_audit = self.clock() + LOCK_AUDIT_SECONDS
            entry_unknown = sum(o.get('phase') == 'ENTRY_UNKNOWN'
                                for o in self.state['orders'].values())
            tp_unknown = sum(o.get('phase') == 'TP_UNKNOWN'
                             for o in self.state['orders'].values())
            if self.state.get('halt') or entry_unknown or tp_unknown or self.state['symbol_quarantine']:
                self.log(f'[CRASH LOCK AUDIT 30M] halt={self.state.get("halt") or "none"}; '
                         f'entry_unknown={entry_unknown}; tp_unknown={tp_unknown}; '
                         f'auto_enabled={self.enabled.is_set()}; '
                         f'alert_pending={len(self.state["outbox"])}; '
                         f'quarantine={",".join(sorted(self.state["symbol_quarantine"])) or "none"}')
        if self.clock() >= self.next_manual_close_scan:
            self.next_manual_close_scan = self.clock() + MANUAL_TP_POLL_SECONDS
            self.review_protected_manual_closes()
        # Manage protective orders even outside the entry window or when halted.
        if self.clock() >= self.next_poll or self.clock() >= self.next_expiry_scan:
            self.next_poll = self.clock()+ORDER_POLL_SECONDS
            for order in list(self.state['orders'].values()):
                if order['phase'] == 'PROTECTED':
                    key = order['entry_id']
                    if self.clock() < self.protected_poll.get(key, 0):
                        continue
                    self.protected_poll[key] = self.clock()+TP_POLL_SECONDS
                if order['phase'] in ('ENTRY_UNKNOWN', 'TP_UNKNOWN', 'PROTECTED'):
                    self.reconcile(order, check_expiry=self.clock() >= self.next_expiry_scan)
            if self.manual_tp_enabled and self.clock() >= self.next_manual_scan:
                self.next_manual_scan = self.clock()+MANUAL_TP_POLL_SECONDS
                if self.manual_guard is None:
                    from crash_manual_tp import ManualTPGuard
                    self.manual_guard = ManualTPGuard(self)
                self.manual_guard.scan()
        if self.clock() >= self.next_expiry_scan:
            self.next_expiry_scan = self.clock()+ENTRY_EXPIRY_SCAN_SECONDS
            try:
                self.expire_untracked_entries()
            except BrokerError as error:
                if error.code != '-1003':
                    raise
                self.log('[CRASH API COOLDOWN] eski giriş taraması ertelendi')
        self.deliver_pending()
        # Never sleep inside the executor: maintain account WS, TP and
        # protective polls while waiting for a safe next-minute preflight.
        now = self.clock()
        due = [(sid, retry) for sid, retry in self.preflight_retries.items()
               if now >= retry['retry_at']]
        if due:
            sid, retry = min(due, key=lambda item: item[1]['retry_at'])
            self.preflight_retries.pop(sid, None)
            if (now - retry['observed_at'] >= 120
                    or not self.enabled.is_set() or self.state.get('halt')):
                self.log(f'[CRASH NIGHT PREFLIGHT EXPIRED] {retry["record"]["symbol"]}; '
                         'recheck window closed or auto unavailable; no BUY sent')
            else:
                # Reuse the original observed_at; enter() rechecks all live
                # position/orders, daily limits, risk and paused/halt flags.
                current = dict(retry, retry_at=now)
                self.preflight_retries[sid] = current
                self.enter(retry['record'], retry['observed_at'])
                # Keep a newly rescheduled retry; clean up only if enter()
                # did not replace the placeholder with a future due time.
                if self.preflight_retries.get(sid) is current:
                    self.preflight_retries.pop(sid, None)
            return
        try:
            record, observed_at = self.queue.get_nowait()
        except queue.Empty:
            return
        self.enter(record, observed_at)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        if isinstance(self.broker, BinanceBroker):
            try:
                import websocket
                from crash_user_stream import AccountStream
                self.account_stream = AccountStream(self.broker, self.on_account_event, self.log, websocket,
                    blocked=getattr(self, 'public_rest_blocked', lambda:False))
                self.account_stream.start()
            except Exception as error:
                self.log(f'[CRASH ACCOUNT WS START ERR] {type(error).__name__}; REST yedek aktif')
        def run():
            while True:
                try:
                    self.step()
                except Exception as error:
                    # Persistence/transport bugs must never allow additional entries.
                    already_halted = bool(self.state['halt'])
                    self.state['halt'] = 'EXECUTOR_ERROR'
                    if not already_halted:
                        self.state['outbox'].append({'text': '⚠️ OTOMATİK İŞLEM HATASI: Yeni girişler durdu. Binance pozisyonlarını ve TP emirlerini kontrol edin.', 'retry_at': 0})
                    self.log(f'[CRASH NIGHT ERR] {type(error).__name__}')
                    try:
                        self.save()
                    except Exception:
                        pass
                time.sleep(1)
        self.thread = threading.Thread(target=run, name='crash-night', daemon=True)
        self.thread.start()
