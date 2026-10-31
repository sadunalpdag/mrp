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
ENTRY_EXPIRY_SCAN_SECONDS = 15 * 60


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
    if not budget.is_finite() or not 0 < budget <= 100:
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
    def __init__(self, code, uncertain=False):
        raw = str(code).removeprefix('BINANCE_')
        self.code = raw if raw.lstrip('-').isdigit() else 'UNKNOWN'
        self.uncertain = uncertain or self.code == 'UNKNOWN'
        super().__init__('BINANCE_'+self.code)


def error_details(error):
    if isinstance(error, BrokerError):
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

    def request(self, method, path, params=None, signed=True):
        import requests
        import hmac
        from urllib.parse import urlencode
        if time.time() < self.cooldown_until:
            raise BrokerError('-1003')
        params = dict(params or {})
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
        result = self.request('GET', '/fapi/v3/positionRisk')
        if not isinstance(result, list):
            raise ValueError('INVALID_POSITION_RESPONSE')
        return result

    def closing_orders(self, symbol):
        normal = self.request('GET', '/fapi/v1/openOrders', {'symbol': symbol})
        algo = self.request('GET', '/fapi/v1/openAlgoOrders', {'symbol': symbol})
        if not isinstance(normal, list) or not isinstance(algo, list):
            raise ValueError('INVALID_ORDER_RESPONSE')
        return normal + algo

    def best_bid(self, symbol):
        result = self.request('GET', '/fapi/v1/ticker/bookTicker', {'symbol': symbol}, signed=False)
        return D(result['bidPrice'])

    def repair_tp(self, symbol, qty, price, client_id, position_side="BOTH"):
        # Maker-only: never deliberately sell immediately after a missed target.
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='SELL', type='LIMIT',
            timeInForce='GTX', quantity=qty, price=price, newClientOrderId=client_id,
            **self.close_params(position_side)))

    def info(self, symbol):
        data = self.request('GET', '/fapi/v1/exchangeInfo', signed=False)
        return next(s for s in data['symbols'] if s['symbol'] == symbol)

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

    def buy(self, symbol, p, client_id):
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='BUY', type='LIMIT',
            timeInForce='GTC', quantity=p['quantity'], price=p['price'], newClientOrderId=client_id,
            newOrderRespType='RESULT', positionSide=p.get('position_side', 'BOTH')))

    def tp(self, symbol, qty, price, client_id, position_side="BOTH"):
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='SELL', type='LIMIT',
            timeInForce='GTC', quantity=qty, price=price, newClientOrderId=client_id,
            **self.close_params(position_side)))


class NightExecutor:
    def __init__(self, directory, broker, deliver, log, clock=time.time, enabled=False, max_daily=5, manual_tp=False):
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
            self.state['paused'] = not enabled
            self.save()
            with open(marker, 'a') as handle:
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.guard.close()
            raise
        settings = self.state.get('trade_settings', {})
        self.max_daily = int(settings.get('max_daily', max_daily))
        self.max_symbols = int(settings.get('max_symbols', 5))
        self.broker.max_symbols = self.max_symbols
        self.position_usdt = D(str(settings.get('position_usdt', '50')))
        if not 1 <= self.max_symbols <= 50 or not 1 <= self.max_daily <= 15 or not self.position_usdt.is_finite() or not 1 <= self.position_usdt <= 100:
            self.guard.close()
            raise ValueError('INVALID_TRADE_SETTINGS')
        self.manual_tp_enabled = manual_tp
        self.manual_guard = None
        self.started = clock()
        self.queue = queue.Queue(maxsize=100)
        self.thread = None
        self.next_poll = 0
        self.next_expiry_scan = 0
        self.next_manual_scan = 0
        self.protected_poll = {}
        self.enabled = threading.Event()
        self.armed_at = clock()
        if enabled:
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
        raw = sum(o['day'] == today for o in list(self.state['orders'].values()))
        adjustment = self.state.get('daily_counter_override', {})
        if adjustment.get('day') == today:
            return max(0, adjustment['count'] + raw - adjustment['raw_at_set'])
        return raw

    def summary(self):
        count = self.daily_count()
        stream = self.account_stream.summary() if self.account_stream else 'Hesap WS: baslatilmadi / REST yedek'
        return f"{'DURDU: '+self.state['halt'] if self.state['halt'] else ('AÇIK' if self.enabled.is_set() else 'KAPALI')} | bugün {count}/{self.max_daily} | sembol sınırı {self.max_symbols} | pozisyon {self.position_usdt} USDT | 10x | 24 saat | {stream}"

    def apply_control(self, action):
        if action.startswith(('limit:', 'amount:', 'symbols:')):
            kind, raw = action.split(':', 1)
            try:
                value = D(raw)
                ceiling = {'amount':100, 'symbols':50, 'limit':15}[kind]
                if not value.is_finite() or not 1 <= value <= ceiling or (kind != 'amount' and value != value.to_integral_value()):
                    raise ValueError('range')
            except (ValueError, ArithmeticError):
                return '⚠️ Kullanım: /sembol 1–50, /limit 1–15 veya /tutar 1–100 (USDT toplam pozisyon, teminat değil).'
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
                raw_at_set=sum(o['day'] == today for o in self.state['orders'].values()))
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

    def submit(self, record, observed_at):
        if not self.enabled.is_set() or observed_at < self.armed_at:
            return
        try:
            self.queue.put_nowait((dict(record), observed_at))
        except queue.Full:
            self.log('[CRASH NIGHT SKIP] QUEUE_FULL')

    def enter(self, record, observed_at):
        from crash_policy import supported_symbol
        now = self.clock()
        sid, symbol = record.get('signal_id'), record.get('symbol')
        if (not self.enabled.is_set() or observed_at < self.armed_at
                or not sid or not supported_symbol(symbol) or self.state['halt'] or self.state['outbox']
                or not 0 <= now-observed_at <= 120 or observed_at < self.started
                or record.get('v2_status') != 'STRONG_V2' or record.get('v2_sent') is not True
                or record.get('tracking_status') not in ('WAIT_ENTRY', 'OPEN') or sid in self.state['orders']):
            return
        if self.daily_count() >= self.max_daily:
            return
        if any(o['phase'] not in ('DONE', 'NO_FILL') and o['symbol'] == symbol for o in self.state['orders'].values()):
            return
        try:
            p = plan(record, self.broker.info(symbol), self.position_usdt)
            position_side = self.broker.preflight(symbol)
            p['position_side'] = position_side if position_side in ('BOTH', 'LONG') else 'BOTH'
        except Exception as error:
            self.alert(f'⚠️ OTOMATİK GİRİŞ YAPILMADI | {symbol}\nTP/giriş ön kontrolü başarısız: {type(error).__name__}: {str(error)[:240] if isinstance(error, (ValueError, BrokerError)) else "CHECK_FAILED"}')
            return
        # Network preflight can take seconds. Recheck window and freshness.
        now = self.clock()
        if not self.enabled.is_set() or not 0 <= now-observed_at <= 120:
            return
        cid = 'cbn'+hashlib.sha256(sid.encode()).hexdigest()[:24]
        order = dict(symbol=symbol, day=day(now), phase='ENTRY_UNKNOWN', entry_kind='GTC', submitted_at=now, entry_id=cid+'e', tp_id=cid+'t', **p)
        self.state['orders'][sid] = order
        self.save()  # Persist before POST; unknown responses are never blindly retried.
        if not self.enabled.is_set():
            order['phase'] = 'NO_FILL'
            self.save()
            return
        try:
            self.broker.buy(symbol, p, order['entry_id'])
        except Exception:
            pass
        self.reconcile(order)

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
            reason = ('GİRİŞ SONUCU BELİRSİZ' if order['phase'] == 'ENTRY_UNKNOWN' else 'TP DOĞRULANAMADI')
            if not order.get('warned'):
                order['warned'] = True
                self.halt(reason+'\n'+order.get('tp_submit_error', error_details(error)), symbol)
            self.log(f'[CRASH NIGHT RECONCILE] {symbol} {order["phase"]} {type(error).__name__}')

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
                    tracked = {o['symbol'] for o in self.state['orders'].values() if o['phase'] not in ('DONE','NO_FILL')}
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
