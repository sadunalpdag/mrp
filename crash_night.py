"""Opt-in night executor. Importing this module never sends an order.

The caller supplies an exchange adapter and explicitly enables live operation.
Only fresh, Telegram-acknowledged OPEN transitions can be submitted.
"""
import fcntl
import hashlib
import json
import os
import queue
import threading
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from zoneinfo import ZoneInfo

ZONE = ZoneInfo('Europe/Paris')
D = Decimal


def night(now):
    hour = datetime.fromtimestamp(now, ZONE).hour
    return hour >= 20 or hour < 8


def day(now):
    return datetime.fromtimestamp(now, ZONE).date().isoformat()


def rounded(value, step, up=False):
    return (D(str(value))/D(str(step))).to_integral_value(
        rounding=ROUND_UP if up else ROUND_DOWN)*D(str(step))


def plan(record, info):
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
    qty = rounded(D('50')/entry, lot['stepSize'])
    if qty < D(lot['minQty']) or qty > D(lot['maxQty']) or qty <= 0:
        raise ValueError('LOT_SIZE')
    minimum = D(filters.get('MIN_NOTIONAL', {}).get('notional', '0'))
    if qty*entry < minimum:
        raise ValueError('MIN_NOTIONAL')
    return {'price': str(entry), 'tp': str(tp), 'quantity': str(qty)}


class BrokerError(Exception):
    """Sanitized API code only: never include credentials or signed URLs."""


class BinanceBroker:
    def __init__(self, key, secret):
        if not key or not secret:
            raise ValueError('MISSING_BINANCE_CREDENTIALS')
        self.key, self.secret = key, secret

    def request(self, method, path, params=None, signed=True):
        import requests
        import hmac
        from urllib.parse import urlencode
        params = dict(params or {})
        if signed:
            params.update(timestamp=int(time.time()*1000), recvWindow=5000)
        query = urlencode(params)
        if signed:
            query += '&signature='+hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        try:
            response = requests.request(method, 'https://fapi.binance.com'+path+'?'+query,
                headers={'X-MBX-APIKEY': self.key}, timeout=(5, 10))
            body = response.json()
        except Exception:
            raise BrokerError('TRANSPORT_OR_RESPONSE_UNKNOWN') from None
        if response.status_code >= 400:
            code = body.get('code', 'UNKNOWN') if isinstance(body, dict) else 'UNKNOWN'
            raise BrokerError(f'BINANCE_{code}')
        return body

    def info(self, symbol):
        data = self.request('GET', '/fapi/v1/exchangeInfo', signed=False)
        return next(s for s in data['symbols'] if s['symbol'] == symbol)

    def preflight(self, symbol):
        if self.request('GET', '/fapi/v1/positionSide/dual').get('dualSidePosition') is not False:
            raise ValueError('ONE_WAY_MODE_REQUIRED')
        positions = self.request('GET', '/fapi/v3/positionRisk')
        active = {p['symbol'] for p in positions if D(p['positionAmt']) != 0}
        if symbol in active or len(active) >= 5:
            raise ValueError('POSITION_EXISTS_OR_MAX_5')
        orders = self.request('GET', '/fapi/v1/openOrders')
        if any(o['symbol'] == symbol for o in orders):
            raise ValueError('EXISTING_SYMBOL_ORDER')
        # Pending orders on other symbols can also occupy future position slots.
        if len(active | {o['symbol'] for o in orders}) >= 5:
            raise ValueError('MAX_5_WITH_PENDING_ORDERS')
        result = self.request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': 10})
        if result.get('leverage') != 10:
            raise ValueError('LEVERAGE_NOT_CONFIRMED')

    def order(self, symbol, client_id):
        return self.request('GET', '/fapi/v1/order', {'symbol': symbol, 'origClientOrderId': client_id})

    def buy(self, symbol, p, client_id):
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='BUY', type='LIMIT',
            timeInForce='IOC', quantity=p['quantity'], price=p['price'], newClientOrderId=client_id,
            newOrderRespType='RESULT'))

    def tp(self, symbol, qty, price, client_id):
        return self.request('POST', '/fapi/v1/order', dict(symbol=symbol, side='SELL', type='LIMIT',
            timeInForce='GTC', quantity=qty, price=price, reduceOnly='true', newClientOrderId=client_id))


class NightExecutor:
    def __init__(self, directory, broker, deliver, log, clock=time.time, enabled=False):
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
            self.state['paused'] = not enabled
            self.save()
            with open(marker, 'a') as handle:
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.guard.close()
            raise
        self.started = clock()
        self.queue = queue.Queue(maxsize=100)
        self.thread = None
        self.next_poll = 0
        self.enabled = threading.Event()
        self.armed_at = clock()
        if enabled:
            self.enabled.set()
        self.controls = queue.Queue(maxsize=10)

    def save(self):
        with open(self.path+'.tmp', 'w', encoding='utf-8') as handle:
            json.dump(self.state, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self.path+'.tmp', self.path)

    def summary(self):
        count = sum(o['day'] == day(self.clock()) for o in list(self.state['orders'].values()))
        return f"{'DURDU: '+self.state['halt'] if self.state['halt'] else ('AÇIK' if self.enabled.is_set() else 'KAPALI')} | bugün {count}/10 | 20:00–08:00 Paris"

    def apply_control(self, action):
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
                    '20:00–08:00 Paris | 10x | en fazla 50 USDT pozisyon / yaklaşık 5 USDT teminat\n'
                    'Günde 10 giriş denemesi, aynı anda en fazla 5 sembol. TP +%1; SL yok.\n'+self.summary())
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
                or not night(now) or not 0 <= now-observed_at <= 5 or observed_at < self.started
                or record.get('v2_status') != 'STRONG_V2' or record.get('v2_sent') is not True
                or record.get('tracking_status') != 'OPEN' or sid in self.state['orders']):
            return
        if sum(o['day'] == day(now) for o in self.state['orders'].values()) >= 10:
            return
        if any(o['phase'] not in ('DONE', 'NO_FILL') and o['symbol'] == symbol for o in self.state['orders'].values()):
            return
        try:
            p = plan(record, self.broker.info(symbol))
            self.broker.preflight(symbol)
        except Exception as error:
            self.alert(f'⚠️ OTOMATİK GİRİŞ YAPILMADI | {symbol}\nTP/giriş ön kontrolü başarısız: {type(error).__name__}: {str(error)[:80] if isinstance(error, (ValueError, BrokerError)) else "CHECK_FAILED"}')
            return
        # Network preflight can take seconds. Recheck window and freshness.
        now = self.clock()
        if not self.enabled.is_set() or not night(now) or not 0 <= now-observed_at <= 5:
            return
        cid = 'cbn'+hashlib.sha256(sid.encode()).hexdigest()[:24]
        order = dict(symbol=symbol, day=day(now), phase='ENTRY_UNKNOWN', entry_id=cid+'e', tp_id=cid+'t', **p)
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

    def reconcile(self, order):
        symbol = order['symbol']
        try:
            if order['phase'] == 'ENTRY_UNKNOWN':
                entry = self.broker.order(symbol, order['entry_id'])
                if entry.get('status') not in ('FILLED', 'EXPIRED', 'CANCELED', 'EXPIRED_IN_MATCH', 'REJECTED'):
                    raise ValueError('ENTRY_RESULT_NOT_FINAL')
                qty = D(entry['executedQty'])
                if qty == 0:
                    order['phase'] = 'NO_FILL'
                    self.save()
                    self.alert(f'ℹ️ OTOMATİK İŞLEM | {symbol}\nIOC giriş dolmadı; açık giriş emri bırakılmadı. Günlük hak kullanıldı.')
                    return
                if qty < 0 or qty > D(order['quantity']):
                    raise ValueError('INVALID_FILLED_QUANTITY')
                order['filled_qty'] = str(qty)
                order['phase'] = 'TP_UNKNOWN'
                self.save()  # On restart query this client ID; never duplicate TP.
                try:
                    self.broker.tp(symbol, order['filled_qty'], order['tp'], order['tp_id'])
                except Exception:
                    pass
            if order['phase'] in ('TP_UNKNOWN', 'PROTECTED'):
                tp = self.broker.order(symbol, order['tp_id'])
                if (tp.get('side') != 'SELL' or tp.get('reduceOnly') is not True
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
            reason = ('GİRİŞ SONUCU BELİRSİZ' if order['phase'] == 'ENTRY_UNKNOWN' else 'TP DOĞRULANAMADI')
            if not order.get('warned'):
                order['warned'] = True
                self.halt(reason, symbol)
            self.log(f'[CRASH NIGHT RECONCILE] {symbol} {order["phase"]} {type(error).__name__}')

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
        # Manage protective orders even outside the entry window or when halted.
        if self.clock() >= self.next_poll:
            self.next_poll = self.clock()+30
            for order in list(self.state['orders'].values()):
                if order['phase'] in ('ENTRY_UNKNOWN', 'TP_UNKNOWN', 'PROTECTED'):
                    self.reconcile(order)
        self.deliver_pending()
        try:
            record, observed_at = self.queue.get_nowait()
        except queue.Empty:
            return
        self.enter(record, observed_at)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
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
