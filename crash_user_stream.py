"""USD-M account events. No order submission, credentials or payloads in logs."""
import json
import threading
import time
from decimal import Decimal
from urllib.parse import urlencode


def order_update(event):
    o = event['o']
    result = dict(symbol=o['s'], clientOrderId=o['c'], side=o['S'],
                  positionSide=o.get('ps', 'BOTH'), type=o['o'], status=o['X'],
                  executedQty=o['z'], origQty=o['q'], price=o['p'],
                  reduceOnly=o.get('R', False), updateTime=int(event['E']))
    for name in ('executedQty', 'origQty', 'price'):
        value = Decimal(str(result[name]))
        if not value.is_finite() or value < 0:
            raise ValueError('INVALID_ORDER_VALUE')
    return result


class AccountStream:
    def __init__(self, broker, emit, log, websocket, clock=time.time, blocked=lambda:False):
        self.broker, self.emit, self.log, self.websocket = broker, emit, log, websocket
        self.clock, self.blocked = clock, blocked
        self.stop_event = threading.Event()
        self.thread = None
        self.ws = None
        self.connected = False
        self.last_event = 0
        self.opened_at = 0
        self.renew_at = 0
        self.reconnects = 0

    def summary(self):
        age = f'{int(max(0,self.clock()-self.last_event))} sn once' if self.last_event else 'henuz yok'
        return f"Hesap WS: {'bagli' if self.connected else 'bagli degil / REST yedek'} | son olay: {age}"

    def on_open(self, ws):
        if ws is not self.ws:
            return
        self.connected = True
        self.opened_at = self.clock()
        self.reconnects += 1
        self.emit({'e':'ACCOUNT_STREAM_RESYNC'})
        self.log('[CRASH ACCOUNT WS OPEN] hesap akisi baglandi; REST mutabakati isteniyor')

    def on_message(self, ws, message):
        if ws is not self.ws:
            return
        try:
            event = json.loads(message)
            event = event.get('data', event)
            kind = event.get('e')
            if kind == 'listenKeyExpired':
                self.connected = False
                self.log('[CRASH ACCOUNT WS EXPIRED] anahtar yenilenecek')
                ws.close()
            elif kind in ('ORDER_TRADE_UPDATE', 'ACCOUNT_UPDATE'):
                self.last_event = self.clock()
                self.emit(event)
        except Exception as error:
            self.log(f'[CRASH ACCOUNT WS MESSAGE ERR] {type(error).__name__}')
            self.emit({'e':'ACCOUNT_STREAM_RESYNC'})

    def on_close(self, ws, *args):
        if ws is self.ws:
            self.connected = False
            self.emit({'e':'ACCOUNT_STREAM_RESYNC'})
            self.log('[CRASH ACCOUNT WS CLOSED] yeniden baglanacak; REST yedek aktif')

    def on_error(self, ws, error):
        # Exception strings may contain the secret listenKey URL.
        self.log(f'[CRASH ACCOUNT WS ERR] {type(error).__name__}')

    def keepalive(self):
        if not self.connected or self.clock() < self.renew_at:
            return
        if self.blocked() or self.clock() < self.broker.cooldown_until:
            self.renew_at = self.clock()+60
            return
        try:
            self.broker.request('PUT', '/fapi/v1/listenKey', signed=False)
            self.renew_at = self.clock()+1800
            self.log('[CRASH ACCOUNT WS KEEPALIVE] oturum uzatildi')
        except Exception as error:
            self.log(f'[CRASH ACCOUNT WS KEEPALIVE ERR] {type(error).__name__}')
            self.connected = False
            if self.ws:
                self.ws.close()

    def _maintain(self):
        while not self.stop_event.wait(15):
            self.keepalive()
            if self.connected and self.clock()-self.opened_at >= 23*3600:
                self.ws.close()

    def _run(self):
        delay = 5
        while not self.stop_event.is_set():
            if self.blocked() or self.clock() < self.broker.cooldown_until:
                self.stop_event.wait(30)
                continue
            try:
                data = self.broker.request('POST', '/fapi/v1/listenKey', signed=False)
                key = data.get('listenKey')
                if not isinstance(key, str) or not key:
                    raise ValueError('MISSING_LISTEN_KEY')
                # New private routing; no key/URL is ever logged or persisted.
                url = 'wss://fstream.binance.com/private/ws?' + urlencode({
                    'listenKey': key,
                    'events': 'ORDER_TRADE_UPDATE/ACCOUNT_UPDATE/listenKeyExpired'})
                self.renew_at = self.clock()+1800
                self.opened_at = 0
                self.ws = self.websocket.WebSocketApp(url, on_open=self.on_open,
                    on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
                self.ws.run_forever(ping_interval=60, ping_timeout=20)
                if self.opened_at and self.clock()-self.opened_at > 60:
                    delay = 5
            except Exception as error:
                self.log(f'[CRASH ACCOUNT WS CONNECT ERR] {type(error).__name__}')
            finally:
                self.connected = False
            self.stop_event.wait(delay)
            delay = min(300, delay*2)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name='crash-account-ws', daemon=True)
        self.thread.start()
        threading.Thread(target=self._maintain, name='crash-account-keepalive', daemon=True).start()

    def stop(self):
        self.stop_event.set()
        if self.ws:
            self.ws.close()
