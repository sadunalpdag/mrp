"""Independent, persisted Telegram health reports; not a trading signal."""
import json
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ZONE = ZoneInfo('Europe/Paris')
INTERVAL = 2 * 3600


def stamp(value):
    return datetime.fromtimestamp(value, ZONE).strftime('%d/%m %H:%M:%S') if value else 'henüz yok'


class Heartbeat:
    def __init__(self, directory, deliver, health, log, clock=time.time):
        self.directory, self.deliver, self.health, self.log, self.clock = directory, deliver, health, log, clock
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, 'crash_heartbeat.json')
        self.lock = threading.RLock()
        self.metrics = {}
        self.thread = None
        self.state = {'next_due': clock()+180}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding='utf-8') as handle:
                    saved = json.load(handle)
                if not isinstance(saved, dict) or not isinstance(saved.get('next_due'), (int, float)):
                    raise ValueError('INVALID_HEARTBEAT_STATE')
                self.state = saved
                self.metrics = self.state.get('metrics', {})
                if not isinstance(self.metrics, dict):
                    self.metrics = {}
            except (ValueError, OSError):
                self.log('[CRASH HEARTBEAT] Invalid saved schedule; restarting in 180 seconds')
        # A previous process's successful scan is not proof this process is healthy.
        self.started = clock()

    def update(self, **values):
        with self.lock:
            self.metrics.update(values)

    def save(self):
        with self.lock:
            self.state['metrics'] = dict(self.metrics)
            with open(self.path+'.tmp', 'w', encoding='utf-8') as handle:
                json.dump(self.state, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self.path+'.tmp', self.path)

    def message(self, now):
        health = self.health()
        with self.lock:
            m = dict(self.metrics)
        age = health.get('ticker_age_seconds')
        scan = m.get('scan_completed_at', 0)
        fresh = (health.get('connected') is True and isinstance(age, (float, int))
                 and 0 <= age <= 120 and health.get('ticker_count', 0) > 0)
        scan_ok = self.started <= scan <= now and now-scan <= 600
        candles_ok = m.get('candles_ok', 0)
        candle_problem = candles_ok == 0 and m.get('candles_wait', 0) > 0
        delivery_problem = m.get('delivery_error') not in (None, '', 'yok')
        healthy = fresh and scan_ok and not candle_problem and not m.get('scan_errors', 0) and not delivery_problem
        signal_at = m.get('last_signal_at')
        if not fresh:
            reason = 'Güncel fiyat akışı doğrulanamıyor.'
        elif not scan_ok:
            reason = 'Son 10 dakikada tamamlanan tarama doğrulanamıyor.'
        elif m.get('scan_errors', 0):
            reason = 'Son taramada hata var; Render loglarını kontrol edin.'
        elif delivery_problem:
            reason = 'Son sinyalin Telegram teslimi doğrulanamadı.'
        elif candle_problem:
            reason = 'Son taramada giriş hesabına uygun mum verisi bulunamadı.'
        else:
            reason = 'Fiyat akışı ve tarama güncel. Sinyal olmaması tek başına arıza değildir.'
        age_text = f'{age:.0f} sn' if isinstance(age, (float, int)) else 'bilinmiyor'
        return (f"{'✅' if healthy else '⚠️'} CRASH BOUNCE — 2 SAATLİK DURUM\n"
                f"{reason}\n"
                f"WS: {'bağlı' if health.get('connected') else 'bağlı değil'} | Son fiyat: {age_text}\n"
                f"Önbellekte sembol: {health.get('ticker_count', 0)}\n"
                f"Son tamamlanan tarama: {stamp(scan)}\n"
                f"Son tarama: uygun mum {candles_ok}, eksik/eski mum {m.get('candles_wait', 0)}, "
                f"yeni aday {m.get('candidates', 0)}\n"
                f"Son başarılı sinyal gönderimi: {stamp(signal_at)} {m.get('last_signal_symbol', '')}\n"
                f"Son sinyal gönderim hatası: {m.get('delivery_error', 'yok')}\n"
                f"Otomatik işlem: {health.get('trading', 'KAPALI')}\n"
                f"Saat dilimi: Europe/Paris. Bu mesaj işlem sinyali değildir.")

    def step(self):
        now = self.clock()
        if now < self.state.get('next_due', 0):
            return
        # Persist retry delay before transport to avoid restart notification floods.
        self.state['next_due'] = now+300
        self.save()
        receipt = self.deliver(self.message(now))
        if receipt.get('ok'):
            self.state.update(next_due=now+INTERVAL, last_sent_at=now,
                              message_id=receipt.get('message_id'), delivery_error=None)
        elif receipt.get('ambiguous'):
            self.state.update(next_due=now+INTERVAL, delivery_error='DELIVERY_UNKNOWN')
        else:
            self.state['delivery_error'] = str(receipt.get('error', 'DELIVERY_FAILED'))[:100]
        self.save()
        self.log(f"[CRASH HEARTBEAT] accepted={bool(receipt.get('ok'))} error={self.state.get('delivery_error')}")

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def run():
            while True:
                try:
                    self.step()
                except Exception as error:
                    self.log(f'[CRASH HEARTBEAT ERR] {type(error).__name__}')
                time.sleep(15)
        self.thread = threading.Thread(target=run, name='crash-heartbeat', daemon=True)
        self.thread.start()
