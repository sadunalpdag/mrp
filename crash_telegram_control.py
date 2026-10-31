"""Owner-only commands. Never runs legacy/free-form Telegram trading handlers."""
import json
import os
import threading
import time

COMMANDS = {'/otomatik_ac': 'on', '/otomatik_kapat': 'off', '/otomatik_durum': 'status'}


class TelegramControl:
    def __init__(self, directory, token, chat_id, owner_id, action, deliver, log, clock=time.time, fetch=None, counter_only=False, require_owner=True):
        self.counter_only = counter_only
        self.require_owner = require_owner
        self.token, self.chat_id = token, str(chat_id)
        # Private chat ID equals the user ID. Group/channel IDs cannot infer an owner.
        fallback = self.chat_id if self.chat_id.isdigit() and int(self.chat_id) > 0 else ''
        self.owner_id = str(owner_id or fallback)
        if require_owner and (not self.owner_id.isdigit() or int(self.owner_id) <= 0):
            raise ValueError('TELEGRAM_OWNER_ID_REQUIRED_FOR_GROUP')
        self.action, self.deliver, self.log, self.clock = action, deliver, log, clock
        self.fetch = fetch or self.get_updates
        self.started = clock()
        self.path = os.path.join(directory, 'crash_telegram_control.json')
        self.offset = 0
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as handle:
                self.offset = int(json.load(handle)['offset'])
            if self.offset < 0:
                raise ValueError('INVALID_UPDATE_OFFSET')
        self.thread = None
        self.last_poll_at = None
        self.last_error = None

    def get_updates(self, offset):
        import requests
        try:
            response = requests.get(f'https://api.telegram.org/bot{self.token}/getUpdates',
                params={'offset': offset, 'timeout': 20, 'allowed_updates': '["message"]'}, timeout=(5, 30))
            data = response.json()
        except Exception:
            raise RuntimeError('TELEGRAM_TRANSPORT_ERROR') from None
        if response.status_code != 200 or data.get('ok') is not True:
            raise RuntimeError(f'TELEGRAM_HTTP_{response.status_code}')
        return data['result']

    def save_offset(self, value):
        with open(self.path+'.tmp', 'w', encoding='utf-8') as handle:
            json.dump({'offset': value}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self.path+'.tmp', self.path)
        self.offset = value

    def command(self, update):
        msg = update.get('message')
        if not isinstance(msg, dict):
            return None
        sender = msg.get('from', {})
        if (str(msg.get('chat', {}).get('id')) != self.chat_id
                or (self.require_owner and str(sender.get('id')) != self.owner_id) or sender.get('is_bot') is not False
                or msg.get('sender_chat') or msg.get('forward_origin') or msg.get('forward_date')
                or msg.get('forward_from') or msg.get('forward_sender_name')
                or msg.get('via_bot') or msg.get('edit_date')):
            return None
        date = msg.get('date', 0)
        now = self.clock()
        if not isinstance(date, (float, int)) or date < self.started or not 0 <= now-date <= 90:
            return None
        # Exact commands only. Quoted, edited, forwarded and free-form signals cannot arm trading.
        text = msg.get('text', '').strip()
        if text == '/rapor':
            return 'report'
        parts = text.split()
        if text in ('/limit', '/tutar', '/sembol'):
            return 'status'
        if len(parts) == 2 and parts[0] in ('/limit', '/tutar', '/sembol') and len(parts[1]) <= 12:
            return {'/limit':'limit:', '/tutar':'amount:', '/sembol':'symbols:'}[parts[0]] + parts[1]
        if self.counter_only:
            if text in ('/sayac', '/otomatik_durum'):
                return 'status'
            parts = text.split()
            if len(parts) == 2 and parts[0] == '/sayac' and parts[1].isascii() and parts[1].isdigit() and len(parts[1]) <= 3:
                return 'counter:'+parts[1]
            return None
        return COMMANDS.get(text)

    def status(self):
        if self.last_error:
            return 'hata: '+self.last_error
        if self.last_poll_at is None or self.clock()-self.last_poll_at > 90:
            return 'güncel komut bağlantısı doğrulanamadı'
        return 'aktif'

    def step(self):
        updates = self.fetch(self.offset)
        self.last_poll_at = self.clock()
        self.last_error = None
        for update in updates:
            uid = update.get('update_id')
            if not isinstance(uid, int) or uid < self.offset:
                continue
            # Persist before acting: restart cannot replay an activation command.
            self.save_offset(uid+1)
            command = self.command(update)
            if command is None:
                continue
            try:
                reply = self.action(command)
            except Exception as error:
                self.log(f'[CRASH CONTROL ACTION ERR] {type(error).__name__}')
                reply = '⚠️ Komut tamamlanamadı. /otomatik_durum ile kontrol edin; Render ayarlarını inceleyin.'
            receipt = self.deliver(reply)
            self.log(f'[CRASH CONTROL] command={command} reply_accepted={bool(receipt.get("ok"))}')

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def run():
            while True:
                try:
                    self.step()
                except Exception as error:
                    # Never print a requests exception containing the bot-token URL.
                    code = str(error) if type(error) is RuntimeError and str(error).startswith('TELEGRAM_') else type(error).__name__
                    self.last_error = code
                    self.log(f'[CRASH CONTROL ERR] {code}')
                    time.sleep(15)
                time.sleep(1)
        self.thread = threading.Thread(target=run, name='crash-telegram-control', daemon=True)
        self.thread.start()
