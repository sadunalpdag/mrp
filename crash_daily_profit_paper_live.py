"""Read-only Binance USD-M account paper monitor; NEVER sends/cancels orders.

Isolated from crash_night trading state. No claim that virtual limit touch is a fill.
All Binance paths are strictly GET-only and use the broker's shared REST budget.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

PARIS = ZoneInfo('Europe/Paris')
POLL_SECONDS = 300


def dec(value):
    try:
        d = Decimal(str(value))
        if not d.is_finite():
            raise ValueError
        return d
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError('INVALID_PAPER_NUMBER') from None


def date_paris(t):
    return datetime.fromtimestamp(t, PARIS).date().isoformat()


class DailyProfitPaperMonitor:
    """Observe actual account equity but create only virtual events and JSON state."""

    def __init__(self, directory, broker, log, deliver=None, clock=time.time, interval=POLL_SECONDS):
        self.broker, self.log, self.deliver, self.clock = broker, log, deliver, clock
        self.interval = max(60, int(interval))
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, 'daily_profit_paper_state.json')
        self.lock = threading.RLock()
        self.thread = None
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as f:
                self.state = json.load(f)
            if self.state.get('version') != 1:
                raise ValueError('INVALID_PAPER_STATE_VERSION')
        else:
            self.state = dict(version=1, enabled=True, target_pct='5',
                              base_ms=None, transfer_checked_ms=None,
                              transfer_ids=[], cumulative_transfers='0',
                              peak=None, day=None, target=None, trigger=None,
                              last=None, alert_pending=False)
        self._save()

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _get(self, path, params=None):
        # Fixed allowlist; impossible to submit a trading order through this class.
        if path not in ('/fapi/v3/account', '/fapi/v3/positionRisk', '/fapi/v1/income'):
            raise ValueError('PAPER_READ_ONLY_PATH')
        return self.broker.request('GET', path, params)

    def _equity(self, account):
        if not isinstance(account, dict) or account.get('multiAssetsMargin') is not False:
            raise ValueError('PAPER_UNSUPPORTED_MULTI_ASSET_OR_ACCOUNT')
        # Non-USDT collateral requires a separate conversion/validation model.
        assets = account.get('assets')
        if not isinstance(assets, list):
            raise ValueError('PAPER_MISSING_ASSETS')
        for a in assets:
            if a.get('asset') != 'USDT' and abs(dec(a.get('marginBalance', '0'))) > Decimal('0.01'):
                raise ValueError('PAPER_NON_USDT_MARGIN')
        margin = dec(account['totalMarginBalance'])
        wallet = dec(account['totalWalletBalance'])
        unreal = dec(account['totalUnrealizedProfit'])
        if margin <= 0 or abs(margin - wallet - unreal) > Decimal('0.1'):
            raise ValueError('PAPER_EQUITY_INCONSISTENT')
        return margin

    def _update_transfers(self, now_ms):
        start_ms = int(self.state['base_ms'])
        since = max(start_ms, int(self.state.get('transfer_checked_ms') or start_ms) - 5 * 60 * 1000)
        if now_ms <= since:
            return
        # Binance incomeHistory supports 7-day windows; query only a short interval.
        if now_ms - since > 6 * 24 * 3600 * 1000:
            raise ValueError('PAPER_TRANSFER_HISTORY_GAP')
        rows = self._get('/fapi/v1/income', {'startTime': since,
                                             'endTime': now_ms, 'limit': 1000})
        if not isinstance(rows, list) or len(rows) >= 1000:
            raise ValueError('PAPER_TRANSFER_HISTORY_INCOMPLETE')
        seen = set(self.state.get('transfer_ids', []))
        flows = dec(self.state.get('cumulative_transfers', '0'))
        for r in rows:
            t = int(r['time'])
            if not start_ms < t <= now_ms:
                continue
            key = str(r.get('tranId') or f"{t}:{r.get('income')}:{r.get('asset')}:{r.get('info')}")
            # Only cashflow-like income types shift the baseline. Ordinary PnL,
            # fees and funding are part of performance, not transfers.
            if r.get('incomeType') not in ('TRANSFER', 'INTERNAL_TRANSFER',
                  'CROSS_COLLATERAL_TRANSFER', 'COIN_SWAP_DEPOSIT',
                  'COIN_SWAP_WITHDRAW', 'STRATEGY_UMFUTURES_TRANSFER'):
                continue
            key = f"{r.get('incomeType')}:{key}"
            if key not in seen:
                if r.get('asset') != 'USDT':
                    raise ValueError('PAPER_NON_USDT_TRANSFER')
                flows += dec(r['income'])
                seen.add(key)
        self.state['transfer_ids'] = sorted(seen)[-3000:]
        self.state['cumulative_transfers'] = str(flows)
        self.state['transfer_checked_ms'] = now_ms

    def _positions(self):
        rows = self._get('/fapi/v3/positionRisk')
        if not isinstance(rows, list):
            raise ValueError('PAPER_POSITION_RESPONSE')
        snapshot = []
        for r in rows:
            amt = dec(r['positionAmt'])
            if amt == 0:
                continue
            px = dec(r['markPrice'])
            if px <= 0:
                raise ValueError('PAPER_BAD_MARK_PRICE')
            side = r.get('positionSide', 'BOTH')
            if side not in ('LONG', 'SHORT', 'BOTH'):
                raise ValueError('PAPER_BAD_SIDE')
            snapshot.append(dict(symbol=r['symbol'], position_side=side,
                                 quantity=str(abs(amt)), observed_price=str(px),
                                 direction='SHORT' if side == 'SHORT' or (side == 'BOTH' and amt < 0) else 'LONG',
                                 virtual_level_revisited=False))
        return snapshot

    def _record_touches(self, positions):
        trig = self.state.get('trigger')
        if not trig:
            return
        live = {(p['symbol'], p['position_side']): dec(p['observed_price']) for p in positions}
        for p in trig['positions']:
            if p['virtual_level_revisited']:
                continue
            mark = live.get((p['symbol'], p['position_side']))
            if mark is None:
                p['not_observed_open'] = True  # NOT proof of a virtual limit fill
            elif ((p['direction'] == 'LONG' and mark >= dec(p['observed_price'])) or
                  (p['direction'] == 'SHORT' and mark <= dec(p['observed_price']))):
                p['virtual_level_revisited'] = True
        trig['indicative_touches'] = sum(bool(p['virtual_level_revisited']) for p in trig['positions'])

    def step(self):
        """One bounded live sample, with ZERO Binance write methods or trading toggles."""
        now = self.clock()
        now_ms = int(now * 1000)
        # REST responses collected before locking so Telegram never blocks on network.
        account = self._get('/fapi/v3/account')
        raw_equity = self._equity(account)
        with self.lock:
            if self.state['base_ms'] is None:
                self.state['base_ms'] = now_ms
                self.state['transfer_checked_ms'] = now_ms
                self.state['peak'] = str(raw_equity)
                self.state['day'] = date_paris(now)
                self.state['target'] = str(raw_equity * (1 + dec(self.state['target_pct']) / 100))
                self.state['last'] = dict(at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                                          equity=str(raw_equity), adjusted_equity=str(raw_equity),
                                          data_quality='BASELINE_AT_ACTIVATION')
                self._save()
                self.log('[DAILY PROFIT PAPER] baseline initialized; no historical peak inferred; no orders')
                return
            self._update_transfers(now_ms)
            adjusted = raw_equity - dec(self.state['cumulative_transfers'])
            if adjusted <= 0:
                raise ValueError('PAPER_ADJUSTED_EQUITY_INVALID')
            today = date_paris(now)
            if today < self.state['day']:
                raise ValueError('PAPER_CLOCK_MOVED_BACK')
            if today != self.state['day']:
                self.state['day'] = today
                self.state['target'] = str(dec(self.state['peak']) *
                                           (1 + dec(self.state['target_pct']) / 100))
            self.state['peak'] = str(max(dec(self.state['peak']), adjusted))
            self.state['last'] = dict(at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                                      equity=str(raw_equity), adjusted_equity=str(adjusted),
                                      data_quality='USDT_TRANSFER_INCOME_ADJUSTED_PAPER_ESTIMATE')
            hit = bool(self.state['enabled'] and not self.state['trigger'] and
                       adjusted >= dec(self.state['target']))
            if hit or self.state['trigger']:
                # A missing live position sample prevents any trigger snapshot.
                positions = self._positions()
                if hit:
                    self.state['trigger'] = dict(at=self.state['last']['at'],
                                                 day=today, target=self.state['target'],
                                                 adjusted_equity=str(adjusted), positions=positions,
                                                 indicative_touches=0)
                    self.state['alert_pending'] = True
                    self.log('[DAILY PROFIT PAPER] PAPER_TARGET_REACHED; virtual prices frozen; NO ORDERS')
                else:
                    self._record_touches(positions)
            self._save()
        if self.state.get('alert_pending') and self.deliver:
            msg = (f"📸 SANAL kâr hedefi: %{self.state['target_pct']} | "
                   f"hesap {self.state['last']['adjusted_equity']} USDT | "
                   f"hedef {self.state['target']} USDT | "
                   f"{len(self.state['trigger']['positions'])} pozisyon fotoğrafı. "
                   'GERÇEK EMİR YOK. /gunluk_kar_durum')
            try:
                receipt = self.deliver(msg)
                if isinstance(receipt, dict) and receipt.get('ok'):
                    with self.lock:
                        self.state['alert_pending'] = False
                        self._save()
            except Exception:
                self.log('[DAILY PROFIT PAPER] telegram delivery failed; event retained')

    def control(self, action):
        with self.lock:
            if action == 'paper_status':
                pass
            elif action in ('paper_on', 'paper_off'):
                self.state['enabled'] = action == 'paper_on'
                self._save()
            elif action.startswith('paper_target:'):
                raw = action.split(':', 1)[1]
                try:
                    val = dec(raw)
                    if not Decimal('0.5') <= val <= Decimal('20'):
                        raise ValueError
                except ValueError:
                    return '⚠️ Kullanım: /gunluk_kar 0.5–20'
                self.state['target_pct'] = str(val)
                # Today's threshold is frozen. Telegram changes apply next Paris day.
                if self.state['base_ms'] is None:
                    self.state['target'] = None
                self._save()
            else:
                return 'Bilinmeyen sanal takip komutu.'
            s = self.state
            last = (s.get('last') or {}).get('adjusted_equity', '?')
            trig = s.get('trigger')
            return (f"📊 SANAL günlük kâr takibi {'AÇIK' if s['enabled'] else 'KAPALI'} | "
                    f"zirve {s.get('peak') or '?'} USDT | son {last} USDT | "
                    f"hedef {s.get('target') or '?'} USDT | "
                    f"ayarlanan %{s['target_pct']} (değişiklik ertesi gün) | "
                    f"fotoğraf {'VAR' if trig else 'YOK'} | "
                    f"sanal fiyat dönüşü {trig.get('indicative_touches', 0) if trig else 0}/"
                    f"{len(trig['positions']) if trig else 0}. "
                    'Gerçek emir veya giriş kilidi uygulanmaz.')

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def run():
            while True:
                try:
                    self.step()
                except Exception as err:
                    self.log(f'[DAILY PROFIT PAPER] sample skipped: {type(err).__name__}: {str(err)[:90]}')
                time.sleep(self.interval)
        self.thread = threading.Thread(target=run, name='daily-profit-paper', daemon=True)
        self.thread.start()
