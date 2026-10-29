"""Missing-TP repair for one-way and hedge USDT perpetual longs; serialized by NightExecutor.

No API calls at import. Disabled unless explicitly configured by the operator.
"""
import hashlib
from decimal import Decimal as D
from crash_night import BrokerError, error_details, rounded
from crash_policy import supported_symbol


def truth(value):
    return value is True or value == 'true'


def tp_coverage(orders, symbol, entry, qty, position_side="BOTH"):
    """Return full/partial/none/conflict; never replace an existing closing order."""
    covered = D('0')
    conflict = False
    for o in orders:
        if o.get('symbol') != symbol or o.get('side') != 'SELL':
            continue
        if o.get('positionSide', 'BOTH') != position_side:
            continue
        status = o.get('status', o.get('algoStatus'))
        if status not in ('NEW', 'PARTIALLY_FILLED'):
            conflict = True  # transitional/unknown order states are not proof of absence
            continue
        typ = o.get('type', o.get('orderType'))
        close_all = truth(o.get('closePosition'))
        reducing = position_side == 'LONG' or truth(o.get('reduceOnly')) or close_all
        if not reducing:
            conflict = True
            continue
        if typ == 'LIMIT':
            target = D(o['price'])
        elif typ in ('TAKE_PROFIT', 'TAKE_PROFIT_MARKET'):
            target = D(o.get('triggerPrice', o.get('stopPrice', '0')))
        else:
            conflict = True  # includes SL/trailing orders: leave user's order set untouched
            continue
        if target <= entry:
            conflict = True
            continue
        if close_all:
            return 'full'
        remaining = D(o.get('origQty', o.get('quantity', '0'))) - D(o.get('executedQty', '0'))
        covered += max(D('0'), remaining)
    if covered >= qty:
        return 'full'
    if covered > 0:
        return 'partial'
    return 'conflict' if conflict else 'none'


def repair_plan(position, info):
    if info.get('status') != 'TRADING' or info.get('contractType') != 'PERPETUAL':
        raise ValueError('SYMBOL_NOT_TRADING')
    entry, qty = D(position['entryPrice']), D(position['positionAmt'])
    if not entry.is_finite() or not qty.is_finite() or entry <= 0 or qty <= 0:
        raise ValueError('INVALID_POSITION')
    fs = {f['filterType']: f for f in info['filters']}
    pf, lot = fs['PRICE_FILTER'], fs['LOT_SIZE']
    tp = rounded(entry*D('1.01'), pf['tickSize'], True)
    if tp < D(pf['minPrice']) or tp > D(pf['maxPrice']):
        raise ValueError('PRICE_FILTER')
    if qty != rounded(qty, lot['stepSize']) or not D(lot['minQty']) <= qty <= D(lot['maxQty']):
        raise ValueError('LOT_SIZE')
    return str(qty), str(tp)


LOCAL_REASONS = {
    'INVALID_POSITION_SIDE': 'Pozisyon yönü doğrulanamadı; TP eksik olduğu söylenemez.',
    'PRICE_FILTER': 'TP fiyatı borsanın fiyat sınırlarına uymuyor.',
    'LOT_SIZE': 'Pozisyon miktarı emir adımına veya miktar sınırına uymuyor.',
    'SYMBOL_NOT_TRADING': 'Sembol aktif USDT perpetual işlem koşullarını sağlamıyor.',
    'INVALID_POSITION': 'Geçerli long miktarı veya ortalama giriş fiyatı okunamadı.',
    'TARGET_PASSED': 'Fiyat %1 TP hedefine ulaşmış/geçmiş; otomatik satış yapılmadı.',
    'PARTIAL_TP': 'Mevcut TP miktarı pozisyonun tamamını kapsamıyor; ikinci emir eklenmedi.',
    'ORDER_CONFLICT': 'Başka kapatma/koşullu emir var; mükerrer emir riski nedeniyle ekleme yapılmadı.',
    'POSITION_CHANGED': 'Pozisyon miktarı/girişi değişti; önce mevcut emirleri kontrol edin.',
    'TP_PROPERTIES': 'TP emrinin yönü, miktarı, fiyatı veya reduce-only özelliği doğrulanamadı.',
    'TP_NO_LONGER_ACTIVE': 'Önceki TP artık aktif değil; otomatik tekrar oluşturulmadı.',
}


class ManualTPGuard:
    def __init__(self, executor):
        self.e, self.b = executor, executor.broker
        self.state = executor.state.setdefault('manual_tp', {})
        self.notices = executor.state.setdefault('manual_tp_notices', {})

    def warn(self, symbol, code, explanation, record=None):
        key = symbol+':'+code
        now = self.e.clock()
        if now < self.notices.get(key, 0):
            return
        self.notices[key] = now+1800
        details = '' if not record else f"\nGiriş: {record.get('entry')} | Miktar: {record.get('qty')} | Hedef TP: {record.get('tp', 'hesaplanamadı')}"
        title = 'TP OLUŞTURULAMADI' if record and record.get('phase') == 'REJECTED' else 'TP KONTROLÜ'
        side_label = '' if symbol.startswith('HESAP') else ' | LONG'
        self.e.alert(f'⚠️ {title} | {symbol}{side_label}{details}\nKod: {code}\nNeden: {explanation}\nPiyasa kapatma emri gönderilmedi; pozisyon ve TP durumunu Binance üzerinden kontrol edin.')

    def fail(self, symbol, error, record=None):
        if isinstance(error, BrokerError) and error.code == '-1003':
            now = self.e.clock()
            if now >= self.notices.get('API_RATE_LIMIT', 0):
                self.notices['API_RATE_LIMIT'] = now+300
                wait = max(0, int(getattr(self.b, 'cooldown_until', now)-now))
                self.e.alert(f'⚠️ API SORGULAMASI BEKLETİLDİ | {symbol}\nKod: -1003\nAPI istek sınırı aşıldı; işlem modülü {wait} sn bekleyecek. TP eksikliği doğrulanmadı; mevcut emirler iptal edilmedi.')
            return
        if isinstance(error, ValueError) and str(error) in LOCAL_REASONS:
            self.warn(symbol, str(error), LOCAL_REASONS[str(error)], record)
        else:
            code = error.code if isinstance(error, BrokerError) else type(error).__name__
            self.warn(symbol, code, error_details(error), record)

    def verify(self, symbol, record):
        """UNKNOWN is query-only forever unless the original order is found."""
        try:
            order = self.b.order(symbol, record['client_id'])
            if (order.get('symbol') != symbol or order.get('type') != 'LIMIT'
                    or order.get('positionSide', 'BOTH') != record.get('position_side', 'BOTH') or order.get('side') != 'SELL'
                    or (record.get('position_side', 'BOTH') == 'BOTH' and not truth(order.get('reduceOnly')))
                    or D(order.get('origQty', '0')) != D(record['qty'])
                    or D(order.get('price', '0')) != D(record['tp'])):
                raise ValueError('TP_PROPERTIES')
            status = order.get('status')
            if status in ('NEW', 'PARTIALLY_FILLED', 'FILLED'):
                first = record['phase'] != 'PROTECTED'
                record['phase'] = 'PROTECTED'
                self.e.save()
                if first:
                    self.e.alert(f'✅ TP EMRİ DOĞRULANDI | {symbol}\nGiriş: {record["entry"]} | Miktar: {record["qty"]}\nTP +%1: {record["tp"]} | Durum: {status}')
                return
            record['phase'] = 'HOLD'
            self.e.save()
            raise ValueError('TP_NO_LONGER_ACTIVE')
        except Exception as error:
            if isinstance(error, BrokerError) and error.code == '-1003':
                self.fail(symbol, error)
                return
            reason = record.get('submit_error', error_details(error))
            self.warn(symbol, 'TP_UNCONFIRMED', 'Emir sonucu/koruması doğrulanamadı. '+reason+' Yeni TP tekrar gönderilmedi.', record)

    def scan(self, symbols=None):
        try:
            positions = self.b.positions()
        except Exception as error:
            if isinstance(error, BrokerError) and error.code == '-1003':
                self.fail('HESAP — coin listesi alınamadı', error)
                return
            self.warn('HESAP — coin listesi alınamadı', 'POSITION_QUERY_FAILED',
                      'Açık pozisyonlar sorgulanamadı; TP eksikliği tespit edilmiş değildir. '+error_details(error))
            return  # failed query is not an empty account / absent TP
        active = {p['symbol'] for p in positions if D(p['positionAmt']) > 0 and p.get('positionSide', 'BOTH') in ('BOTH', 'LONG')}
        for symbol, record in self.state.items():
            if (symbols is None or symbol in symbols) and symbol not in active and record['phase'] != 'UNKNOWN':
                record['absent'] = True
        self.e.save()
        excluded = {o['symbol'] for o in self.e.state['orders'].values() if o['phase'] not in ('DONE', 'NO_FILL')}
        for p in positions:
            symbol = p.get('symbol')
            if symbols is not None and symbol not in symbols:
                continue
            if not supported_symbol(symbol) or D(p['positionAmt']) <= 0 or symbol in excluded:
                continue
            if p.get('positionSide') == 'SHORT':
                continue
            try:
                self.inspect(p)
            except Exception as error:
                self.fail(symbol, error, self.state.get(symbol))
            if self.e.clock() < getattr(self.b, 'cooldown_until', 0):
                break

    def inspect(self, p):
        symbol = p['symbol']
        position_side = p.get('positionSide', 'BOTH')
        if position_side not in ('BOTH', 'LONG'):
            raise ValueError('INVALID_POSITION_SIDE')
        entry, qty = D(p['entryPrice']), D(p['positionAmt'])
        r = self.state.get(symbol)
        fingerprint = str(entry.normalize())+':'+str(qty.normalize())
        if r and r['phase'] == 'UNKNOWN':
            self.verify(symbol, r)
            return
        orders = self.b.closing_orders(symbol)  # BOTH normal and conditional required
        coverage = tp_coverage(orders, symbol, entry, qty, position_side)
        if coverage == 'full':
            return
        if coverage == 'partial':
            raise ValueError('PARTIAL_TP')
        if coverage == 'conflict':
            raise ValueError('ORDER_CONFLICT')
        if r and not r.get('absent') and (r['fingerprint'] != fingerprint or r.get('position_side', 'BOTH') != position_side):
            raise ValueError('POSITION_CHANGED')
        if not r or r.get('absent'):
            version = (r or {}).get('version', 0)+1
            r = dict(position_side=position_side, fingerprint=fingerprint, version=version, entry=str(entry), qty=str(qty), phase='READY', attempts=0, retry_at=0)
            self.state[symbol] = r
            self.e.save()
        if r['phase'] == 'PROTECTED':
            self.verify(symbol, r)
            self.warn(symbol, 'TP_NOT_IN_OPEN_ORDERS', 'Pozisyon açık ancak daha önce doğrulanan TP açık emirlerde bulunamadı; tekrar emir gönderilmedi.', r)
            return
        if r['phase'] == 'HOLD' or r['attempts'] >= 2 or self.e.clock() < r['retry_at']:
            return
        qty_text, target = repair_plan(p, self.b.info(symbol))
        r['tp'] = target
        # Refresh position AND both order sets immediately before submission.
        latest = next((x for x in self.b.positions() if x['symbol']==symbol and x.get('positionSide', 'BOTH') == position_side and D(x['positionAmt'])>0), None)
        if latest is None:
            return
        if D(latest['entryPrice']) != entry or D(latest['positionAmt']) != qty:
            raise ValueError('POSITION_CHANGED')
        if tp_coverage(self.b.closing_orders(symbol), symbol, entry, qty, position_side) != 'none':
            return
        if self.b.best_bid(symbol) >= D(target):
            raise ValueError('TARGET_PASSED')
        r['attempts'] += 1
        digest = hashlib.sha256(f'{symbol}:{position_side}:{fingerprint}:{r["version"]}:{r["attempts"]}'.encode()).hexdigest()[:26]
        r.update(client_id='cbm'+digest, phase='UNKNOWN', retry_at=self.e.clock()+60)
        self.e.save()  # No POST unless durable intent was written.
        try:
            kwargs = {'position_side': 'LONG'} if position_side == 'LONG' else {}
            self.b.repair_tp(symbol, qty_text, target, r['client_id'], **kwargs)
        except Exception as error:
            r['submit_error'] = error_details(error)
            if (isinstance(error, BrokerError) and not error.uncertain
                    and error.code in {'-1021', '-1022', '-2014', '-2015', '-2019', '-2022', '-1111',
                                       '-1013', '-1015', '-1003', '-4014', '-4023', '-4061', '-4164', '-5022'}):
                r['phase'] = 'REJECTED'  # one later retry after fresh checks, maximum two attempts
                self.e.save()
                self.fail(symbol, error, r)
                return
            self.e.save()
        self.verify(symbol, r)
