"""Read-only diagnostics for Crash Bounce candle validation."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation


def candle_diagnostic(rows, now, last_price, tick_size):
    closed = []
    malformed = 0
    for row in rows:
        try:
            if len(row) <= 6:
                malformed += 1
            elif int(row[6]) < now * 1000:
                closed.append(row)
        except (TypeError, ValueError):
            malformed += 1
    recent = closed[-16:]
    reasons = []
    if len(closed) < 16:
        reasons.append('EKSIK_MUM')
    last = 'yok'
    age = None
    gaps = 0
    if recent:
        stamp = int(recent[-1][6]) / 1000
        age = round(now - stamp, 1)
        last = datetime.fromtimestamp(stamp, timezone.utc).isoformat()
        if age > 1800:
            reasons.append('ESKI_MUM')
        try:
            gaps = sum(int(b[0])-int(a[0]) != 900000 for a,b in zip(recent,recent[1:]))
            if gaps:
                reasons.append('MUM_ZAMAN_BOSLUGU')
            values = [Decimal(str(r[3])) for r in recent]
            values += [Decimal(str(last_price)), Decimal(str(tick_size))]
            if any(not v.is_finite() or v <= 0 for v in values):
                reasons.append('GECERSIZ_FIYAT_VEYA_TICK')
        except (IndexError, TypeError, ValueError, InvalidOperation):
            malformed += 1
    if malformed:
        reasons.append('BOZUK_MUM')
    return (f"neden={','.join(reasons) or 'PLAN_OLUSMADI'}; "
            f"kapali_mum={len(closed)}/16; toplam_mum={len(rows)}; "
            f"son_mum_UTC={last}; mum_yasi_sn={age}; azami_yas_sn=1800; "
            f"zaman_boslugu={gaps}; bozuk_mum={malformed}")
