"""Deterministic Crash Bounce V2 virtual transitions from Binance Futures aggTrades.

Trade prints are observations, not proof a limit order could have filled.
An ordered entry must precede an ordered TP; this module never places orders.
"""
import math
from crash_policy import TRACKING_SECONDS

MODEL = 'ws_aggtrade_upcross_v3'
TERMINAL = {'TP', 'TIMEOUT', 'NO_ENTRY_OBSERVED', 'INVALIDATED_BEFORE_ENTRY',
            'V2_FILTERED', 'MISSED_SIGNAL'}


def advance_trade(record, price, observed_at, trade_id):
    """Update a V2 record using one ordered trade print; return transition or None."""
    if record.get('tracking_model') != MODEL or record.get('tracking_status') in TERMINAL:
        return None
    try:
        price = float(price)
        observed_at = float(observed_at)
        trade_id = int(trade_id)
        entry = float(record['entry'])
        target = float(record['tp'])
        bottom = float(record['reference_low'])
        created = float(record['created_at'])
        since = float(record.get('entered_at', created)) if record.get('tracking_status') == 'OPEN' else created
    except (TypeError, ValueError, KeyError, OverflowError):
        return None
    if (not all(math.isfinite(x) for x in (price, observed_at, entry, target, bottom, created, since))
            or price <= 0 or observed_at <= created or observed_at > since + TRACKING_SECONDS):
        return None
    previous_id = record.get('ws_last_trade_id')
    if previous_id is not None and trade_id <= previous_id:
        return None
    previous_at = float(record.get('ws_last_event_at', created))
    gap = observed_at - previous_at
    if gap < 0:
        return None
    prior_price = float(record.get('previous_watch_price', record.get('last_price', price)))
    record['ws_last_trade_id'] = trade_id
    record['ws_last_event_at'] = observed_at
    record['previous_watch_price'] = price
    record['last_observed_at'] = observed_at
    record['last_observed_price'] = price
    record['samples'] = record.get('samples', 0) + 1
    record['max_gap_seconds'] = max(float(record.get('max_gap_seconds', 0)), gap)
    if gap > 30:
        record['coverage_gaps'] = record.get('coverage_gaps', 0) + 1
    status = record.get('tracking_status', 'WAIT_ENTRY')
    if record.get('derivatives_shadow') is not None:
        record['shadow_new_low_observed'] = record.get('shadow_new_low_observed', False) or price < bottom
        if status == 'OPEN':
            record['shadow_mae_pct'] = min(record.get('shadow_mae_pct', 0.0), 100*(price/entry-1))
    # Gaps cannot prove which threshold crossed first.
    reliable = gap <= 30 and (previous_id is not None or gap <= 10)
    if status == 'WAIT_ENTRY':
        if price < bottom:
            record.update(tracking_status='INVALIDATED_BEFORE_ENTRY', finished_at=observed_at,
                          invalidation_price=price, result_source='WS_AGGTRADE')
            return 'INVALIDATED_BEFORE_ENTRY'
        if not reliable:
            if prior_price < entry and price >= target:
                record.update(tracking_status='MISSED_SIGNAL', finished_at=observed_at,
                              v2_reason='ENTRY_ABOVE_TP', missing_sequence=True,
                              missed_at=observed_at, missed_price=price,
                              result_source='WS_AGGTRADE')
                return 'MISSED_SIGNAL'
            return None
        if prior_price < entry <= price:
            if price >= target:
                record.update(tracking_status='MISSED_SIGNAL', finished_at=observed_at,
                              v2_reason='ENTRY_ABOVE_TP', missed_at=observed_at,
                              missed_price=price, result_source='WS_AGGTRADE')
                return 'MISSED_SIGNAL'
            record.update(tracking_status='OPEN', entered_at=observed_at,
                          virtual_fill_price=entry, observed_entry_price=price,
                          entry_trade_id=trade_id, entry_source='WS_AGGTRADE',
                          max_favorable_pct=max(0.0, (price/entry-1)*100),
                          max_adverse_pct=min(0.0, (price/entry-1)*100))
            return 'OPEN'
    if status == 'OPEN':
        ret = (price / entry - 1)*100
        record['max_favorable_pct'] = max(record.get('max_favorable_pct', 0), ret)
        record['max_adverse_pct'] = min(record.get('max_adverse_pct', 0), ret)
        if (observed_at > float(record.get('entered_at', observed_at)) and
                trade_id > int(record.get('entry_trade_id', trade_id)) and price >= target):
            record.update(tracking_status='TP', finished_at=observed_at,
                          observed_return_pct=(target/entry-1)*100,
                          tp_hours=(observed_at-record['entered_at'])/3600,
                          tp_trade_id=trade_id, result_source='WS_AGGTRADE')
            return 'TP'
    return None

