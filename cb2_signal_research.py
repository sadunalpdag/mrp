"""CB2 signal-only journal and archive-delayed entry comparisons.

No exchange REST, no orders, no notification side effects.
New observations are persisted separately from paper trades.
"""
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOG = logging.getLogger("cb2_signal_research")
MAX_SIGNALS = 2000
MAX_CHECK_PER_CYCLE = 12
FIFTEEN = 900000


def _ms(iso):
    return int(datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp() * 1000)


def _iso(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat(timespec="seconds")


def _as_number(v):
    try:
        x = float(v)
        if x > 0 and x < float("inf"):
            return x
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def initialize(db):
    db.execute("CREATE TABLE IF NOT EXISTS signal_observations (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.commit()


def _id(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def _insert(db, *, symbol, family, source_id, seen_at, reference_price=None,
            source_signal_at=None, archetype=None, sqs=None, rvol=None,
            confidence=None):
    if not isinstance(symbol, str) or not symbol.endswith("USDT"):
        return False
    sid = _id(family, source_id)
    if db.execute("SELECT 1 FROM signal_observations WHERE id=?", (sid,)).fetchone():
        return False
    ref = _as_number(reference_price)
    obj = {
        "id": sid, "family": family, "source_id": str(source_id),
        "symbol": symbol, "observed_at": seen_at,
        "source_signal_at": source_signal_at,
        "archetype": archetype, "sqs": sqs, "rvol": rvol,
        "reference_price": ref, "reference_status": (
            "API_SNAPSHOT_NOT_VERIFIED_FILL" if ref else "MISSING_PRICE"),
        "status": "WAITING_ARCHIVE", "entry_scenarios": {},
        "last_check": None, "note": "Signal observation only; no order placed",
    }
    db.execute("INSERT INTO signal_observations VALUES (?, ?)",
               (sid, json.dumps(obj, ensure_ascii=False, allow_nan=False)))
    db.commit()
    return True


def capture_hbreak(db, source_rows, now_iso):
    """Capture observed HBREAK triggers even when CB2's 20m fill gate rejects them."""
    added = 0
    now = _ms(now_iso)
    for row in source_rows:
        source_id = row.get("id")
        when = row.get("opened_at")
        if not source_id or not when:
            continue
        try:
            age = now - _ms(when)
        except (ValueError, TypeError):
            continue
        # Existing DB contains older historical signals; never relabel them as
        # a newly observed real-time signal when the journal is first enabled.
        if not (0 <= age <= 90 * 60 * 1000):
            continue
        added += int(_insert(
            db, symbol=row.get("coin"), family="hbreak_breakout",
            source_id=source_id, seen_at=when,
            source_signal_at=row.get("triggered_at") or when,
            reference_price=row.get("entry"),
            archetype=row.get("archetype"), sqs=row.get("sqs"),
            rvol=row.get("rvol"),
        ))
    return added


def capture_board(db, name, result, now_iso):
    """Journal the existing paid scanner response; never call an extra API."""
    if not isinstance(result, dict) or (result.get("meta") or {}).get("stale"):
        return 0
    added = 0
    # Scanner rows can be watchlists, not confirmed breakouts. Preserve labels.
    for row in result.get("rows") or []:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if row.get("quote") != "USDT" or not isinstance(symbol, str):
            continue
        slot = _ms(now_iso) // FIFTEEN
        price = row.get("price") or row.get("lastPrice")
        added += int(_insert(
            db, symbol=symbol, family=name,
            source_id=f"{symbol}:{slot}", seen_at=now_iso,
            reference_price=price,
            archetype=str(row.get("archetype") or name),
            sqs=row.get("sqs"), rvol=row.get("rvol"),
        ))
    return added


def _scenario(candles, trigger_ms, delay_minutes, reference_price=None, pullback=None):
    """15m OHLC simulation, with no intrabar timing claims.

    Delay entries use the NEXT bar open after the requested waiting period.
    Pullback uses a limit at the specified reference discount.
    """
    earliest = trigger_ms + delay_minutes * 60000
    eligible = [c for c in candles if int(c[0]) >= earliest]
    if not eligible:
        return {"status": "PENDING_BARS"}
    if pullback is not None:
        if reference_price is None:
            return {"status": "NO_REFERENCE_PRICE"}
        price = reference_price * (1 - pullback)
        found = next((i for i,c in enumerate(eligible)
                      if float(c[3]) <= price), None)
        if found is None:
            return {"status": "NOT_FILLED"}
        entry_index = found
        entry_ms = int(eligible[entry_index][0])
        entry_mode = "LIMIT_OHLC_APPROX"
    else:
        price = float(eligible[0][1])
        entry_ms = int(eligible[0][0])
        entry_index = 0
        entry_mode = "NEXT_15M_OPEN_APPROX"

    target = price * 1.01
    minimum = price
    tp_ms = None
    ambiguous = False
    for i,c in enumerate(eligible[entry_index:], start=entry_index):
        low = float(c[3])
        high = float(c[2])
        minimum = min(minimum, low)
        if high >= target:
            # Same-candle fill -> TP ordering cannot be established from OHLC.
            if i == entry_index or low < price:
                ambiguous = True
            tp_ms = int(c[6])
            break
    return {
        "status": "TP_1_PERCENT" if tp_ms else "OPEN_IN_OBSERVED_WINDOW",
        "entry_price": round(price, 12), "entry_at": _iso(entry_ms),
        "entry_method": entry_mode,
        "tp_at_candle_close": _iso(tp_ms) if tp_ms else None,
        "tp_minutes_upper_bound": round((tp_ms-entry_ms)/60000, 2) if tp_ms else None,
        "max_drawdown_pct_observed": round(min(0, (minimum/price-1)*100), 4),
        "intrabar_order_uncertain": ambiguous,
    }


def analyze(db, archive_klines, http, now_iso):
    """Only archived 15m candles; never use present-day unavailable candles."""
    now = _ms(now_iso)
    records = db.execute("SELECT id,payload FROM signal_observations ORDER BY rowid DESC LIMIT ?",
                         (MAX_SIGNALS,)).fetchall()
    count = 0
    for sid, raw in sorted(records, key=lambda x: json.loads(x[1])["observed_at"]):
        if count >= MAX_CHECK_PER_CYCLE:
            break
        record = json.loads(raw)
        if record.get("status") == "COMPLETE_48H":
            continue
        seen_ms = _ms(record["observed_at"])
        if now - seen_ms < 36 * 3600000:
            continue
        if record.get("last_check") and now - _ms(record["last_check"]) < 2 * 3600000:
            continue
        # Only include complete bars after signal. A missing middle archive day
        # causes archive_klines to return an incomplete prefix.
        bars = archive_klines(record["symbol"], "15m",
                              start=(seen_ms // FIFTEEN) * FIFTEEN,
                              limit=400, http=http)
        count += 1
        record["last_check"] = now_iso
        bars = [c for c in bars if int(c[0]) > seen_ms]
        horizon = seen_ms + 48 * 3600000
        available = [c for c in bars if int(c[0]) < horizon]
        if not available:
            record["status"] = "PENDING_ARCHIVE"
        else:
            complete = int(available[-1][6]) >= horizon - 1
            record["status"] = "COMPLETE_48H" if complete else "PARTIAL_ARCHIVE"
            ref = record.get("reference_price")
            record["entry_scenarios"] = {
                "next_15m_open": _scenario(available, seen_ms, 0),
                "wait_15m": _scenario(available, seen_ms, 15),
                "wait_30m": _scenario(available, seen_ms, 30),
                "pullback_0_5pct": _scenario(available, seen_ms, 0, ref, .005),
                "pullback_1pct": _scenario(available, seen_ms, 0, ref, .01),
            }
            record["missing_5m"] = "5m entry timing not evaluated: only 15m daily archive is used"
            record["simulation_warning"] = (
                "OHLC approximation; TP/low order and limit fills are not guaranteed")
        db.execute("UPDATE signal_observations SET payload=? WHERE id=?",
                   (json.dumps(record, ensure_ascii=False, allow_nan=False), sid))
    db.commit()
    if count:
        LOG.info("[CB2 SIGNAL JOURNAL] checked=%d", count)
    return count


def export(db):
    records = db.execute(
        "SELECT payload FROM signal_observations ORDER BY rowid DESC LIMIT ?",
        (MAX_SIGNALS,)).fetchall()
    data = [json.loads(x[0]) for x in records]
    return {
        "mode": "OBSERVATION_ONLY", "count": len(data),
        "status_counts": {
            k: sum(x["status"] == k for x in data)
            for k in sorted(set(x["status"] for x in data))
        },
        "signals": data,
    }
