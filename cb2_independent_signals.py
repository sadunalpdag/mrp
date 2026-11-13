"""Independent CoinBeacon API2 six-strategy signal observation (PAPER_ONLY).

Sources: CB2 /public/setups, trend, squeeze, volume-surge, liquidation
and occasional OI snapshot. No HBREAK dependency and no Binance REST.
Watchlist records are not simulated trades.
"""
import json
import time
from datetime import datetime, timezone

import cb2_signal_research as research

SETUPS_PARAMS = {
    "exchange": "binance-futures", "timeframe": "15m",
    "archetype": "breakout_watch,pattern_breakout,trendline_break,breakout_retest",
    "bias": "bullish", "minScore": "50",
}
FIREDS = {"pattern_breakout", "trendline_break", "breakout_retest"}
NAMES = ("rvol", "retest", "trend", "squeeze", "volume", "oi_squeeze")
MAX_ROWS_PER_SCAN = 100
MAX_NEW_PER_STRATEGY_PER_CYCLE = 20


def _num(x):
    try:
        v = float(x)
        return v if abs(v) < float("inf") else None
    except (TypeError, ValueError):
        return None


def _recent(db, family, symbol, since_ms, status=None):
    # At most the most recent 2500 observations; SQLite is isolated from trades.
    for (raw,) in db.execute(
        "SELECT payload FROM signal_observations ORDER BY rowid DESC LIMIT 2500"
    ):
        record = json.loads(raw)
        if record.get("family") != family or record.get("symbol") != symbol:
            continue
        if status and record.get("status") != status:
            continue
        try:
            if research._ms(record["observed_at"]) >= since_ms:
                return record
        except (KeyError, ValueError, TypeError):
            continue
    return None


def _record_watch(db, kind, symbol, now_iso, price=None, **extras):
    now_ms = research._ms(now_iso)
    if _recent(db, kind + "_watch", symbol, now_ms - 4 * 3600000,
               status="WATCH"):
        return False
    return research._insert(
        db, symbol=symbol, family=kind + "_watch",
        source_id=symbol + ":" + str(now_ms // (4 * 3600000)),
        seen_at=now_iso, reference_price=price,
        status="WATCH", extras={
            "source": "coinbeacon-api2", "signal_quality": "SCREEN_WATCH_ONLY",
            "strategy": kind, **extras,
        }
    )


def watch_screener(db, name, board, now_iso):
    if not isinstance(board, dict) or (board.get("meta") or {}).get("stale"):
        return 0
    new = 0
    for row in (board.get("rows") or [])[:MAX_ROWS_PER_SCAN]:
        if not isinstance(row, dict):
            continue
        # Screener may select a spot or other venue: require Binance perpetuals.
        if row.get("venue") != "binance-futures":
            continue
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol.endswith("USDT"):
            continue
        if _record_watch(
            db, name, symbol, now_iso, price=row.get("price"),
            preset=name, source_venue=row.get("venue"),
        ):
            new += 1
            if new >= MAX_NEW_PER_STRATEGY_PER_CYCLE:
                break
    return new


def _fresh_setups(setups, now_iso):
    now_ms = research._ms(now_iso)
    out = []
    if not isinstance(setups, dict):
        return out
    for s in (setups.get("setups") or [])[:150]:
        if not isinstance(s, dict):
            continue
        arch = s.get("archetype") or {}
        if not isinstance(arch, dict):
            continue
        symbol = s.get("symbol")
        if not (isinstance(symbol, str) and symbol.endswith("USDT")):
            continue
        if s.get("tf") != "15m" or s.get("bias") != "bullish":
            continue
        quality = _num(s.get("sqs"))
        if quality is None or quality < 50:
            continue
        comp = _num(s.get("computedAt"))
        if comp is None or now_ms - comp < -10 * 60000 or now_ms - comp > 50 * 60000:
            continue
        age = arch.get("barsSinceTrigger")
        if age is None:
            age = 0
        try:
            age = int(age)
        except (TypeError, ValueError):
            continue
        if age < 0 or age > 2:
            continue
        kind = arch.get("key")
        if kind not in FIREDS:
            continue
        event_ms = int(comp) - age * 900000
        # First observation is the polling time, never claim an earlier live entry.
        out.append((s, kind, event_ms))
    return out


def capture_setups(db, setups, now_iso, trend_data):
    """Each strategy is evaluated from its own CoinBeacon feed, not HBREAK."""
    matrix = {
        x.get("symbol"): x.get("scores") or {}
        for x in ((trend_data or {}).get("items") or [])
        if isinstance(x, dict)
    }
    fresh = _fresh_setups(setups, now_iso)
    counts = {k: 0 for k in NAMES}
    now_ms = research._ms(now_iso)
    for s, kind, event_ms in fresh:
        sym = s["symbol"]
        arch = s["archetype"]
        part = s.get("participation") or {}
        rvol = _num(part.get("rvol"))
        scores = matrix.get(sym) or {}
        trend_1h = _num((scores.get("1h") or {}).get("score"))
        trend_4h = _num((scores.get("4h") or {}).get("score"))
        event_id = sym + ":" + kind + ":" + str(event_ms)
        candidates = []
        if rvol is not None and rvol >= 1.5:
            candidates.append(("rvol", "BREAKOUT_RVOL_1_5"))
        if kind == "breakout_retest":
            candidates.append(("retest", "BREAKOUT_RETEST"))
        if (trend_1h is not None and trend_1h > 0 and
                trend_4h is not None and trend_4h > 0):
            candidates.append(("trend", "BREAKOUT_TREND_1H_4H"))
        if rvol is not None and rvol >= 2.0:
            candidates.append(("volume", "BREAKOUT_RVOL_2"))
        # A squeeze must have been on the watchlist BEFORE the breakout.
        sw = _recent(db, "squeeze_watch", sym, now_ms - 12 * 3600000,
                     status="WATCH")
        if (sw and research._ms(sw["observed_at"]) < now_ms - 900000
                and event_ms >= research._ms(sw["observed_at"])):
            candidates.append(("squeeze", "WATCH_THEN_BREAKOUT"))
        for strategy, signal_quality in candidates:
            if counts[strategy] >= MAX_NEW_PER_STRATEGY_PER_CYCLE:
                continue
            created = research._insert(
                db, symbol=sym, family=strategy, source_id=event_id,
                seen_at=now_iso, reference_price=s.get("price"),
                source_signal_at=research._iso(event_ms),
                archetype=kind, sqs=s.get("sqs"), rvol=rvol,
                extras={
                    "signal_quality": signal_quality, "source": "coinbeacon-api2/setups",
                    "source_venue": "binance-futures",
                    "trend_1h": trend_1h, "trend_4h": trend_4h,
                    "multi_tf_confirmed": s.get("multiTfConfirmed"),
                    "first_seen_is_not_entry": True,
                },
            )
            counts[strategy] += int(created)
    return counts, len(fresh)


def capture_liquidations(db, board, now_iso, market, breakout_setups):
    """Require fresh short liquidations, a fresh bullish setup and falling OI.

    If OI is unavailable, save WATCH only. Never infer OI from price.
    """
    if not isinstance(board, dict) or board.get("stale"):
        return {"oi_squeeze": 0, "oi_watch": 0}
    now = research._ms(now_iso)
    shorts = {}
    for row in board.get("tape") or []:
        if (isinstance(row, dict) and row.get("side") == "short" and
                now - (_num(row.get("ts")) or 0) <= 30 * 60000):
            symbol = row.get("symbol")
            if isinstance(symbol, str) and symbol.endswith("USDT"):
                shorts[symbol] = shorts.get(symbol, 0) + (_num(row.get("notionalUsd")) or 0)
    for row in (board.get("cascades") or {}).get("active") or []:
        if (isinstance(row, dict) and row.get("side") == "short" and
                now - (_num(row.get("lastEventAt")) or 0) <= 30 * 60000):
            symbol = row.get("symbol")
            if isinstance(symbol, str) and symbol.endswith("USDT"):
                shorts[symbol] = max(shorts.get(symbol, 0), _num(row.get("shortUsd")) or 0)
    candidates = [(sym, amount) for sym,amount in shorts.items() if amount >= 25000]
    candidates.sort(key=lambda x: -x[1])
    result = {"oi_squeeze": 0, "oi_watch": 0}
    setups_for_symbol = {row["symbol"]: (row, kind, when)
                         for row,kind,when in _fresh_setups(breakout_setups, now_iso)}
    for sym,amount in candidates[:8]:
        if _record_watch(db, "oi_squeeze", sym, now_iso, short_liq_usd=amount,
                         signal_quality="SHORT_LIQ_ONLY_OI_PENDING"):
            result["oi_watch"] += 1
        if sym not in setups_for_symbol:
            continue
        if result["oi_squeeze"] >= 1:  # max one extra OI request in a cycle
            break
        # Reuse existing CB2 10-credit budget and the one-per-hour limit.
        used = int(db.execute(
            "SELECT v FROM meta WHERE k=?",
            ("oi_independent_day_" + datetime.now(timezone.utc).date().isoformat(),)
        ).fetchone()[0]) if db.execute(
            "SELECT 1 FROM meta WHERE k=?",
            ("oi_independent_day_" + datetime.now(timezone.utc).date().isoformat(),)
        ).fetchone() else 0
        if used >= 24:
            break
        next_due = db.execute("SELECT v FROM meta WHERE k='next_oi_independent'").fetchone()
        if next_due and time.time() < float(next_due[0]):
            break
        data = market.cb("/public/open-interest/" + sym + "/history", 10,
                         {"period": "15m"})
        if not isinstance(data, dict):
            break
        db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('next_oi_independent',?)",
                   (str(time.time()+3600),))
        db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)",
                   ("oi_independent_day_" + datetime.now(timezone.utc).date().isoformat(),
                    str(used+1)))
        db.commit()
        points = data.get("points") or []
        if data.get("symbol") != sym or len(points) < 5:
            break
        pts = sorted(points, key=lambda p: _num(p.get("time")) or 0)
        if now - (_num(pts[-1].get("time")) or 0)*1000 > 3600000:
            break
        begin = _num(pts[-5].get("oi"))
        end = _num(pts[-1].get("oi"))
        if begin is None or begin <= 0 or end is None:
            break
        change = 100*(end/begin-1)
        if change > -0.2:
            break
        item,kind,event_ms = setups_for_symbol[sym]
        created = research._insert(
            db, symbol=sym, family="oi_squeeze",
            source_id=sym + ":" + kind + ":" + str(event_ms),
            seen_at=now_iso, reference_price=item.get("price"),
            source_signal_at=research._iso(event_ms),
            archetype=kind, sqs=item.get("sqs"),
            rvol=(item.get("participation") or {}).get("rvol"),
            extras={"source": "coinbeacon-api2/liquidations+setups+oi",
                    "signal_quality": "SHORT_LIQ_AND_FALLING_OI_CONFIRMED",
                    "oi_change_pct": round(change, 3),
                    "short_liq_usd": round(amount, 2),
                    "source_venue": "binance-futures"},
        )
        result["oi_squeeze"] += int(created)
    return result


def expire_watches(db, now_iso):
    now = research._ms(now_iso)
    changed = 0
    for sid,raw in db.execute("SELECT id,payload FROM signal_observations WHERE payload LIKE '%\"WATCH\"%'").fetchall():
        entry = json.loads(raw)
        if entry.get("status") != "WATCH":
            continue
        if now - research._ms(entry["observed_at"]) < 12*3600000:
            continue
        entry["status"] = "WATCH_EXPIRED"
        db.execute("UPDATE signal_observations SET payload=? WHERE id=?",
                   (json.dumps(entry, ensure_ascii=False),sid))
        changed += 1
    if changed:
        db.commit()
    return changed
