"""CoinBeacon key-level snapshot for the isolated BTC paper observer.

Reads ONLY the documented public levels detector, never any trade endpoint.
The detector is a capped board, NOT a symbol-specific API. Missing BTC is
reported, never interpreted as a valid zero-level reading.
"""
import logging
import math
import os

import requests

LOG = logging.getLogger("btc_coinbeacon_levels")
HOUR_MS = 3_600_000
TIMEFRAMES = ("1h", "4h")
FRAME_MS = {"1h": HOUR_MS, "4h": 4 * HOUR_MS}
ENDPOINT = "https://api.coinbeacon.io/public/detectors/levels"


def _num(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _time(value):
    v = _num(value)
    return int(v) if v is not None and v > 1_000_000_000_000 else None


def _touches(prices, level):
    last = -10
    count = 0
    for index, price in enumerate(prices):
        if abs(price / level - 1) <= .0025 and index - last >= 4:
            count += 1
            last = index
    return count


def normalize(data, hour_ms, reference_price, timeframe):
    """Return only BTCUSDT Binance-Futures levels known at the observation time."""
    if not isinstance(data, dict) or data.get("kind") != "levels":
        raise ValueError("COINBEACON_LEVELS_WRONG_PAYLOAD")
    if data.get("exchange") != "binance-futures" or data.get("timeframe") != timeframe:
        raise ValueError("COINBEACON_LEVELS_WRONG_MARKET")
    rows = data.get("items")
    if not isinstance(rows, list):
        raise ValueError("COINBEACON_LEVELS_MISSING_ITEMS")
    levels = []
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") != "BTCUSDT":
            continue
        if row.get("exchange") != "binance-futures" or row.get("tf") != timeframe:
            continue
        item = row.get("item")
        if not isinstance(item, dict):
            continue
        computed = _time(row.get("computedAt"))
        available = _time(item.get("availableAt", row.get("availableAt")))
        if computed is None or computed > hour_ms or hour_ms - computed > 2 * FRAME_MS[timeframe]:
            continue
        if available is not None and available > hour_ms:
            continue
        period = item.get("period")
        if isinstance(period, dict):
            valid_from = _time(period.get("validFrom"))
            valid_to = _time(period.get("validTo"))
            if valid_from is not None and hour_ms < valid_from:
                continue
            if valid_to is not None and hour_ms >= valid_to:
                continue
        snapshot_price = _num(row.get("price"))
        # Reject levels from a stale or incompatible BTC market snapshot.
        if snapshot_price is None or abs(snapshot_price / reference_price - 1) > .03:
            continue
        price = _num(item.get("price"))
        if price is None or price <= 0 or not .95 * reference_price <= price <= 1.05 * reference_price:
            continue
        levels.append({
            "price": round(price, 2),
            "label": str(item.get("label") or "UNKNOWN")[:55],
            "type": str(item.get("type") or "UNKNOWN")[:25],
            "timeframe": timeframe,
            "computed_at_ms": computed,
            "available_at_ms": available,
            # CoinBeacon's 'kind' is relative to its snapshot, NOT a fixed
            # support/resistance role. We reclassify relative to closed BTC.
        })
    return levels


def fetch_levels(hour_ms, reference_price, priority_checker):
    key = (os.getenv("api2") or os.getenv("API2") or os.getenv("COINBEACON_API_KEY") or "").strip()
    if not key:
        return {"status": "NO_API_KEY", "levels": [], "frames": {}}
    levels = []
    frames = {}
    for tf in TIMEFRAMES:
        if priority_checker():
            return {"status": "LIVE_CRASH_PRIORITY", "levels": [], "frames": frames}
        try:
            response = requests.get(
                ENDPOINT,
                params={"exchange": "binance-futures", "timeframe": tf, "quote": "USDT"},
                headers={"X-API-Key": key}, timeout=15)
            if response.status_code == 429:
                frames[tf] = "RATE_LIMIT"
                return {"status": "RATE_LIMIT", "levels": [], "frames": frames}
            response.raise_for_status()
            payload = response.json()
            found = normalize(payload, hour_ms, reference_price, tf)
            frames[tf] = "BTC_FOUND" if found else "BTC_MISSING_OR_STALE"
            levels.extend(found)
        except (requests.RequestException, ValueError, TypeError) as exc:
            frames[tf] = "ERROR_" + type(exc).__name__
            LOG.warning("[BTC COINBEACON LEVELS] %s %s", tf, type(exc).__name__)
    return {"status": "LEVELS_AVAILABLE" if levels else "NO_VALID_BTC_LEVELS",
            "levels": levels, "frames": frames}


def evaluate(snapshot, candles, fetched):
    """Prefer a closed-bar-confirmed CoinBeacon pair; otherwise retain Binance model."""
    report = {"status": fetched["status"], "frames": fetched.get("frames", {}),
              "level_count": len(fetched.get("levels", [])),
              "support": None, "resistance": None}
    if fetched["status"] != "LEVELS_AVAILABLE":
        return report
    price = candles[-1][4]
    candidates = fetched["levels"]
    lows = [bar[3] for bar in candles[-96:]]
    highs = [bar[2] for bar in candles[-96:]]
    supports = sorted((lev for lev in candidates if lev["price"] <= price),
                      key=lambda lev: (price - lev["price"], 0 if lev["timeframe"] == "1h" else 1))
    resistances = sorted((lev for lev in candidates if lev["price"] > price),
                         key=lambda lev: (lev["price"] - price, 0 if lev["timeframe"] == "1h" else 1))
    for s in supports:
        if s["price"] <= 0 or (price / s["price"] - 1) > .015:
            continue
        st = _touches(lows, s["price"])
        if st < 2:
            continue
        for r in resistances:
            width = 100 * (r["price"] / s["price"] - 1)
            if not .9 <= width <= 5.0:
                continue
            rt = _touches(highs, r["price"])
            if rt < 2:
                continue
            report.update(status="CONFIRMED_PAIR", support=s["price"],
                          resistance=r["price"], support_level=s,
                          resistance_level=r, width_pct=round(width, 3),
                          support_touches=st, resistance_touches=rt)
            return report
    report["status"] = "NO_REPEATED_PAIR"
    return report


def apply(snapshot, candles, hour_ms, priority_checker):
    """Enrich snapshot and optionally replace range levels with verified CoinBeacon levels."""
    reference_price = candles[-1][4]
    fetched = fetch_levels(hour_ms, reference_price, priority_checker)
    report = evaluate(snapshot, candles, fetched)
    snapshot["binance_levels"] = {
        "support": snapshot.get("support"), "resistance": snapshot.get("resistance"),
        "regime": snapshot.get("regime")}
    snapshot["coinbeacon"] = report
    snapshot["level_source"] = "BINANCE_CLOSED_CANDLES"
    if report["status"] != "CONFIRMED_PAIR" or snapshot.get("regime") != "RANGE":
        return snapshot
    support, resistance = report["support"], report["resistance"]
    snapshot.update(support=support, resistance=resistance, width_pct=report["width_pct"],
                    support_touches=report["support_touches"],
                    resistance_touches=report["resistance_touches"],
                    support_distance_pct=round(100 * (reference_price / support - 1), 3),
                    level_source="COINBEACON_VERIFIED_CLOSED_CANDLES")
    snapshot.pop("candidate", None)  # Old Binance candidate must not survive a source switch.
    proximity = 100 * (reference_price / support - 1)
    if 0 <= proximity <= .30:
        entry, target, invalidation = support * 1.001, resistance * .999, support * .997
        fee = .08
        estimated_net = 100 * (target / entry - 1) - fee
        if entry <= reference_price and estimated_net >= .45 and target > entry > invalidation:
            snapshot["candidate"] = {
                "side": "PAPER_LONG", "entry": round(entry, 2),
                "target": round(target, 2),
                "invalidation": round(invalidation, 2),
                "estimated_net_pct": round(estimated_net, 3),
                "plan": "COINBEACON_SUPPORT_TO_RESISTANCE"}
    return snapshot
