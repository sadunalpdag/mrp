import os
import json
import time
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))

VIRTUAL_FILE = os.path.join(DATA_DIR, "virtual_entry_trades.json")
RETENTION_DAYS = 15
TP_PCT = 0.006


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_virtual_trades():
    if not os.path.exists(VIRTUAL_FILE):
        return []
    try:
        with open(VIRTUAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_virtual_trades(data):
    os.makedirs(os.path.dirname(VIRTUAL_FILE), exist_ok=True)
    with open(VIRTUAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cleanup_old_trades():
    data = load_virtual_trades()
    limit = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)

    cleaned = []
    for t in data:
        try:
            created = datetime.fromisoformat(t["created_at"])
            if created >= limit:
                cleaned.append(t)
        except Exception:
            cleaned.append(t)

    save_virtual_trades(cleaned)


def create_virtual_long(symbol, entry_level, current_price, confidence, reason, setup_data=None):
    data = load_virtual_trades()

    for t in data:
        if t["symbol"] == symbol and t["status"] in ("WAIT_ENTRY", "OPEN"):
            return False

    trade = {
        "id": f"{symbol}_{int(time.time())}",
        "symbol": symbol,
        "direction": "LONG",
        "status": "WAIT_ENTRY",
        "entry_level": float(entry_level),
        "tp_level": round(float(entry_level) * (1 + TP_PCT), 8),
        "current_price_at_signal": float(current_price) if current_price else 0.0,
        "confidence": confidence,
        "reason": reason,
        "created_at": now_iso(),
        "entry_hit_at": None,
        "closed_at": None,
        "close_minutes": None,
        "max_profit_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "setup": setup_data or {},
    }

    data.append(trade)
    save_virtual_trades(data)
    return True


def update_virtual_trades(symbol, last_price):
    data = load_virtual_trades()
    changed = False

    for t in data:
        if t["symbol"] != symbol:
            continue

        price = float(last_price)

        if t["status"] == "WAIT_ENTRY":
            if price <= float(t["entry_level"]):
                t["status"] = "OPEN"
                t["entry_hit_at"] = now_iso()
                changed = True

        elif t["status"] == "OPEN":
            entry = float(t["entry_level"])
            tp = float(t["tp_level"])

            profit_pct = (price - entry) / entry
            t["max_profit_pct"] = max(float(t.get("max_profit_pct", 0)), profit_pct)

            drawdown_pct = (price - entry) / entry
            t["max_drawdown_pct"] = min(float(t.get("max_drawdown_pct", 0)), drawdown_pct)

            if price >= tp:
                t["status"] = "TP_CLOSED"
                t["closed_at"] = now_iso()

                start = datetime.fromisoformat(t["entry_hit_at"])
                end = datetime.fromisoformat(t["closed_at"])
                t["close_minutes"] = round((end - start).total_seconds() / 60, 2)
                changed = True

    if changed:
        save_virtual_trades(data)

    return changed


def analyze_virtual_trades():
    data = load_virtual_trades()
    closed = [t for t in data if t["status"] == "TP_CLOSED"]
    open_trades = [t for t in data if t["status"] in ("WAIT_ENTRY", "OPEN")]

    if not data:
        return {
            "total": 0,
            "message": "Yeterli veri yok",
        }

    avg_close = None
    if closed:
        avg_close = sum(t["close_minutes"] for t in closed) / len(closed)

    return {
        "total": len(data),
        "closed_count": len(closed),
        "open_count": len(open_trades),
        "tp_success_rate": round(len(closed) / len(data) * 100, 2),
        "avg_close_minutes": round(avg_close, 2) if avg_close else None,
        "best_confidence_min": find_best_confidence_threshold(data),
        "recommendation": make_entry_recommendation(data),
    }


def find_best_confidence_threshold(data):
    best = None

    for threshold in range(60, 96, 5):
        filtered = [t for t in data if int(t.get("confidence", 0)) >= threshold]
        if len(filtered) < 5:
            continue

        closed = [t for t in filtered if t["status"] == "TP_CLOSED"]
        rate = len(closed) / len(filtered)

        if best is None or rate > best["success_rate"]:
            best = {
                "confidence": threshold,
                "sample": len(filtered),
                "success_rate": round(rate * 100, 2),
            }

    return best


def make_entry_recommendation(data):
    closed = [t for t in data if t["status"] == "TP_CLOSED"]

    if len(data) < 20:
        return "Henüz karar için az veri var. En az 20 sanal işlem beklenmeli."

    success_rate = len(closed) / len(data)

    if success_rate >= 0.65:
        return "Entry sistemi güçlü. Confirmed LONG sonrası entry level kullanılabilir."
    elif success_rate >= 0.45:
        return "Orta kalite. Sadece confidence yüksekse işlem açılmalı."
    else:
        return "Zayıf. Confirmed LONG tek başına yeterli değil, ekstra filtre gerekli."
