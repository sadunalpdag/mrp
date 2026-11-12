"""Read-only daily high-water profit target PAPER monitor.

Absolutely no Binance order endpoints, no writes to trading state, no broker import.
Real Crash Bounce TP, queued orders and position management remain unchanged.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests

LOG = logging.getLogger("daily_profit_lock_paper")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[PROFIT-PAPER] %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)
LOG.propagate = False

PARIS = ZoneInfo("Europe/Paris")
URL = "https://fapi.binance.com"
STATE_FILE = Path(os.getenv("DATA_DIR", "/data")) / "daily_profit_lock_paper.json"
INTERVAL = max(120, int(os.getenv("DAILY_PROFIT_PAPER_INTERVAL_SEC", "180")))
PCT = Decimal(os.getenv("DAILY_PROFIT_PAPER_TARGET_PCT", "5"))
if not (Decimal("0.1") <= PCT <= Decimal("100")):
    raise ValueError("DAILY_PROFIT_PAPER_TARGET_OUT_OF_RANGE")


def utc():
    return datetime.now(timezone.utc)


def money(x):
    try:
        d = Decimal(str(x))
        if not d.is_finite():
            raise ValueError("NON_FINITE")
        return d
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("INVALID_DECIMAL") from exc


def persist(st):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf8") as file:
        json.dump(st, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, STATE_FILE)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf8") as file:
            result = json.load(file)
        if result.get("mode") != "PAPER_ONLY":
            raise ValueError("STATE_MODE_NOT_PAPER")
        return result
    return None


class RateWait(Exception):
    def __init__(self, delay):
        super().__init__("READ_ONLY_API_BACKOFF")
        self.delay = min(3600, max(60, int(delay)))


class MarketReader:
    """Only signed and unsigned GET. Never accept a method parameter."""
    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.key = os.getenv("BINANCE_API_KEY", "").strip()
        self.secret = os.getenv("BINANCE_SECRET_KEY", "").strip()

    def signed_get(self, path, params=None):
        if not self.key or not self.secret:
            raise ValueError("BINANCE_READ_CREDENTIALS_UNAVAILABLE")
        if path not in ("/fapi/v3/account", "/fapi/v3/positionRisk"):
            raise ValueError("READ_ONLY_ENDPOINT_NOT_ALLOWLISTED")
        q = dict(params or {})
        q["timestamp"] = int(time.time() * 1000)
        q["recvWindow"] = 5000
        query = urlencode(q)
        signature = hmac.new(self.secret.encode("utf8"), query.encode("utf8"),
                             hashlib.sha256).hexdigest()
        response = self.session.get(
            URL + path + "?" + query + "&signature=" + signature,
            headers={"X-MBX-APIKEY": self.key}, timeout=12)
        if response.status_code in (418, 429):
            raise RateWait(response.headers.get("Retry-After", "1800"))
        response.raise_for_status()
        return response.json()

    def best_book(self, symbol):
        if not isinstance(symbol, str) or not symbol.endswith("USDT") or not symbol.isalnum():
            return None
        r = self.session.get(
            URL + "/fapi/v1/ticker/bookTicker", params={"symbol": symbol}, timeout=10)
        if r.status_code in (418, 429):
            raise RateWait(r.headers.get("Retry-After", "1800"))
        r.raise_for_status()
        raw = r.json()
        return {"bid": money(raw["bidPrice"]), "ask": money(raw["askPrice"])}

    def equity(self):
        account = self.signed_get("/fapi/v3/account")
        wallet = money(account["totalWalletBalance"])
        unrealized = money(account["totalUnrealizedProfit"])
        equity = wallet + unrealized
        if equity <= 0:
            raise ValueError("NON_POSITIVE_ACCOUNT_EQUITY")
        return equity, wallet, unrealized

    def positions(self):
        raw = self.signed_get("/fapi/v3/positionRisk")
        if not isinstance(raw, list):
            raise ValueError("POSITION_SNAPSHOT_INVALID")
        rows = []
        for p in raw:
            amt = money(p.get("positionAmt", "0"))
            if amt == 0:
                continue
            symbol = p.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            mark = money(p.get("markPrice"))
            if mark <= 0:
                raise ValueError("MISSING_MARK_PRICE")
            side = "LONG" if amt > 0 else "SHORT"
            rows.append({
                "symbol": symbol,
                "positionSide": p.get("positionSide", "BOTH"),
                "side": side,
                "qty": str(abs(amt)),
                "snapshot_price": str(mark),
                "hypothetical_close": "SELL LIMIT" if side == "LONG" else "BUY LIMIT",
                "paper_status": "WAIT",
                "paper_fill_at": None,
            })
        return rows


def maybe_trigger(st, equity, positions_fn, observed):
    """Pure target transition; positions_fn is invoked once only after crossing."""
    if not st.get("target") or st.get("active_lock") or st.get("triggered_today"):
        return False
    if equity < money(st["target"]):
        return False
    positions = positions_fn()
    frozen = {
        "triggered_at": observed.isoformat(timespec="seconds"),
        "trigger_equity": str(equity),
        "target": st["target"],
        "day": st["paris_day"],
        "positions": positions,
        "paper_only": True,
        "paper_note": "Book quote crossing simulates a possible fill; no order sent",
    }
    st["snapshots"].append(frozen)
    st["active_lock"] = frozen
    st["triggered_today"] = True
    return True


def roll_day(st, day):
    if day == st["paris_day"]:
        return False
    # Previous day's highest *ever observed* equity becomes today's reference.
    peak = money(st["all_time_peak"])
    st["paris_day"] = day
    st["reference_peak"] = str(peak)
    st["target"] = str(peak * (Decimal("1") + PCT / 100))
    st["triggered_today"] = False
    st["day_peak"] = None
    st["reference_status"] = "READY"
    # Never dismiss an unfinished simulated lock at midnight.
    return True


def tick(st, reader, observed=None):
    observed = observed or utc()
    day = observed.astimezone(PARIS).date().isoformat()
    equity, wallet, unrealized = reader.equity()
    if st is None:
        # No reliable previous-day high exists on first start. Observe until midnight.
        return {
            "mode": "PAPER_ONLY", "version": 1, "paris_day": day,
            "reference_status": "BOOTSTRAP",
            "all_time_peak": str(equity), "day_peak": str(equity),
            "reference_peak": None, "target": None,
            "triggered_today": False, "active_lock": None, "snapshots": [],
            "last_equity": str(equity), "wallet": str(wallet),
            "unrealized": str(unrealized), "sampled_at": observed.isoformat(),
            "samples": 1, "transfer_adjustment": "NOT_VERIFIED",
            "disclaimer": "No real trading lock, no real TP/cancel, paper only",
        }, False, False

    rolled = roll_day(st, day)
    st["last_equity"] = str(equity)
    st["wallet"] = str(wallet)
    st["unrealized"] = str(unrealized)
    st["sampled_at"] = observed.isoformat()
    st["samples"] = int(st.get("samples", 0)) + 1
    st["day_peak"] = str(max(money(st["day_peak"]) if st.get("day_peak") else equity, equity))
    st["all_time_peak"] = str(max(money(st["all_time_peak"]), equity))
    triggered = maybe_trigger(st, equity, reader.positions, observed)
    # Keep all historical snapshots; never alter frozen entry/target prices.
    active = st.get("active_lock")
    if active:
        for position in active.get("positions", []):
            if position["paper_status"] != "WAIT":
                continue
            try:
                book = reader.best_book(position["symbol"])
            except RateWait:
                raise
            except Exception:
                continue
            if not book:
                continue
            limit = money(position["snapshot_price"])
            crossed = (book["bid"] >= limit if position["side"] == "LONG"
                       else book["ask"] <= limit)
            if crossed:
                position["paper_status"] = "PAPER_QUOTE_CROSSED"
                position["paper_fill_at"] = observed.isoformat(timespec="seconds")
        if all(p["paper_status"] != "WAIT" for p in active.get("positions", [])):
            active["all_paper_crossed"] = True
            active["completed_at"] = observed.isoformat(timespec="seconds")
            st["active_lock"] = None
    return st, triggered, rolled


def alert(st):
    token = os.getenv("BOT_TOKEN", "").strip()
    chat_id = os.getenv("CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    snap = st["snapshots"][-1]
    text = (
        "🧪 GÜNLÜK KÂR KİLİDİ — SADECE SANAL\n"
        f"Paris gün: {snap['day']}\n"
        f"Eşik: {snap['target']} USDT\n"
        f"Görülen özsermaye: {snap['trigger_equity']} USDT\n"
        f"Fotoğrafı çekilen pozisyon: {len(snap['positions'])}\n"
        "GERÇEK EMİR, TP veya ALIŞ KİLİDİ UYGULANMADI.\n"
        "Para transferi düzeltmesi henüz doğrulanmadı."
    )
    try:
        requests.post("https://api.telegram.org/bot" + token + "/sendMessage",
                      data={"chat_id": chat_id, "text": text}, timeout=10).raise_for_status()
    except Exception as exc:
        LOG.warning("Telegram paper alert failed (%s)", type(exc).__name__)


def run():
    reader = MarketReader()
    LOG.info("Starting READ-ONLY paper monitor: every=%ds target=%s%%; orders=NEVER",
             INTERVAL, PCT)
    backoff_until = 0
    previous_hour = None
    while True:
        if time.time() < backoff_until:
            time.sleep(min(INTERVAL, max(1, backoff_until - time.time())))
            continue
        try:
            st = load_state()
            st, triggered, rolled = tick(st, reader)
            persist(st)
            hour = utc().strftime("%Y-%m-%dT%H")
            if rolled or previous_hour != hour:
                LOG.info("paper state=%s day=%s peak=%s reference=%s target=%s equity=%s lock=%s transfers=%s",
                         st["reference_status"], st["paris_day"], st["all_time_peak"],
                         st.get("reference_peak"), st.get("target"), st["last_equity"],
                         bool(st.get("active_lock")), st["transfer_adjustment"])
            if triggered:
                LOG.warning("PAPER TARGET HIT day=%s target=%s equity=%s positions=%d NO_REAL_ORDERS",
                            st["paris_day"], st["snapshots"][-1]["target"],
                            st["last_equity"], len(st["snapshots"][-1]["positions"]))
                alert(st)
            previous_hour = hour
        except RateWait as exc:
            backoff_until = time.time() + exc.delay
            LOG.warning("Binance 418/429; read-only monitor backed off %ds", exc.delay)
        except Exception as exc:
            LOG.warning("Paper observation skipped (%s): %s",
                        type(exc).__name__, str(exc)[:180])
        time.sleep(INTERVAL)


def start():
    thread = threading.Thread(target=run, name="daily-profit-lock-PAPER-only",
                              daemon=True)
    thread.start()
    return thread
