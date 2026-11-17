"""Separate CB2 Binance Futures real LIMIT executor.

Off unless CB2_REAL_TRADING_ENABLED=1. No modifications to Crash Bounce code,
state, positions, or orders. Uses its own persistent order IDs and JSON ledger.
No retries of uncertain BUY/TP submissions: reconcile by exact client order ID.
"""
import fcntl
import hashlib
import json
import logging
import os
import threading
import time
from decimal import Decimal, ROUND_UP
from pathlib import Path

from cb2_real_order_gateway import (PreflightBlocked, read_crash_snapshot,
                                    validate_fresh_momentum, plan_ask_limit)

D = Decimal
LOG = logging.getLogger("cb2_real_executor")
PENDING_TTL = 8 * 3600
_ACTIVE = None


class Executor:
    def __init__(self, directory, crash_directory, broker, deliver, clock=time.time):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "cb2_real_orders.json"
        self.marker = self.directory / "cb2_real_orders.initialized"
        self.guard = open(str(self.path) + ".lock", "a")
        fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.crash_directory = crash_directory
        self.broker = broker
        self.deliver = deliver
        self.clock = clock
        self.lock = threading.RLock()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if not isinstance(self.state.get("orders"), dict) or not isinstance(self.state.get("outbox"), list):
                raise ValueError("INVALID_CB2_ORDER_LEDGER")
        elif self.marker.exists():
            raise ValueError("CB2_ORDER_LEDGER_MISSING")
        else:
            self.state = {"orders": {}, "outbox": [], "halt": None}
            self._save()
            self.marker.touch()
        self.thread = None

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, allow_nan=False, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _alert(self, message, blocking=False):
        self.state["outbox"].append({
            "message": message, "blocking": blocking, "retry_at": 0})
        self._save()

    def _notify(self):
        if not self.state["outbox"]:
            return
        item = self.state["outbox"][0]
        if self.clock() < item["retry_at"]:
            return
        item["retry_at"] = self.clock() + 300
        self._save()
        try:
            receipt = self.deliver(item["message"])
            ok = receipt.get("ok") if isinstance(receipt, dict) else receipt is True
            if ok:
                self.state["outbox"].pop(0)
                self._save()
        except Exception:
            LOG.warning("[CB2 REAL TELEGRAM] notification delivery failed")

    def enqueue(self, item):
        with self.lock:
            if os.getenv("CB2_REAL_TRADING_ENABLED", "0") != "1":
                return False
            ident = str(item.get("id", ""))
            if not ident or ident in self.state["orders"]:
                return False
            try:
                validate_fresh_momentum(item, self.clock())
            except PreflightBlocked:
                return False
            cid = "cbm" + hashlib.sha256(ident.encode()).hexdigest()[:24]
            record = {
                "id": ident, "signal_id": ident, "symbol": item["symbol"],
                "observed_at": item["observed_at"],
                "sqs": item["sqs"], "indicators": item["indicators"],
                "filters_passed": True, "archetype": item["archetype"],
                "day": time.strftime("%Y-%m-%d", time.gmtime(self.clock())),
                "phase": "QUEUED", "created_at": self.clock(),
                "entry_id": cid + "e", "tp_id": cid + "t",
                "next_check": 0,
            }
            # Durable enqueue. A duplicate signal ID never opens a second BUY.
            self.state["orders"][ident] = record
            self._save()
            LOG.info("[CB2 REAL] queued %s %s", item["symbol"], ident)
            return True

    def _halt(self, rec, reason):
        if self.state["halt"]:
            return  # Never flood or reset an existing protective halt.
        self.state["halt"] = reason
        self._save()
        self._alert("⚠️ CB2 GERÇEK İŞLEM KORUMA | " + rec["symbol"] +
                    "\n" + reason + "\nYeni CB2 alışları durdu; mevcut Binance pozisyon ve emirlerini kontrol edin.",
                    blocking=True)

    def _fresh_entry(self, rec):
        now = self.clock()
        validate_fresh_momentum(rec, now)
        snapshot = read_crash_snapshot(self.crash_directory, now)
        if self.state["halt"] or any(x.get("blocking") for x in self.state["outbox"]):
            raise PreflightBlocked("CB2_SAFETY_ALERT_PENDING")
        if rec["symbol"] in snapshot["quarantined_symbols"]:
            raise PreflightBlocked("CRASH_SYMBOL_QUARANTINED")
        if not os.getenv("BOT_TOKEN") or not os.getenv("CHAT_ID"):
            raise PreflightBlocked("TELEGRAM_NOT_CONFIGURED")
        if any(o is not rec and o["symbol"] == rec["symbol"]
               and o["phase"] not in ("DONE", "NO_FILL", "BLOCKED", "CLOSED_EXTERNALLY")
               for o in self.state["orders"].values()):
            raise PreflightBlocked("CB2_EXISTING_SYMBOL_ORDER")
        # Crash's live preflight uses the same Binance account positions/orders.
        # Reserve ONE of the operator's 11 open slots for Crash to reduce races.
        self.broker.max_open_positions = max(1, snapshot["max_open_positions"] - 1)
        self.broker.max_pending_entries = snapshot["max_pending_entries"]
        self.broker.max_symbols = snapshot["max_symbols"]
        symbol = rec["symbol"]
        info = self.broker.info(symbol)
        side = self.broker.preflight(symbol)
        # Exchange book ticker AFTER capacity and leverage preflight, minimizing staleness.
        quote = self.broker.request("GET", "/fapi/v1/ticker/bookTicker",
                                    {"symbol": symbol}, signed=False)
        if not isinstance(quote, dict) or quote.get("symbol") != symbol:
            raise PreflightBlocked("BINANCE_ASK_MISSING")
        proposed = plan_ask_limit(symbol, quote.get("askPrice"), info,
                                  snapshot["amount_usdt"])
        if not 0 <= self.clock() - rec["created_at"] <= 120:
            raise PreflightBlocked("SIGNAL_EXPIRED_AFTER_PREFLIGHT")
        proposed["position_side"] = side
        return proposed, snapshot

    def _enter(self, rec):
        symbol = rec["symbol"]
        try:
            plan, snapshot = self._fresh_entry(rec)
        except Exception as exc:
            from crash_night import BrokerError
            if isinstance(exc, BrokerError) and exc.not_sent:
                # Rate limit preflight never sent a BUY; this signal expires.
                reason = "BINANCE_PREFLIGHT_DEFERRED"
            else:
                reason = str(exc)[:90] if isinstance(exc, (PreflightBlocked, ValueError)) else type(exc).__name__
            rec.update(phase="BLOCKED", reason=reason, entry_never_sent=True)
            self._save()
            self._alert("🧪 CB2 SANAL TAKİP | " + symbol +
                        "\nGerçek LIMIT BUY açılmadı: " + reason, blocking=False)
            return
        # Unique persisted client ID and intent BEFORE any POST.
        rec.update(phase="ENTRY_UNKNOWN", plan=plan, submitted_at=self.clock(),
                   price=plan["price"], quantity=plan["quantity"],
                   position_side=plan["position_side"], amount_usdt=str(snapshot["amount_usdt"]))
        self._save()
        try:
            self.broker.buy(symbol, plan, rec["entry_id"])
        except Exception as exc:
            from crash_night import BrokerError
            if isinstance(exc, BrokerError) and exc.not_sent:
                rec.update(phase="NO_FILL", entry_never_sent=True,
                           reason="BUY_NOT_SENT_RATE_LIMIT")
                self._save()
                self._alert("🧪 CB2 SANAL TAKİP | " + symbol +
                            "\nBinance koruması: gerçek BUY isteği gönderilmedi.")
                return
            # Timeout/unknown POST: NEVER submit again; exact-ID query only.
            rec["reason"] = "BUY_RESPONSE_UNCERTAIN_" + type(exc).__name__
            self._save()
            self._halt(rec, "CB2_BUY_RESPONSE_UNCERTAIN")
            return
        LOG.info("[CB2 REAL BUY POST] %s client=%s", symbol, rec["entry_id"])
        self._reconcile_entry(rec)

    @staticmethod
    def _verify(entry, rec):
        return (isinstance(entry, dict) and
                entry.get("symbol") == rec["symbol"] and
                entry.get("clientOrderId") == rec["entry_id"] and
                entry.get("side") == "BUY" and
                entry.get("type") == "LIMIT")

    def _reconcile_entry(self, rec):
        from crash_night import BrokerError
        symbol = rec["symbol"]
        try:
            entry = self.broker.order(symbol, rec["entry_id"])
        except BrokerError as exc:
            # -2013 might be exchange indexing delay, not proof of no BUY.
            if exc.code == "-2013":
                rec["next_check"] = self.clock() + 30
            else:
                rec["next_check"] = self.clock() + 60
            self._save()
            return
        if not self._verify(entry, rec):
            self._halt(rec, "CB2_ENTRY_IDENTITY_MISMATCH")
            return
        status = entry.get("status")
        try:
            qty = D(str(entry["executedQty"]))
            if not qty.is_finite() or qty < 0 or qty > D(rec["quantity"]):
                raise ValueError
        except (KeyError, ValueError, ArithmeticError):
            self._halt(rec, "CB2_ENTRY_EXECUTED_QTY_INVALID")
            return
        if status in ("NEW", "PARTIALLY_FILLED"):
            age = self.clock() - rec["submitted_at"]
            if qty > 0 or age >= PENDING_TTL:
                # Cancel ONLY the CB2-owned exact BUY ID. For a partial fill
                # requery immediately to protect any confirmed filled amount.
                try:
                    self.broker.cancel_entry(symbol, rec["entry_id"])
                except Exception:
                    pass
                try:
                    final = self.broker.order(symbol, rec["entry_id"])
                    if not self._verify(final, rec):
                        self._halt(rec, "CB2_ENTRY_IDENTITY_AFTER_CANCEL")
                        return
                    after_qty = D(str(final["executedQty"]))
                    if (not after_qty.is_finite() or after_qty < 0
                            or after_qty > D(rec["quantity"])):
                        self._halt(rec, "CB2_CANCEL_FILLED_QTY_INVALID")
                        return
                    entry, qty, status = final, after_qty, final.get("status")
                except Exception:
                    rec["next_check"] = self.clock() + 15
                    self._save()
                    if qty > 0 and not rec.get("partial_fill_warned"):
                        rec["partial_fill_warned"] = True
                        self._save()
                        self._halt(rec, "CB2_PARTIAL_FILL_NOT_PROTECTED")
                    return
                if status in ("NEW", "PARTIALLY_FILLED"):
                    rec["next_check"] = self.clock() + 15
                    self._save()
                    if qty > 0 and not rec.get("partial_fill_warned"):
                        rec["partial_fill_warned"] = True
                        self._save()
                        self._halt(rec, "CB2_PARTIAL_FILL_CANCEL_UNCONFIRMED")
                    return
                # On CANCELED/FILLED/EXPIRED with nonzero quantity below,
                # place TP immediately using the final exact-ID fill details.
            else:
                rec["next_check"] = self.clock() + 45
                if not rec.get("pending_notified"):
                    rec["pending_notified"] = True
                    self._save()
                    self._alert("⏳ CB2 GERÇEK LIMIT BUY BEKLİYOR | " + symbol +
                                "\nAlış: " + rec["price"] +
                                " | Miktar: " + rec["quantity"] +
                                "\nHenüz dolum doğrulanmadı; %1 TP ancak dolumdan sonra kurulacak.")
                return
        if status not in ("FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"):
            self._halt(rec, "CB2_ENTRY_STATUS_UNEXPECTED")
            return
        if qty == 0:
            rec["phase"] = "NO_FILL"
            self._save()
            self._alert("ℹ️ CB2 GERÇEK ALIŞ DOLMADI | " + symbol +
                        "\nLIMIT emir borsada kapanmış; sanal takip sürüyor.")
            return
        # Do not invent an average execution price. Binance reports avgPrice;
        # cumulative notional is an independent fallback.
        try:
            avg = D(str(entry.get("avgPrice", 0)))
            if avg <= 0:
                avg = D(str(entry.get("cumQuote", 0))) / qty
            if not avg.is_finite() or avg <= 0:
                raise ArithmeticError
            tick = D(str({f["filterType"]: f for f in
                          self.broker.info(symbol)["filters"]}["PRICE_FILTER"]["tickSize"]))
            tp = (avg * D("1.01") / tick).to_integral_value(rounding=ROUND_UP) * tick
            if tp <= avg:
                raise ArithmeticError
        except (KeyError, ValueError, ArithmeticError, ZeroDivisionError):
            self._halt(rec, "CB2_FILL_PRICE_OR_TP_UNVERIFIED")
            return
        rec.update(phase="TP_READY", filled_qty=str(qty), avg_fill=str(avg),
                   tp_price=format(tp, "f"), next_check=0)
        self._save()
        self._place_tp(rec)

    def _place_tp(self, rec):
        # Do not re-POST after a possible network send; move to TP_UNKNOWN first.
        rec["phase"] = "TP_UNKNOWN"
        rec["next_check"] = self.clock() + 15
        self._save()
        try:
            self.broker.tp(rec["symbol"], rec["filled_qty"],
                           rec["tp_price"], rec["tp_id"],
                           position_side=rec["position_side"])
        except Exception as exc:
            from crash_night import BrokerError
            if isinstance(exc, BrokerError) and exc.not_sent:
                rec["phase"] = "TP_READY"
                rec["next_check"] = self.clock() + 30
                self._save()
                return
            self._halt(rec, "CB2_TP_SUBMISSION_UNCERTAIN")
            return
        self._reconcile_tp(rec)

    def _reconcile_tp(self, rec):
        from crash_night import BrokerError
        try:
            tp = self.broker.order(rec["symbol"], rec["tp_id"])
        except BrokerError as exc:
            rec["next_check"] = self.clock() + 30
            self._save()
            if exc.code == "-2013":
                # No assumption that a POST was unreceived.
                if not rec.get("tp_missing_warned"):
                    rec["tp_missing_warned"] = True
                    self._save()
                    self._halt(rec, "CB2_TP_NOT_VERIFIED")
            return
        if not (isinstance(tp, dict) and tp.get("symbol") == rec["symbol"]
                and tp.get("clientOrderId") == rec["tp_id"]
                and tp.get("side") == "SELL"
                and tp.get("positionSide", "BOTH") == rec["position_side"]
                and D(str(tp.get("origQty", -1))) == D(rec["filled_qty"])
                and D(str(tp.get("price", -1))) == D(rec["tp_price"])):
            self._halt(rec, "CB2_TP_ORDER_IDENTITY_INVALID")
            return
        status = tp.get("status")
        if status == "FILLED":
            rec["phase"] = "DONE"
            self._save()
            self._alert("✅ CB2 GERÇEK TP DOLDU | " + rec["symbol"] +
                        "\nAlış: " + rec["avg_fill"] + " | TP: " + rec["tp_price"])
        elif status in ("NEW", "PARTIALLY_FILLED"):
            first = rec["phase"] != "PROTECTED"
            rec["phase"] = "PROTECTED"
            rec["next_check"] = self.clock() + 45
            self._save()
            if first:
                self._alert("✅ CB2 GERÇEK ALIŞ + TP DOĞRULANDI | " + rec["symbol"] +
                            "\nAlış ort.: " + rec["avg_fill"] +
                            " | Miktar: " + rec["filled_qty"] +
                            "\nTP +%1: " + rec["tp_price"] + " | 10x | SL yok")
        else:
            self._halt(rec, "CB2_TP_NOT_ACTIVE")

    def step(self):
        with self.lock:
            self._notify()
            # TP/BUY reconciliation gets priority over all new exposure.
            now = self.clock()
            risky = [r for r in self.state["orders"].values()
                     if r["phase"] in ("ENTRY_UNKNOWN", "TP_READY", "TP_UNKNOWN", "PROTECTED")
                     and now >= r.get("next_check", 0)]
            if risky:
                r = sorted(risky, key=lambda x: x.get("next_check", 0))[0]
                if r["phase"] == "ENTRY_UNKNOWN":
                    self._reconcile_entry(r)
                elif r["phase"] == "TP_READY":
                    self._place_tp(r)
                else:
                    self._reconcile_tp(r)
                return
            if (os.getenv("CB2_REAL_TRADING_ENABLED", "0") != "1"
                    or self.state["halt"]
                    or any(o.get("blocking") for o in self.state["outbox"])
                    or any(r["phase"] == "TP_UNKNOWN" for r in self.state["orders"].values())):
                return
            queued = [r for r in self.state["orders"].values() if r["phase"] == "QUEUED"]
            if queued:
                self._enter(min(queued, key=lambda r: r["created_at"]))

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def run():
            while True:
                try:
                    self.step()
                except Exception as exc:
                    LOG.error("[CB2 REAL EXECUTOR ERR] %s", type(exc).__name__)
                    with self.lock:
                        self.state["halt"] = "CB2_EXECUTOR_ERROR"
                        self._save()
                        self._alert("⚠️ CB2 GERÇEK YÜRÜTÜCÜ HATASI: yeni CB2 alışları durdu. "
                                    "Mevcut emirler Binance üzerinden kontrol edilmeli.", blocking=True)
                time.sleep(1)
        self.thread = threading.Thread(target=run, name="cb2-real-executor", daemon=True)
        self.thread.start()


def start_executor(base_dir, crash_dir, broker, deliver):
    global _ACTIVE
    if os.getenv("CB2_REAL_TRADING_ENABLED", "0") != "1":
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    executor = Executor(base_dir, crash_dir, broker, deliver)
    _ACTIVE = executor
    executor.start()
    return executor


def enqueue_signal(item):
    if _ACTIVE is None:
        return False
    return _ACTIVE.enqueue(item)
