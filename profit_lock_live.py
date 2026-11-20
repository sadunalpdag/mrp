"""Live fixed-goal close coordinator for the Crash Night executor.

Explicit opt-in only. Persist before every exchange mutation. Never issue a
MARKET order. If a read or write is ambiguous, retain the entry lock and alert.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from fixed_profit_target import initial_state, observe, closing_verified, roll_after_midnight
from profit_close_reconcile import normalize_positions, reconcile_close

D = Decimal
PARIS = ZoneInfo("Europe/Paris")


class LiveProfitLock:
    def __init__(self, executor):
        self.ex = executor
        self.next_poll = 0

    @property
    def state(self):
        return self.ex.state.get("profit_lock_live")

    def locked(self):
        return bool(self.state and self.state.get("phase") != "WAIT_TARGET")

    def active(self):
        # A triggered lock is not dismissible by disabling the feature.
        return self.locked() or os.getenv("CRASH_FIXED_PROFIT_LOCK_ENABLED", "0") == "1"

    def _save(self, state):
        self.ex.state["profit_lock_live"] = state
        self.ex.save()

    def _alert(self, msg):
        st = self.state or {}
        if st.get("last_alert") == msg:
            return
        st["last_alert"] = msg
        self._save(st)
        self.ex.alert("⚠️ SABİT KÂR HEDEFİ | " + msg)

    def _sample(self):
        # Use the existing independent USD-M income/transfer validator for the
        # cashflow-adjusted equity and for the operator's configured percentage.
        # Never guess a historical peak or silently interpret a deposit as profit.
        path = os.path.join(os.getenv("DATA_DIR", "/data"),
                            "daily_profit_paper_state.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        latest = data.get("last") or {}
        updated = datetime.fromisoformat(latest["at"].replace("Z", "+00:00"))
        now = datetime.fromtimestamp(self.ex.clock(), timezone.utc)
        if abs((now - updated).total_seconds()) > 900:
            raise ValueError("STALE_ACCOUNT_EQUITY_OBSERVER")
        if latest.get("data_quality") != "USDT_TRANSFER_INCOME_ADJUSTED_PAPER_ESTIMATE":
            raise ValueError("EQUITY_NOT_TRANSFER_VALIDATED")
        checked = int(data.get("transfer_checked_ms") or 0) / 1000
        if abs(self.ex.clock() - checked) > 900:
            raise ValueError("STALE_TRANSFER_RECONCILIATION")
        pct = D(str(data["target_pct"]))
        equity = D(str(latest["adjusted_equity"]))
        target = D(str(data["target"]))
        if not (D("0.5") <= pct <= D("20")) or equity <= 0 or target <= 0:
            raise ValueError("INVALID_FIXED_GOAL_SOURCE")
        return equity, pct, target, now

    def _positions(self):
        return self.ex.broker.positions()

    def _orders(self):
        # Both regular and Algo lists are required to verify closing state.
        normal = self.ex.broker.request("GET", "/fapi/v1/openOrders",
                                        budget_priority="protect")
        algo = self.ex.broker.request("GET", "/fapi/v1/openAlgoOrders",
                                      budget_priority="protect")
        if not isinstance(normal, list) or not isinstance(algo, list):
            raise ValueError("INVALID_ACCOUNT_ORDERS")
        return normal, algo

    def _tracked_ids(self):
        entries, exits = set(), set()
        for order in self.ex.state["orders"].values():
            if order.get("entry_id"):
                entries.add(order["entry_id"])
            if order.get("tp_id"):
                exits.add(order["tp_id"])
        st = self.state or {}
        for item in st.get("exit_plan", {}).values():
            exits.add(item["id"])
        return entries, exits

    def _verify(self):
        entries, exits = self._tracked_ids()
        positions = self._positions()
        normal, algo = self._orders()
        snapshot = self.state.get("position_snapshot") or []
        return reconcile_close(snapshot, positions, normal, algo, entries, exits), positions, normal, algo

    def _block(self):
        self.ex.enabled.clear()
        # Discard stale retry candidates while keeping all TP protection alive.
        self.ex.preflight_retries.clear()

    def _build_snapshot(self, positions):
        result = []
        for p in positions:
            amt = D(str(p["positionAmt"]))
            if not amt:
                continue
            side = p.get("positionSide", "BOTH")
            if side not in ("BOTH", "LONG", "SHORT"):
                raise ValueError("INVALID_HEDGE_SIDE")
            mark = D(str(p["markPrice"]))
            if mark <= 0:
                raise ValueError("INVALID_MARK_PRICE")
            result.append({"symbol": p["symbol"], "position_side": side,
                           "amount": str(amt), "price": str(mark)})
        return result

    def _cancel_verified_entries(self, normal):
        ids, _ = self._tracked_ids()
        for item in normal:
            cid = item.get("clientOrderId")
            if cid not in ids:
                continue
            # Only cancel an explicitly tracked, ordinary opening order.
            if item.get("side") != "BUY" or item.get("type") != "LIMIT":
                continue
            self.ex.broker.cancel_entry(item["symbol"], cid)

    def _cancel_owned_existing_tp(self, normal):
        _, tp_ids = self._tracked_ids()
        for item in normal:
            cid = item.get("clientOrderId")
            if cid not in tp_ids or not str(cid).startswith("cbn"):
                continue
            # Never cancel an unowned or manually submitted TP.
            self.ex.broker.cancel_owned_tp(item["symbol"], cid)

    def _prepare_exits(self, positions, normal, algo):
        # Unknown/manual open orders need human reconciliation, not deletion.
        entries, exits = self._tracked_ids()
        check = reconcile_close(self.state.get("position_snapshot") or [], positions,
                                normal, algo, entries, exits)
        if check["unknown_opening_orders"]:
            raise ValueError("UNKNOWN_OPENING_ORDERS_REQUIRE_REVIEW")
        # Unknown closing orders are also left intact. Avoid duplicate TP.
        for o in normal + algo:
            cid = o.get("clientOrderId") or o.get("clientAlgoId")
            if cid not in exits and not (o.get("side") == "BUY" and
                                        o.get("type") == "LIMIT"):
                raise ValueError("UNTRACKED_ORDER_REQUIRES_REVIEW")
        if check["owned_pending_entries"]:
            self._cancel_verified_entries(normal)
            return
        self._cancel_owned_existing_tp(normal)
        return

    def _exit_one(self, item, live):
        key = item["symbol"] + "|" + item["position_side"]
        plan = self.state.setdefault("exit_plan", {})
        if key in plan:
            # NEVER resubmit after an uncertain write. Broker operator must
            # verify the deterministic client ID and position quantities.
            return
        side = "SELL" if D(str(live["positionAmt"])) > 0 else "BUY"
        position_side = live.get("positionSide", "BOTH")
        if position_side not in ("BOTH", "LONG", "SHORT"):
            raise ValueError("UNSUPPORTED_POSITION_MODE")
        info = self.ex.broker.info(item["symbol"], budget_priority="protect")
        filters = {x["filterType"]: x for x in info["filters"]}
        from crash_night import rounded
        tick = D(str(filters["PRICE_FILTER"]["tickSize"]))
        lot = D(str(filters["LOT_SIZE"]["stepSize"]))
        # Frozen target uses current mark price at FIRST trigger. A favorable
        # price movement may fill immediately; it is still LIMIT, not MARKET.
        price = rounded(item["price"], tick, up=(side == "SELL"))
        qty = rounded(abs(D(str(live["positionAmt"]))), lot)
        if price <= 0 or qty <= 0 or qty > abs(D(str(live["positionAmt"]))):
            raise ValueError("INVALID_LIMIT_CLOSE_QUANTITY")
        cid = "plc" + hashlib.sha256(
            (self.state["triggered_day"] + "|" + key).encode()
        ).hexdigest()[:24]
        payload = {"symbol": item["symbol"], "side": side, "type": "LIMIT",
                   "timeInForce": "GTC", "quantity": str(qty),
                   "price": str(price), "newClientOrderId": cid}
        if position_side == "BOTH":
            payload["reduceOnly"] = "true"
        else:
            payload["positionSide"] = position_side
        # Persist exact attempt before submission. On network timeout never
        # blindly issue the same (or another) close again.
        plan[key] = {"id": cid, "payload": payload, "status": "UNKNOWN"}
        self._save(self.state)
        self.ex.broker.request("POST", "/fapi/v1/order", payload)
        plan[key]["status"] = "SENT"
        self._save(self.state)

    def _closing_step(self):
        self._block()
        st = self.state
        verified, positions, normal, algo = self._verify()
        if verified["all_clear"]:
            now = datetime.fromtimestamp(self.ex.clock(), timezone.utc)
            self._save(closing_verified(st, True, True, True, now))
            return
        # First reconcile and cancel exact bot-owned entries and TP. Never
        # indiscriminately cancel all account orders.
        if st.get("exit_plan"):
            # Exits were attempted; check partial fills and keep the lock.
            # Unfilled exits retain their frozen prices; no MARKET fallback.
            return
        self._prepare_exits(positions, normal, algo)
        verified, positions, normal, algo = self._verify()
        if verified["owned_pending_entries"] or verified["unknown_opening_orders"]:
            return
        # All surviving order conflicts need verification on next tick.
        _, exits = self._tracked_ids()
        if any((o.get("clientOrderId") or o.get("clientAlgoId")) in exits
               for o in normal + algo):
            return
        current = normalize_positions(positions)
        for item in st["position_snapshot"]:
            live_amount = current.get((item["symbol"], item["position_side"]))
            if live_amount is None:
                continue
            row = next(p for p in positions if p["symbol"] == item["symbol"]
                       and p.get("positionSide", "BOTH") == item["position_side"]
                       and D(str(p["positionAmt"])) != 0)
            self._exit_one(item, row)

    def step(self):
        if not self.active() or self.ex.clock() < self.next_poll:
            return
        self.next_poll = self.ex.clock() + 120
        if self.locked():
            self._block()
        try:
            equity, pct, imported_target, instant = self._sample()
            st = self.state
            if not st:
                st = initial_state(equity, pct, instant)
                # Retain the existing frozen target rather than using a
                # freshly computed transient intraday high.
                st["target"] = str(imported_target)
                st["reference"] = str(imported_target / (D("1") + pct / 100))
                self._save(st)
            if st["phase"] == "WAIT_TARGET":
                new, hit = observe(st, equity, instant)
                if hit:
                    self._block()
                    # Trigger and lock persist before any network mutations.
                    positions = self._positions()
                    new["position_snapshot"] = self._build_snapshot(positions)
                    new["exit_plan"] = {}
                    self._save(new)
                    self._alert("Hedefe ulaşıldı; yalnız LIMIT kapanış başlatıldı. "
                                "Yeni alımlar kilitlendi.")
                elif new != st:
                    self._save(new)
                return
            if st["phase"] == "CLOSING":
                self._closing_step()
            elif st["phase"] == "WAIT_MIDNIGHT":
                self._block()
                verify, _, _, _ = self._verify()
                new = roll_after_midnight(
                    st, equity, pct, instant, verify["positions_flat"],
                    verify["owned_entries_cleared"] and not verify["unknown_opening_orders"],
                    verify["closing_orders_cleared"])
                if new != st:
                    self._save(new)
                    if new["phase"] == "WAIT_TARGET" and not self.ex.state.get("paused") and not self.ex.state.get("halt"):
                        self.ex.enabled.set()
        except Exception as err:
            if self.locked():
                self._block()
            self._alert(type(err).__name__ + ": " + str(err)[:120] +
                        " | Kontrol gerekli, otomatik MARKET emri yok.")
