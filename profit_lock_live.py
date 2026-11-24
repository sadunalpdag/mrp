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
        self._startup_reconciled = False

    @property
    def state(self):
        return self.ex.state.get("profit_lock_live")

    def locked(self):
        return bool(self.state and self.state.get("phase") != "WAIT_TARGET")

    def active(self):
        # A triggered lock is not dismissible by disabling the feature.
        return self.locked() or os.getenv("CRASH_FIXED_PROFIT_LOCK_ENABLED", "0") == "1"

    def entries_blocked(self):
        # Only a confirmed target-triggered close cycle can suspend new BUYs.
        # An old ERROR_LOCK without any cycle/snapshot/exit plan is a monitor
        # initialization failure, not evidence of a real closing operation.
        st = self.state or {}
        if st.get("phase") == "ERROR_LOCK" and not any(
                st.get(key) for key in ("cycle_id", "position_snapshot", "exit_plan")):
            return False
        return self.locked()

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

    def _sample(self, force_income=False):
        """Validated single-asset Binance equity, adjusted for cash transfers.

        Only the INITIAL reference is imported from the old paper monitor;
        thereafter our durable target is never updated by its daily peak.
        """
        data_dir = os.getenv("DATA_DIR", "/data")
        config_path = os.path.join(data_dir, "daily_profit_lock_paper_settings.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        else:
            config = {"target_pct": os.getenv("DAILY_PROFIT_PAPER_TARGET_PCT", "5")}
        pct = D("1")  # User-defined fixed live target: 1% per completed cycle
        if not D("0.5") <= pct <= D("20"):
            raise ValueError("INVALID_USER_CONFIGURED_PERCENT")
        now = datetime.fromtimestamp(self.ex.clock(), timezone.utc)
        acc = self.ex.broker.request("GET", "/fapi/v3/account")
        if not isinstance(acc, dict):
            raise ValueError("INVALID_ACCOUNT_RESPONSE")
        # Binance account responses may omit multiAssetsMargin or encode it as
        # a string. Margin mode is not required for USDT-valued account totals.
        # Validate equity fields below instead of rejecting valid responses.
        assets = acc.get("assets")
        if not isinstance(assets, list):
            raise ValueError("INVALID_ACCOUNT_ASSETS")
        # In Multi-Assets mode Binance reports account totals in USDT value.
        # Do not reject non-USDT collateral; use validated account-wide equity.
        wallet = D(str(acc["totalWalletBalance"]))
        unreal = D(str(acc["totalUnrealizedProfit"]))
        raw = D(str(acc["totalMarginBalance"]))
        if raw <= 0 or abs(raw - wallet - unreal) > D("0.1"):
            raise ValueError("INCONSISTENT_ACCOUNT_EQUITY")

        # On activation, import the already-published current fixed target.
        # No historical target reconstruction and no silent %1 substitution.
        if self.state is None:
            return raw, pct, raw * (D("1") + pct / 100), now

        st = self.state
        target = D(str(st["target"]))
        now_ms = int(self.ex.clock() * 1000)
        base = int(st.get("cashflow_from_ms") or 0)
        if base == 0:
            # First authenticated observation of a newly initialized lock.
            st["cashflow_from_ms"] = now_ms
            st["cashflow_checked_ms"] = now_ms
            st["cashflow_seen"] = []
            st["cashflow_total"] = "0"
            self._save(st)
            return raw, pct, target, now
        previous = int(st.get("cashflow_checked_ms") or 0)
        if previous < base or now_ms - previous > 6 * 24 * 3600 * 1000:
            raise ValueError("CASHFLOW_HISTORY_GAP")
        # Check transfer/income history at most once per ten minutes during
        # ordinary monitoring. A possible profit-target hit forces a fresh
        # income reconciliation before any closing lock can be triggered.
        if not force_income and now_ms - previous < 600_000:
            adjusted = raw - D(str(st.get("cashflow_total", "0")))
            if adjusted <= 0:
                raise ValueError("INVALID_ADJUSTED_EQUITY")
            return adjusted, pct, target, now
        # Re-read five minutes to absorb late-arriving income records.
        start = max(base + 1, previous - 300000)
        rows = self.ex.broker.request("GET", "/fapi/v1/income",
                         {"startTime": start, "endTime": now_ms, "limit": 1000})
        if not isinstance(rows, list) or len(rows) >= 1000:
            raise ValueError("CASHFLOW_HISTORY_INCOMPLETE")
        seen = set(st.get("cashflow_seen", []))
        total = D(str(st.get("cashflow_total", "0")))
        transfer_types = {
            "TRANSFER", "INTERNAL_TRANSFER", "CROSS_COLLATERAL_TRANSFER",
            "COIN_SWAP_DEPOSIT", "COIN_SWAP_WITHDRAW",
            "STRATEGY_UMFUTURES_TRANSFER"
        }
        for row in rows:
            if row.get("incomeType") not in transfer_types:
                continue
            stamp = int(row["time"])
            if stamp <= base or stamp > now_ms:
                continue
            if row.get("asset") != "USDT":
                raise ValueError("NON_USDT_CASHFLOW")
            key = str(row.get("incomeType")) + ":" + str(
                row.get("tranId") or (
                    str(stamp) + ":" + str(row.get("income")) + ":" +
                    str(row.get("info"))))
            if key in seen:
                continue
            total += D(str(row["income"]))
            seen.add(key)
        # Strictly retain deduplication history. Removing keys could double
        # count transfers in overlapping future Binance income queries.
        st["cashflow_seen"] = sorted(seen)
        st["cashflow_total"] = str(total)
        st["cashflow_checked_ms"] = now_ms
        self._save(st)
        adjusted = raw - total
        if adjusted <= 0:
            raise ValueError("INVALID_ADJUSTED_EQUITY")
        return adjusted, pct, target, now

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
        """Cancel every account order during a verified target-close cycle.

        Includes manual orders. Only invoked after a persisted CLOSING trigger.
        Binance confirmation is checked again before any new LIMIT exit.
        """
        if (self.state or {}).get("phase") != "CLOSING" or not self.state.get("cycle_id"):
            raise ValueError("CLOSE_CYCLE_NOT_CONFIRMED")
        planned_ids = {v.get("id") for v in (self.state.get("exit_plan") or {}).values()}
        for order in normal:
            if order.get("clientOrderId") in planned_ids:
                continue  # Already submitted cycle close; never cancel/reissue it.
            symbol = order.get("symbol")
            order_id = order.get("orderId")
            client_id = order.get("clientOrderId")
            if not symbol or (order_id is None and not client_id):
                raise ValueError("UNIDENTIFIED_OPEN_ORDER")
            payload = {"symbol": symbol}
            if order_id is not None:
                payload["orderId"] = order_id
            else:
                payload["origClientOrderId"] = client_id
            self.ex.broker.request("DELETE", "/fapi/v1/order", payload)
        for order in algo:
            symbol = order.get("symbol")
            algo_id = order.get("algoId")
            client_id = order.get("clientAlgoId")
            if not symbol or (algo_id is None and not client_id):
                raise ValueError("UNIDENTIFIED_ALGO_ORDER")
            payload = {"symbol": symbol}
            if algo_id is not None:
                payload["algoId"] = algo_id
            else:
                payload["clientAlgoId"] = client_id
            self.ex.broker.request("DELETE", "/fapi/v1/algoOrder", payload)

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
        # Fresh current mark price, not a stale trigger snapshot. A favorable
        # price movement may fill immediately; it is still LIMIT, not MARKET.
        mark = D(str(live["markPrice"]))
        if not mark.is_finite() or mark <= 0:
            raise ValueError("INVALID_CURRENT_MARK_PRICE")
        price = rounded(mark, tick, up=(side == "SELL"))
        qty = rounded(abs(D(str(live["positionAmt"]))), lot)
        if price <= 0 or qty <= 0 or qty > abs(D(str(live["positionAmt"]))):
            raise ValueError("INVALID_LIMIT_CLOSE_QUANTITY")
        cid = "plc" + hashlib.sha256(
            (str(self.state.get("cycle_id") or self.state["triggered_day"]) + "|" + key).encode()
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
        # A missing previously-submitted exit is not proof of completion.
        # If a position remains, halt retries and alert for manual review.
        live_keys = normalize_positions(positions)
        open_ids = {str(o.get("clientOrderId") or o.get("clientAlgoId") or "")
                    for o in normal + algo}
        for key, attempt in (st.get("exit_plan") or {}).items():
            symbol, position_side = key.split("|", 1)
            if (symbol, position_side) in live_keys and attempt["id"] not in open_ids:
                self._alert("LIMIT kapanış emri Binance açık emirlerinde görülmüyor: "
                            + symbol + " " + position_side + "; pozisyon hâlâ açık. "
                            "Otomatik tekrar yok; manuel kontrol gerekiyor.")
        if verified["all_clear"]:
            now = datetime.fromtimestamp(self.ex.clock(), timezone.utc)
            completed = closing_verified(st, True, True, True, now)
            # Closing is exchange-confirmed; immediately arm the next cycle.
            equity, pct, _, _ = self._sample()
            self._save(roll_after_midnight(completed, equity, pct, now, True, True, True))
            if not self.ex.state.get("paused") and not self.ex.state.get("halt"):
                self.ex.enabled.set()
            next_phase = (self.state or {}).get("phase", "UNKNOWN")
            self.ex.log("[PROFIT CYCLE] BINANCE VERIFIED FLAT; lock released; next_phase=%s; entries_enabled=%s" %
                        (next_phase, self.ex.enabled.is_set()))
            self.ex.alert("✅ SABİT KÂR HEDEFİ | Binance pozisyon ve emirleri sıfır doğrulandı; "
                          "kapanış kilidi kaldırıldı. Yeni alış durumu: " +
                          ("AÇIK" if self.ex.enabled.is_set() else "KAPALI (başka kontrol)"))
            return
        # Cancel account-wide pending orders, including manual and algo orders.
        # Do not submit a close until Binance confirms both order books empty.
        planned_ids = {v.get("id") for v in (st.get("exit_plan") or {}).values()}
        unexpected = [o for o in normal if o.get("clientOrderId") not in planned_ids]
        if unexpected or algo:
            self._prepare_exits(positions, unexpected, algo)
            _, positions, normal, algo = self._verify()
            if any(o.get("clientOrderId") not in planned_ids for o in normal) or algo:
                return
        current = normalize_positions(positions)
        snap_keys = {(x["symbol"], x["position_side"])
                     for x in st["position_snapshot"]}
        # Late entry fills may appear after an acknowledged cancellation.
        # Freeze each newly discovered position once, before any exit POST.
        for row in positions:
            qty = D(str(row["positionAmt"]))
            key = (row["symbol"], row.get("positionSide", "BOTH"))
            if qty and key not in snap_keys:
                added = self._build_snapshot([row])[0]
                st["position_snapshot"].append(added)
                snap_keys.add(key)
                self._save(st)
        for item in st["position_snapshot"]:
            live_amount = current.get((item["symbol"], item["position_side"]))
            if live_amount is None:
                continue
            row = next(p for p in positions if p["symbol"] == item["symbol"]
                       and p.get("positionSide", "BOTH") == item["position_side"]
                       and D(str(p["positionAmt"])) != 0)
            self._exit_one(item, row)

    def _reconcile_restart(self):
        """Clear a stale lock at process startup only when Binance is fully flat.

        A persisted closing cycle with live positions/orders must be resumed,
        never forgotten; otherwise restart could allow overlapping exposure.
        """
        if self._startup_reconciled:
            return
        self._startup_reconciled = True
        st = self.state or {}
        phase = st.get("phase")
        if phase in (None, "WAIT_TARGET"):
            return
        if phase == "ERROR_LOCK" and not any(
                st.get(k) for k in ("cycle_id", "position_snapshot", "exit_plan")):
            self.ex.state.pop("profit_lock_live", None)
            self.ex.save()
            self.ex.log("[PROFIT CYCLE] restart cleared untriggered error lock")
            return
        try:
            positions = normalize_positions(self._positions())
            normal, algo = self._orders()
            if positions or normal or algo:
                self.ex.log("[PROFIT CYCLE] restart retains active closing lock: positions=%d orders=%d algo=%d" %
                            (len(positions), len(normal), len(algo)))
                return
            self.ex.state.pop("profit_lock_live", None)
            self.ex.save()
            self.ex.log("[PROFIT CYCLE] restart verified flat account; stale lock cleared; new target will initialize")
            if not self.ex.state.get("paused") and not self.ex.state.get("halt"):
                self.ex.enabled.set()
            self.ex.alert("✅ SABİT KÂR HEDEFİ | Yeniden başlatmada Binance hesap boş doğrulandı; "
                          "eski kapanış kilidi kaldırıldı. Yeni alış: " +
                          ("AÇIK" if self.ex.enabled.is_set() else "KAPALI"))
        except Exception as exc:
            self.ex.log("[PROFIT CYCLE] restart reconciliation unavailable: %s; lock retained" %
                        type(exc).__name__)

    def step(self):
        self._reconcile_restart()
        st = self.state or {}
        if st.get("phase") == "ERROR_LOCK" and not any(
                st.get(key) for key in ("cycle_id", "position_snapshot", "exit_plan")):
            # Recover only an untriggered monitor error. Never reset a real
            # closing cycle, its exchange orders, or its persisted snapshot.
            self.ex.state.pop("profit_lock_live", None)
            self.ex.save()
            self.ex.log("[PROFIT CYCLE] cleared stale untriggered ERROR_LOCK; monitoring reinitializes")
            if not self.ex.state.get("paused") and not self.ex.state.get("halt"):
                self.ex.enabled.set()
        if not self.active() or self.ex.clock() < self.next_poll:
            return
        self.next_poll = self.ex.clock() + 120
        if self.locked():
            self._block()
        try:
            equity, pct, imported_target, instant = self._sample()
            st = self.state
            if st and st.pop("monitor_error", None):
                self._save(st)
                if st.get("phase") == "WAIT_TARGET" and not self.ex.state.get("paused") and not self.ex.state.get("halt"):
                    self.ex.enabled.set()
            if not st:
                st = initial_state(equity, pct, instant)
                # Retain the existing frozen target rather than using a
                # freshly computed transient intraday high.
                st["target"] = str(imported_target)
                st["reference"] = str(imported_target / (D("1") + pct / 100))
                start_ms = int(self.ex.clock() * 1000)
                st.update(cashflow_from_ms=start_ms, cashflow_checked_ms=start_ms,
                          cashflow_seen=[], cashflow_total="0")
                self._save(st)
            if st["phase"] == "WAIT_TARGET":
                # Verify cash transfers immediately before declaring a target
                # hit, even when the regular ten-minute income poll is not due.
                if equity >= D(str(st["target"])):
                    equity, pct, _, instant = self._sample(force_income=True)
                # One-off operator target, consumed durably before observation.
                # Never change a triggered close cycle or reapply after restart.
                override = os.getenv("CRASH_FIXED_TARGET_ONCE", "").strip()
                marker = "2026-11-24:1005"
                if (override == marker and instant.astimezone(PARIS).date().isoformat() == "2026-11-24" and
                        self.ex.state.get("profit_target_once_applied") != marker):
                    target = D("1005")
                    if not target.is_finite() or target <= 0:
                        raise ValueError("INVALID_ONE_OFF_TARGET")
                    st["target"] = str(target)
                    st["one_off_target"] = marker
                    self.ex.state["profit_target_once_applied"] = marker
                    self._save(st)
                    self.ex.log("[PROFIT CYCLE] one-off target applied: 1005 USDT; 2026-11-24")
                new, hit = observe(st, equity, instant)
                if hit:
                    self._block()
                    # Trigger and lock persist before any network mutations.
                    positions = self._positions()
                    new["position_snapshot"] = self._build_snapshot(positions)
                    new["exit_plan"] = {}
                    new["cycle_id"] = str(int(self.ex.clock() * 1000))
                    self._save(new)
                    self._alert("Hedefe ulaşıldı; yalnız LIMIT kapanış başlatıldı. "
                                "Yeni alımlar kilitlendi.")
                elif new != st:
                    self._save(new)
                return
            if st["phase"] == "CLOSING":
                self._closing_step()
            elif st["phase"] in ("WAIT_REBASE", "WAIT_MIDNIGHT"):
                self._block()
                verify, _, _, _ = self._verify()
                new = roll_after_midnight(
                    st, equity, pct, instant, verify["positions_flat"],
                    verify["owned_entries_cleared"] and not verify["unknown_opening_orders"],
                    verify["closing_orders_cleared"] and not verify["unknown_closing_orders"])
                if new != st:
                    self._save(new)
                    if new["phase"] == "WAIT_TARGET" and not self.ex.state.get("paused") and not self.ex.state.get("halt"):
                        self.ex.enabled.set()
        except Exception as err:
            # Monitoring failures must not disable entries unless a genuine
            # target-triggered closing cycle is underway.
            if self.locked() and (self.state or {}).get("phase") != "ERROR_LOCK":
                self._block()
            st = self.state or {"phase": "ERROR_LOCK", "target": None}
            st["monitor_error"] = type(err).__name__ + ": " + str(err)[:120]
            self._save(st)
            if st.get("phase") == "WAIT_TARGET" and not self.ex.state.get("paused") and not self.ex.state.get("halt"):
                self.ex.enabled.set()
            self._alert(type(err).__name__ + ": " + str(err)[:120] +
                        " | Kontrol gerekli, otomatik MARKET emri yok.")
