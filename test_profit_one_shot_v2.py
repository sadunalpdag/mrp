"""Offline proof of opt-in one-shot profit closing; no live exchange calls."""
import os
from decimal import Decimal
from unittest.mock import patch
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from test_profit_lock_live import FakeExecutor
from profit_lock_live import LiveProfitLock


class OneShotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.extra = patch.dict(os.environ, {"DATA_DIR": self.temp.name,
                       "CRASH_FIXED_PROFIT_LOCK_ENABLED": "1",
                       "CRASH_PROFIT_ONE_SHOT_V2": "1"})
        self.extra.start()
        self.addCleanup(self.extra.stop)
        self.clock = [datetime(2026, 11, 20, 15, tzinfo=timezone.utc).timestamp()]
        self.ex = FakeExecutor(self.clock)
        self.ex.log = lambda message: None
        self.ctrl = LiveProfitLock(self.ex)

    def advance(self):
        self.clock[0] += 130

    def test_release_after_confirmed_limit_and_ignore_new_manual_order(self):
        broker = self.ex.broker
        broker.positions_data = [{"symbol": "ETHUSDT", "positionSide": "BOTH",
                                  "positionAmt": "1", "markPrice": "100"}]
        self.ctrl.step()
        self.advance()
        broker.balance = Decimal("880")
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "CLOSING")
        self.assertTrue(self.ctrl.state["one_shot_v2"])
        self.advance()
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "WATCH_EXITS")
        self.assertTrue(self.ex.enabled.is_set())
        self.assertFalse(self.ctrl.entries_blocked())
        self.assertEqual(self.ctrl.closing_symbols(), {"ETHUSDT"})
        count = len(broker.writes)
        broker.regular.append({"symbol": "BTCUSDT", "clientOrderId": "manual-new",
                               "side": "BUY", "type": "LIMIT"})
        self.advance()
        self.ctrl.step()
        self.assertEqual(len(broker.writes), count)
        self.assertTrue(any(x.get("clientOrderId") == "manual-new"
                            for x in broker.regular))

    def test_legacy_cycle_is_never_migrated_while_open(self):
        self.ex.state["profit_lock_live"] = {
            "phase": "CLOSING", "target": "880", "cycle_id": "existing",
            "position_snapshot": [], "exit_plan": [],
            "cashflow_from_ms": int(self.clock[0]*1000),
            "cashflow_checked_ms": int(self.clock[0]*1000),
            "cashflow_total": "0", "cashflow_seen": []}
        # The old cycle remains locked regardless of the new feature toggle.
        self.assertTrue(self.ctrl.entries_blocked())
        self.assertFalse(self.ctrl.state.get("one_shot_v2", False))




class OldExitRepriceTests(unittest.TestCase):
    """No real exchange writes; assert idempotent cancel-then-LIMIT ordering."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {
            "DATA_DIR": self.temp.name,
            "CRASH_FIXED_PROFIT_LOCK_ENABLED": "1",
            "CRASH_PROFIT_ONE_SHOT_V2": "1",
            "CRASH_REPRICE_OLD_EXITS_ENABLED": "1",
            "CRASH_FIXED_TARGET_ONCE": "2026-11-26:1027",
        })
        env.start()
        self.addCleanup(env.stop)
        self.clock = [datetime(2026, 11, 26, 19, tzinfo=timezone.utc).timestamp()]
        self.ex = FakeExecutor(self.clock)
        self.ex.log = lambda msg: None
        self.ctrl = LiveProfitLock(self.ex)
        self.ex.state["profit_lock_live"] = {
            "phase": "WATCH_EXITS", "target": "1017",
            "triggered_day": "2026-11-24", "cycle_id": "old-cycle",
            "one_shot_v2": True, "one_off_target": "2026-11-24:1017",
            "cashflow_from_ms": int(self.clock[0] * 1000),
            "cashflow_checked_ms": int(self.clock[0] * 1000),
            "cashflow_seen": [], "cashflow_total": "0",
            "position_snapshot": [
                {"symbol": "ETHUSDT", "position_side": "BOTH",
                 "amount": "1", "price": "100"},
                {"symbol": "BTCUSDT", "position_side": "BOTH",
                 "amount": "-0.05", "price": "60000"}],
            "exit_plan": {
                "ETHUSDT|BOTH": {"id": "plc-old-eth", "status": "SENT"},
                "BTCUSDT|BOTH": {"id": "plc-old-btc", "status": "SENT"}},
        }
        self.ex.broker.balance = Decimal("1028")
        self.ex.broker.positions_data = [
            {"symbol": "ETHUSDT", "positionSide": "BOTH",
             "positionAmt": "1", "markPrice": "99.25"},
            {"symbol": "BTCUSDT", "positionSide": "BOTH",
             "positionAmt": "-0.05", "markPrice": "60100"},
        ]
        self.ex.broker.regular = [
            {"symbol": "ETHUSDT", "clientOrderId": "plc-old-eth",
             "type": "LIMIT", "side": "SELL", "positionSide": "BOTH",
             "reduceOnly": True},
            {"symbol": "BTCUSDT", "clientOrderId": "plc-old-btc",
             "type": "LIMIT", "side": "BUY", "positionSide": "BOTH",
             "reduceOnly": True},
        ]
        broker = self.ex.broker
        original = broker.request

        def request(method, path, params=None, budget_priority=None):
            if method == "DELETE" and path == "/fapi/v1/order":
                broker.writes.append((method, path, dict(params)))
                broker.regular = [
                    x for x in broker.regular
                    if x["clientOrderId"] != params["origClientOrderId"]]
                return {"status": "CANCELED"}
            result = original(method, path, params, budget_priority)
            if method == "POST" and path == "/fapi/v1/order":
                broker.regular[-1]["type"] = "LIMIT"
                broker.regular[-1]["positionSide"] = "BOTH"
            return result
        broker.request = request

    def advance(self):
        self.clock[0] += 130

    def step(self):
        self.advance()
        self.ctrl.step()

    def test_two_old_exits_replaced_once_and_new_limits_verified(self):
        self.ctrl.step()  # target is promoted to 1027, jobs frozen
        self.assertEqual(self.ctrl.state["target"], "1027")
        self.assertEqual(self.ctrl.state["phase"], "REPRICE_OLD_EXITS")
        self.assertEqual(self.ex.broker.writes, [])
        for _ in range(12):
            self.step()
            if self.ctrl.state["phase"] == "WATCH_EXITS":
                break
        self.assertEqual(self.ctrl.state["phase"], "WATCH_EXITS")
        writes = self.ex.broker.writes
        self.assertEqual(len([w for w in writes if w[0] == "DELETE"]), 2)
        posts = [w for w in writes if w[0] == "POST"]
        self.assertEqual(len(posts), 2)
        self.assertEqual({p[2]["side"] for p in posts}, {"SELL", "BUY"})
        self.assertTrue(all(p[2]["type"] == "LIMIT" for p in posts))
        self.assertTrue(all(p[2].get("reduceOnly") == "true" for p in posts))
        self.assertTrue(self.ex.enabled.is_set())
        self.step()
        self.assertEqual(len(self.ex.broker.writes), 4)  # never reprice again

    def test_missing_old_order_blocks_all_mutations(self):
        self.ex.broker.regular.pop()
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "WATCH_EXITS")
        self.assertEqual(self.ex.broker.writes, [])

    def test_conflicting_manual_order_blocks_all_mutations(self):
        self.ex.broker.regular.append({
            "symbol": "ETHUSDT", "clientOrderId": "manual-protected",
            "type": "LIMIT", "side": "SELL", "positionSide": "BOTH"})
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "WATCH_EXITS")
        self.assertEqual(self.ex.broker.writes, [])

    def test_changed_position_quantity_blocks_all_mutations(self):
        self.ex.broker.positions_data[0]["positionAmt"] = "2"
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "WATCH_EXITS")
        self.assertEqual(self.ex.broker.writes, [])

    def test_disabled_feature_never_changes_orders(self):
        os.environ["CRASH_REPRICE_OLD_EXITS_ENABLED"] = "0"
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "WATCH_EXITS")
        self.assertEqual(self.ex.broker.writes, [])

    def test_new_target_cycle_not_repriced_again(self):
        self.ctrl.state["one_off_target"] = "2026-11-26:1027"
        self.ctrl.state["target"] = "1027"
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "WATCH_EXITS")
        self.assertEqual(self.ex.broker.writes, [])

    def test_uncertain_cancel_is_not_sent_twice(self):
        self.ctrl.step()
        original = self.ex.broker.request
        def uncertain(method, path, params=None, budget_priority=None):
            if method == "DELETE":
                original(method, path, params, budget_priority)
                raise TimeoutError("exchange response lost")
            return original(method, path, params, budget_priority)
        self.ex.broker.request = uncertain
        self.step()
        self.assertEqual(len([w for w in self.ex.broker.writes if w[0] == "DELETE"]), 1)
        self.assertEqual(self.ctrl.state["phase"], "REPRICE_OLD_EXITS")
        self.ex.broker.request = original
        self.step()
        self.assertEqual(len([w for w in self.ex.broker.writes if w[0] == "DELETE"]), 1)

    def test_unknown_post_does_not_duplicate(self):
        self.ctrl.step()
        self.step()  # cancel ETH
        original = self.ex.broker.request
        def failed_post(method, path, params=None, budget_priority=None):
            if method == "POST":
                raise TimeoutError("exchange POST status unknown")
            return original(method, path, params, budget_priority)
        self.ex.broker.request = failed_post
        self.step()
        self.assertEqual(self.ctrl.state["reprice_jobs"]["ETHUSDT|BOTH"]["status"],
                         "POST_UNCERTAIN")
        self.step()
        self.assertEqual(len([w for w in self.ex.broker.writes if w[0] == "POST"]), 0)

if __name__ == "__main__":
    import unittest
    unittest.main()
