"""Offline proof of opt-in one-shot profit closing; no live exchange calls."""
import os
from decimal import Decimal
from unittest.mock import patch
from test_profit_lock_live import LiveLockTests


class OneShotTests(LiveLockTests):
    def setUp(self):
        super().setUp()
        self.extra = patch.dict(os.environ, {"CRASH_PROFIT_ONE_SHOT_V2": "1"})
        self.extra.start()
        self.addCleanup(self.extra.stop)

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


if __name__ == "__main__":
    import unittest
    unittest.main()
