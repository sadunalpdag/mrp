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


if __name__ == "__main__":
    import unittest
    unittest.main()
