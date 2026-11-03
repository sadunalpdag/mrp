"""Regression smoke tests against the currently deployed production commit.

Offline only: never submits an exchange order.
"""
import tempfile
import unittest
from unittest.mock import Mock
from crash_night import NightExecutor
from test_crash_services import FakeBroker
from research_archive import append_research_runtime
import os
import json


class DeploymentSmokeTests(unittest.TestCase):
    def test_saved_tp_halt_survives_restart_and_blocks_new_entry(self):
        with tempfile.TemporaryDirectory() as path:
            broker = FakeBroker()
            one = NightExecutor(path, broker, Mock(return_value={"ok": True}), Mock(),
                                clock=lambda: 1793669000., enabled=True)
            one.state["halt"] = "TP DOĞRULANAMADI\\nKod: TP_NOT_ACTIVE"
            one.save()
            one.guard.close()
            two = NightExecutor(path, broker, Mock(return_value={"ok": True}), Mock(),
                                clock=lambda: 1793669001., enabled=True)
            try:
                self.assertTrue(two.state["halt"])
                two.enter(dict(signal_id="new", symbol="AUSDT", v2_status="STRONG_V2",
                               v2_sent=True, tracking_status="OPEN", entry="10", tp="10.1"),
                          1793669001.)
                self.assertFalse(broker.buys)
                self.assertIn("Açılamadı", two.apply_control("on"))
            finally:
                two.guard.close()

    def test_unlock_empty_exchange_does_not_arm_new_trades(self):
        with tempfile.TemporaryDirectory() as path:
            broker = FakeBroker()
            broker.positions = Mock(return_value=[])
            broker.open_entry_orders = Mock(return_value=[])
            broker.open_algo_orders = Mock(return_value=[])
            ex = NightExecutor(path, broker, Mock(return_value={"ok": True}), Mock(),
                               clock=lambda: 1793669001., enabled=True)
            try:
                ex.state["halt"] = "TP DOĞRULANAMADI\\nKod: TP_NOT_ACTIVE"
                status = ex.apply_control("unlock")
                self.assertIn("KAPALI", status)
                self.assertIsNone(ex.state["halt"])
                self.assertFalse(ex.enabled.is_set())
                self.assertTrue(ex.state["paused"])
                self.assertFalse(broker.buys)
            finally:
                ex.guard.close()

    def test_research_event_archive_appends_without_truncation(self):
        with tempfile.TemporaryDirectory() as path:
            self.assertTrue(append_research_runtime(path, "[CRASH NIGHT] started", 1793669000.))
            self.assertTrue(append_research_runtime(path, "[CRASH NIGHT] resumed", 1793669001.))
            folder = os.path.join(path, "research_archive", "runtime")
            files = os.listdir(folder)
            self.assertEqual(len(files), 1)
            with open(os.path.join(folder, files[0]), encoding="utf-8") as f:
                events = [json.loads(line) for line in f]
            self.assertEqual(len(events), 2)
            self.assertNotEqual(events[0]["message"], events[1]["message"])


if __name__ == "__main__":
    unittest.main()
