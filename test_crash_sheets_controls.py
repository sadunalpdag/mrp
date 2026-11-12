"""Regression tests: spreadsheet is never an implicit live-trading instruction."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from decimal import Decimal

import crash_sheets_controls as ctrl


class FakeExecutor:
    def __init__(self):
        self.enabled = Mock()
        self.enabled.is_set.return_value = False
        self.state = {"paused": True}
        self.max_daily = 20
        self.position_usdt = Decimal("500")
        self.max_symbols = 20
        self.max_open_positions = 10
        self.max_pending_entries = 15
        self.manual_tp_enabled = False
        self.calls = []

    def request_control(self, action):
        self.calls.append(action)
        if action == "on":
            self.enabled.is_set.return_value = True
            self.state["paused"] = False
        elif action == "off":
            self.enabled.is_set.return_value = False
            self.state["paused"] = True
        else:
            key, value = action.split(":")
            attribute = {"limit":"max_daily", "amount":"position_usdt",
                         "symbols":"max_symbols", "open_limit":"max_open_positions",
                         "pending_limit":"max_pending_entries"}[key]
            setattr(self, attribute, Decimal(value) if key == "amount" else int(Decimal(value)))
        return "✅ Done"


class PanelTests(unittest.TestCase):
    def test_first_sync_discards_stale_approvals_without_executing(self):
        tab = SimpleNamespace(id=7, title="Bot Status", col_count=2,
                              get=Mock(return_value=[]))
        sheet = SimpleNamespace(batch_update=Mock(), values_batch_update=Mock())
        with patch.object(ctrl, "_values", return_value={}), patch.object(ctrl, "_apply") as apply:
            result = ctrl.sync_panel(sheet, tab, FakeExecutor())
        self.assertEqual(result, 0)
        apply.assert_not_called()
        ranges = [d["range"] for d in sheet.values_batch_update.call_args.args[0]["data"]]
        self.assertIn("'Bot Status'!F2:F15", ranges)
        self.assertIn("'Bot Status'!I2:I15", ranges)
        self.assertIn("'Bot Status'!F1", ranges)
        self.assertIn("'Bot Status'!I1", ranges)
        self.assertFalse(any("A1" in r or "B1" in r for r in ranges))

    def test_existing_panel_only_processes_approved_change(self):
        cells = [list(ctrl.HEADER)] + [[label, "old", "", description, "", ""]
                     for _, label, description, _ in ctrl.ROWS]
        # One desired value without confirmation must not run.
        cells[3][2] = "7"
        cells[4][2] = "15"
        cells[4][5] = "UYGULA"
        tab = SimpleNamespace(id=7, title="Bot Status", col_count=9,
                              get=Mock(return_value=cells))
        sheet = SimpleNamespace(batch_update=Mock(), values_batch_update=Mock())
        executor = FakeExecutor()
        with patch.object(ctrl, "_values", return_value={}):
            applied = ctrl.sync_panel(sheet, tab, executor)
        self.assertEqual(applied, 1)
        self.assertEqual(executor.calls, ["limit:15"])
        data = sheet.values_batch_update.call_args.args[0]["data"]
        cleared = {d["range"] for d in data}
        self.assertIn("'Bot Status'!I5", cleared)
        self.assertIn("'Bot Status'!F5", cleared)
        self.assertNotIn("'Bot Status'!F4", cleared)
        self.assertNotIn("'Bot Status'!I4", cleared)

    def test_raising_risk_requires_risk_onay(self):
        executor = FakeExecutor()
        with self.assertRaisesRegex(ValueError, "RISK ONAY"):
            ctrl._apply("AMOUNT", "1000", "UYGULA", executor)
        self.assertEqual(executor.calls, [])
        result = ctrl._apply("AMOUNT", "1000", "RISK ONAY", executor)
        self.assertTrue(result[0])
        self.assertEqual(executor.position_usdt, Decimal("1000"))

    def test_lowering_risk_only_needs_normal_confirmation(self):
        executor = FakeExecutor()
        self.assertTrue(ctrl._apply("OPEN", "5", "UYGULA", executor)[0])
        self.assertEqual(executor.max_open_positions, 5)

    def test_live_on_requires_two_guards_and_existing_queue(self):
        executor = FakeExecutor()
        with patch.dict(os.environ, {"CRASH_SHEETS_ALLOW_LIVE_ON": "0"}):
            with self.assertRaisesRegex(ValueError, "CANLI ONAY"):
                ctrl._apply("LIVE", "AÇIK", "UYGULA", executor)
            with self.assertRaisesRegex(ValueError, "Render"):
                ctrl._apply("LIVE", "AÇIK", "CANLI ONAY", executor)
        self.assertEqual(executor.calls, [])
        with patch.dict(os.environ, {"CRASH_SHEETS_ALLOW_LIVE_ON": "1"}):
            self.assertTrue(ctrl._apply("LIVE", "AÇIK", "CANLI ONAY", executor)[0])
        self.assertEqual(executor.calls, ["on"])
        self.assertTrue(ctrl._apply("LIVE", "KAPALI", "UYGULA", executor)[0])
        self.assertEqual(executor.calls, ["on", "off"])

    def test_bad_numbers_and_uneditable_counter(self):
        executor = FakeExecutor()
        for invalid in ("NaN", "Infinity", "51", "-1", "1.5", "=1+5"):
            with self.assertRaises(ValueError):
                ctrl._apply("DAILY", invalid, "RISK ONAY", executor)
        with self.assertRaisesRegex(ValueError, "sadece okunur"):
            ctrl._apply("COUNTER", "0", "UYGULA", executor)
        self.assertEqual(executor.calls, [])

    def test_no_executor_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "bulunamadi"):
            ctrl._apply("LIVE", "KAPALI", "UYGULA", None)

    def test_paper_target_updates_persisted_settings_only(self):
        settings = {"enabled": True, "target_pct": "5"}
        def control(action):
            if action.startswith("paper_target:"):
                settings["target_pct"] = action.split(":")[1]
        import sys
        fake = SimpleNamespace(control=control, load_settings=lambda: settings)
        with patch.dict(sys.modules, {"daily_profit_lock_paper": fake}):
            self.assertTrue(ctrl._apply("TARGET", "7,5%", "UYGULA", None)[0])
        self.assertEqual(settings["target_pct"], "7.5")


if __name__ == "__main__":
    unittest.main()
