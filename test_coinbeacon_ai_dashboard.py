"""Offline regression tests: independent Gemini J:O settings, never Crash D:I."""
import os
import tempfile
import unittest
from unittest.mock import patch

import coinbeacon_ai_dashboard as dashboard
import coinbeacon_ai_settings as config


class MockTab:
    title = "Bot Status"
    id = 11
    col_count = 15

    def __init__(self, rows=None):
        self.rows = rows or []

    def get(self, area):
        assert area == "J1:O8"
        return self.rows


class MockWorkbook:
    def __init__(self):
        self.writes = []

    def values_batch_update(self, request):
        self.writes.extend(request["data"])

    def batch_update(self, request):
        raise AssertionError("Grid resize unexpected")


class GeminiDashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "DATA_DIR": self.temp.name, "CB_AI_LEARNING_ENABLED": "0",
            "CB_AI_DAILY_GEMINI_CALL_CAP": "100", "CB_AI_MAX_ACTIVE": "30",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def rows(self):
        return [dashboard.HEADER] + [
            [label, "", "", "", "", ""] for _, label, _ in dashboard.ROWS
        ] + [[dashboard.DAILY_LABEL, "", "", "", "", ""]]

    def test_daily_counter_is_read_only(self):
        tab = MockTab(self.rows())
        sheet = MockWorkbook()
        with patch("coinbeacon_ai_learning.daily_dispatch_stats", return_value={
                "coins": 3, "signals": 5, "start": "2026-11-17 01:00",
                "end": "2026-11-18 01:00"}):
            self.assertEqual(dashboard.sync_panel(sheet, tab), 0)
        left = next(x for x in sheet.writes if x["range"].endswith("J1:K8"))
        self.assertEqual(left["values"][-1], [dashboard.DAILY_LABEL, "3 coin / 5 sinyal"])
        self.assertEqual(config.load_settings()["enabled"],False)

    def test_initialization_only_j_to_o_does_not_toggle(self):
        sheet = MockWorkbook()
        self.assertEqual(dashboard.sync_panel(sheet, MockTab()), 0)
        self.assertEqual(config.load_settings()["enabled"], False)
        self.assertTrue(all(item["range"].startswith("'Bot Status'!J") or
                            item["range"].startswith("'Bot Status'!K") or
                            item["range"].startswith("'Bot Status'!L") or
                            item["range"].startswith("'Bot Status'!M") or
                            item["range"].startswith("'Bot Status'!N") or
                            item["range"].startswith("'Bot Status'!O")
                            for item in sheet.writes))

    def test_config_persists_without_mutating_crash_flags(self):
        tab = MockTab(self.rows())
        tab.rows[2][2] = "125"
        tab.rows[2][5] = "UYGULA"
        sheet = MockWorkbook()
        self.assertEqual(dashboard.sync_panel(sheet, tab), 1)
        self.assertEqual(config.load_settings()["daily_call_cap"], 125)
        self.assertEqual(config.load_settings()["max_active"], 30)
        self.assertEqual(os.environ["CB_AI_LEARNING_ENABLED"], "0")
        self.assertTrue(any(w["range"].endswith("O3") for w in sheet.writes))

    def test_on_refuses_missing_api_key(self):
        tab = MockTab(self.rows())
        tab.rows[1][2] = "AÇIK"
        tab.rows[1][5] = "UYGULA"
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            sheet = MockWorkbook()
            self.assertEqual(dashboard.sync_panel(sheet, tab), 0)
            self.assertFalse(config.load_settings()["enabled"])

    def test_input_validation_and_disk_settings(self):
        with self.assertRaises(ValueError):
            config.update_setting("track_hours", 25)
        with self.assertRaises(ValueError):
            config.update_setting("check_minutes", 5)
        with self.assertRaises(ValueError):
            config.update_setting("enabled", 1)
        config.update_setting("track_hours", 6)
        config.update_setting("check_minutes", 30)
        config.update_setting("analysis_phase", "initial")
        self.assertEqual(config.load_settings()["track_hours"], 6)
        self.assertEqual(config.load_settings()["check_minutes"], 30)
        self.assertEqual(config.load_settings()["analysis_phase"], "initial")

    def test_enable_starts_only_paper_observer(self):
        tab = MockTab(self.rows())
        tab.rows[1][2] = "AÇIK"
        tab.rows[1][5] = "UYGULA"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key",
                                     "CB2_LIVE_MOMENTUM_ENABLED": "1"}):
            with patch("coinbeacon_ai_learning.start", return_value=True) as launch:
                self.assertEqual(dashboard.sync_panel(MockWorkbook(), tab), 1)
                launch.assert_called_once_with()
        self.assertTrue(config.load_settings()["enabled"])


if __name__ == "__main__":
    unittest.main()
