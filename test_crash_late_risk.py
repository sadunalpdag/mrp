"""Regression checks: late data can repair reporting, never authorize an order."""
import unittest
from unittest.mock import Mock

import crash_late_risk as late
import crash_sheets_dashboard as sheets
from crash_failure_shadow import VERSION, entry_block_reason


class LateRiskTests(unittest.TestCase):
    def setUp(self):
        late._NEXT_CHECK.clear()
        self.created = 1_794_000_000.
        self.record = {
            "symbol": "TESTUSDT", "signal_id": "signal-1",
            "created_at": self.created, "entry": "1",
            "v2_status": "MISSING_5D_HISTORY",
        }
        self.state = {"history": [self.record],
                      "symbols": {"TESTUSDT": dict(self.record)}}
        self.result = {"status": "STRONG_V2", "qualified": True, "candles": 480,
                       "strength_score": {"metrics": {
                           "green_volume_share_1h": .4,
                           "volume_ratio_vs_prior_12h": 2.,
                           "rsi_delta_1h": 1.,
                           "ema25_slope_change_1h_pp": .02,
                       }}}
        self.validator = Mock(return_value=self.result)
        self.save = Mock()
        self.log = Mock()

    def refresh(self, offset=500):
        return late.refresh_late_risk(self.state, self.created+offset,
                                      self.validator, self.save, self.log)

    def test_late_history_fills_three_display_flags_without_authorizing_buy(self):
        self.assertEqual(self.refresh(), 1)
        self.validator.assert_called_once_with(self.record, self.created)
        self.assertEqual(self.record["late_failure_shadow"]["known_flags"], 3)
        self.assertNotIn("failure_shadow", self.record)
        self.assertEqual(entry_block_reason(self.record), "RISK_ASSESSMENT_MISSING")
        self.assertEqual(self.record["v2_status"], "MISSING_5D_HISTORY")
        self.assertEqual(self.state["symbols"]["TESTUSDT"]["late_failure_shadow"]["known_flags"], 3)
        self.save.assert_called_once_with(self.state)
        count, flags, rule = sheets._risk(self.record)
        self.assertEqual(count, 0)
        self.assertEqual(flags, ["🟢 YOK"] * 3)
        self.assertIn("OTOMATİK ALIŞ YOK", rule)

    def test_backfilled_live_candidate_cannot_become_green_or_enter(self):
        self.record["v2_status"] = "STRONG_V2"
        self.refresh()
        risk, _, _ = sheets._risk(self.record)
        self.assertEqual(risk, 0)
        self.assertEqual(sheets._status(self.record, risk)[1], "red")
        self.assertEqual(entry_block_reason(self.record), "RISK_ASSESSMENT_MISSING")

    def test_existing_complete_live_risk_never_changed(self):
        self.record["v2_status"] = "STRONG_V2"
        self.record["failure_shadow"] = {"version": VERSION, "flags": {
            "weak_volume": False, "falling_rsi_ema": True,
            "repeat_lower_entry": False}}
        self.assertEqual(self.refresh(), 0)
        self.validator.assert_not_called()
        self.save.assert_not_called()
        self.assertNotIn("late_failure_shadow", self.record)

    def test_not_ready_is_retried_without_fabricating_results(self):
        self.validator.return_value = {"status": "MISSING_5D_HISTORY", "candles": 477}
        self.assertEqual(self.refresh(), 0)
        self.assertEqual(self.refresh(offset=510), 0)
        self.assertEqual(self.validator.call_count, 1)
        self.assertNotIn("late_failure_shadow", self.record)
        self.assertEqual(self.refresh(offset=801), 0)
        self.assertEqual(self.validator.call_count, 2)
        self.save.assert_not_called()

    def test_old_candidate_does_not_reopen_order_or_backfill(self):
        self.assertEqual(self.refresh(offset=86401), 0)
        self.validator.assert_not_called()

    def test_each_known_flag_survives_one_unknown_flag(self):
        self.record["failure_shadow"] = {"version": VERSION,
                                         "flags": {"weak_volume": True}}
        count, flags, rule = sheets._risk(self.record)
        self.assertIsNone(count)
        self.assertEqual(flags, ["🔴 VAR", "?", "?"])
        self.assertIn("EKSİK", rule)
        # The historical value remains authoritative over later display data.
        self.refresh()
        count, flags, rule = sheets._risk(self.record)
        self.assertEqual(count, 1)
        self.assertEqual(flags, ["🔴 VAR", "🟢 YOK", "🟢 YOK"])
        self.assertIn("SONRADAN", rule)
        self.assertEqual(entry_block_reason(self.record), "RISK_METRICS_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
