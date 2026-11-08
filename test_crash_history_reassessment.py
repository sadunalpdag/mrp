"""Offline, zero-exchange side-effect regression tests for late history."""
import unittest
from unittest.mock import Mock
from crash_history_reassessment import reassess_expired


class ReassessmentTests(unittest.TestCase):
    def setUp(self):
        self.now = 2_000_001_000
        self.old = dict(symbol="TESTUSDT", signal_id="test-1",
            created_at=self.now-400, entry="1.00",
            v2_status="MISSING_5D_HISTORY",
            v2_reason="HISTORY_NOT_READY_IN_TIME",
            v2_delivery_phase="CANDIDATE", v2_delivery_status="EXPIRED_NOT_SENT",
            v2_sent=False, tracking_status="V2_FILTERED")
        self.state = dict(history=[self.old], symbols={"TESTUSDT": dict(self.old)})
        self.log = Mock()
        self.valid = dict(status="STRONG_V2", qualified=True, candles=480,
            missing_reason=None, strength_score={"metrics": {
                "green_volume_share_1h": 0.10,
                "volume_ratio_vs_prior_12h": 1.2,
                "rsi_delta_1h": -6,
                "ema25_slope_change_1h_pp": -0.09,
            }})

    def test_expired_candidate_research_recovers_risk_without_live_signal(self):
        check = Mock(return_value=self.valid)
        changed = reassess_expired(self.state, self.now, check, self.log)
        self.assertEqual(changed, 1)
        check.assert_called_once_with(self.old, self.old["created_at"])
        self.assertEqual(self.old["retrospective_v2"]["status"], "STRONG_V2")
        self.assertTrue(self.old["retrospective_v2"]["research_only"])
        self.assertEqual(self.old["failure_shadow"]["known_flags"], 3)
        self.assertEqual(self.old["failure_shadow"]["flag_count"], 2)
        self.assertEqual(self.state["symbols"]["TESTUSDT"]["failure_shadow"],
                         self.old["failure_shadow"])
        for key, expected in (("v2_status", "MISSING_5D_HISTORY"),
                              ("v2_delivery_status", "EXPIRED_NOT_SENT"),
                              ("v2_sent", False), ("tracking_status", "V2_FILTERED")):
            self.assertEqual(self.old[key], expected)
        self.assertEqual(reassess_expired(self.state, self.now+2, check, self.log), 0)
        self.assertEqual(check.call_count, 1)

    def test_missing_data_waits_and_does_not_invent_risk(self):
        check = Mock(side_effect=[
            dict(status="MISSING_5D_HISTORY", qualified=False,
                 candles=476, missing_reason="TOO_FEW_5D_CANDLES"),
            self.valid])
        self.assertEqual(reassess_expired(self.state, self.now, check, self.log), 1)
        self.assertNotIn("failure_shadow", self.old)
        self.assertEqual(reassess_expired(self.state, self.now+200, check, self.log), 0)
        self.assertEqual(reassess_expired(self.state, self.now+301, check, self.log), 1)
        self.assertEqual(self.old["failure_shadow"]["known_flags"], 3)

    def test_filtered_backfill_cannot_become_live_candidate(self):
        check = Mock(return_value={"status":"FILTERED_V2",
                                   "qualified":False,"candles":479})
        self.assertEqual(reassess_expired(self.state, self.now, check, self.log), 1)
        self.assertNotIn("failure_shadow", self.old)
        self.assertEqual(self.old["v2_status"], "MISSING_5D_HISTORY")
        self.assertFalse(self.old["v2_sent"])

    def test_filtered_decision_with_real_strength_metrics_gets_var_yok_labels(self):
        result = dict(self.valid, status="FILTERED_V2", qualified=False)
        validator = Mock(return_value=result)
        self.assertEqual(reassess_expired(self.state, self.now, validator, self.log), 1)
        self.assertEqual(self.old["retrospective_v2"]["status"], "FILTERED_V2")
        self.assertEqual(self.old["retrospective_v2"]["known_risk_flags"], 3)
        self.assertEqual(self.old["failure_shadow"]["known_flags"], 3)
        self.assertEqual(self.old["v2_status"], "MISSING_5D_HISTORY")
        self.assertFalse(self.old["v2_sent"])

    def test_previous_filtered_record_without_risk_is_backfilled(self):
        self.old["retrospective_v2"] = {
            "status": "FILTERED_V2", "checked_at": self.now - 600,
            "research_only": True}
        result = dict(self.valid, status="FILTERED_V2", qualified=False)
        self.assertEqual(reassess_expired(
            self.state, self.now, Mock(return_value=result), self.log), 1)
        self.assertEqual(self.old["failure_shadow"]["known_flags"], 3)
        self.assertEqual(self.old["v2_delivery_status"], "EXPIRED_NOT_SENT")

    def test_nonexpired_or_nonhistorical_candidates_are_not_modified(self):
        self.old["v2_status"] = "STRONG_V2"
        check = Mock(return_value=self.valid)
        self.assertEqual(reassess_expired(self.state, self.now, check, self.log), 0)
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
