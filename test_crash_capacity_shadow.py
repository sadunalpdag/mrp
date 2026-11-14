import unittest

from crash_capacity_shadow import attach, advance, expire, summarize
from crash_policy import TRACKING_SECONDS, POST_ENTRY_TRACKING_SECONDS


def sample(status="WAIT_ENTRY"):
    rec = {
        "signal_id": "ABCUSDT-1000", "symbol": "ABCUSDT",
        "tracking_model": "ws_aggtrade_upcross_v3", "tracking_status": status,
        "created_at": 1000.0, "reference_low": 90.0,
        "entry": 100.0, "tp": 101.0, "previous_watch_price": 99.0,
    }
    if status == "OPEN":
        rec.update(entered_at=1001.0, entry_trade_id=10, ws_last_trade_id=10,
                   ws_last_event_at=1001.0)
    return rec


class CapacityShadowTests(unittest.TestCase):
    def test_risk_tier_tp_and_drawdown(self):
        rec = sample()
        self.assertTrue(attach(rec, "OPEN_LIMIT_REACHED",
                               {"price": "100", "tp": "100.5"}, "250", "0.005", 1002))
        self.assertFalse(attach(rec, "OPEN_LIMIT_REACHED",
                                {"price": "100", "tp": "100.5"}, "250", "0.005", 1002))
        self.assertEqual(advance(rec, 100, 1003, 1), "OPEN")
        self.assertIsNone(advance(rec, 99.5, 1004, 2))
        self.assertEqual(advance(rec, 100.6, 1005, 3), "TP")
        shadow = rec["capacity_virtual"]
        self.assertEqual(shadow["tracking_status"], "TP")
        self.assertLess(shadow["max_adverse_pct"], 0)
        stats = summarize([rec])
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["tp_rate_resolved_pct"], 100.0)
        self.assertAlmostEqual(stats["avg_tp_minutes"], 2 / 60, places=1)

    def test_entry_must_precede_tp(self):
        rec = sample()
        self.assertTrue(attach(rec, "PENDING_LIMIT_REACHED",
                               {"price": "100", "tp": "101"}, "500", "0.01", 1002))
        self.assertEqual(advance(rec, 101.2, 1003, 1), "MISSED_SIGNAL")
        self.assertNotEqual(rec["capacity_virtual"]["tracking_status"], "TP")

    def test_no_entry_is_not_counted_as_tp_failure(self):
        rec = sample()
        self.assertTrue(attach(rec, "PENDING_LIMIT_REACHED",
                               {"price": "100", "tp": "101"}, "500", "0.01", 1002))
        self.assertEqual(expire(rec, 1000 + TRACKING_SECONDS), "NO_ENTRY_OBSERVED")
        self.assertIsNone(summarize([rec])["tp_rate_resolved_pct"])

    def test_open_trade_timeout_and_no_duplicate(self):
        rec = sample("OPEN")
        self.assertTrue(attach(rec, "OPEN_LIMIT_REACHED",
                               {"price": "100", "tp": "101"}, "500", "0.01", 1002))
        self.assertEqual(expire(rec, 1001 + POST_ENTRY_TRACKING_SECONDS), "TIMEOUT")
        self.assertEqual(summarize([rec])["tp_rate_resolved_pct"], 0)

    def test_invalid_reason_is_noop(self):
        rec = sample()
        self.assertFalse(attach(rec, "AUTO_OFF",
                                {"price": "100", "tp": "101"}, "500", "0.01", 1002))
        self.assertNotIn("capacity_virtual", rec)


if __name__ == "__main__":
    unittest.main()
