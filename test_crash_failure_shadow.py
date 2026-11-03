"""Offline prospective TP-risk label and automatic BUY gate regression tests."""
import csv
import io
import tempfile
import unittest
from unittest.mock import Mock

from crash_failure_shadow import (
    VERSION, assess, entry_block_reason, export_csv, format_warning, summarize,
)
from crash_night import NightExecutor
from test_crash_services import FakeBroker, timestamp


def candidate(symbol="AUSDT", at=1_000_000.0, entry="10",
              green=0.50, volume=2.0, rsi=1.0, ema=0.04):
    return {
        "symbol": symbol, "signal_id": symbol + ":" + str(at),
        "created_at": at, "entry": entry, "tp": "10.10",
        "v2_status": "STRONG_V2", "v2_sent": True,
        "tracking_status": "WAIT_ENTRY",
        "tracking_model": "ws_aggtrade_upcross_v3",
        "v2_features": {"strength_score": {
            "score": 45.0, "metrics": {
                "green_volume_share_1h": green,
                "volume_ratio_vs_prior_12h": volume,
                "rsi_delta_1h": rsi,
                "ema25_slope_change_1h_pp": ema,
            }
        }}
    }


class NonclosureLabelsTests(unittest.TestCase):
    def risk(self, record, earlier=()):
        shadow = assess(record, earlier)
        record["failure_shadow"] = shadow
        return shadow

    def test_clean_signal_permitted(self):
        r = candidate()
        shadow = self.risk(r)
        self.assertEqual(shadow["flag_count"], 0)
        self.assertEqual(shadow["known_flags"], 3)
        self.assertFalse(shadow["blocks_live_entry"])
        self.assertIsNone(entry_block_reason(r))

    def test_weak_green_share_and_volume_blocks_auto_buy(self):
        r = candidate(green=.17, volume=1.27, rsi=2, ema=.02)
        shadow = self.risk(r)
        self.assertEqual(shadow["flags"], {
            "weak_volume": True, "falling_rsi_ema": False,
            "repeat_lower_entry": False})
        self.assertEqual(shadow["flag_count"], 1)
        self.assertIn("RISK_weak_volume", entry_block_reason(r))
        self.assertIn("GERÇEK ALIŞ YAPILMAYACAK", format_warning(shadow))

    def test_falling_rsi_and_ema_blocks_auto_buy(self):
        r = candidate(rsi=-6, ema=-.11)
        shadow = self.risk(r)
        self.assertTrue(shadow["flags"]["falling_rsi_ema"])
        self.assertIsNotNone(entry_block_reason(r))

    def test_repeat_lower_prior_strong_signal_only(self):
        older = candidate(symbol="AUSDT", at=1_000_000-3600, entry="11")
        r = candidate()
        shadow = self.risk(r, [older])
        self.assertTrue(shadow["flags"]["repeat_lower_entry"])
        self.assertLess(shadow["metrics"]["repeat_prior_entry_drop_pct"], -2)
        self.assertIn("repeat_lower_entry", entry_block_reason(r))
        older["created_at"] = 1_000_001
        self.assertFalse(assess(r, [older])["flags"]["repeat_lower_entry"])

    def test_missing_data_is_unknown_not_clean(self):
        r = candidate(volume=None)
        shadow = self.risk(r)
        self.assertIsNone(shadow["flags"]["weak_volume"])
        self.assertEqual(shadow["known_flags"], 2)
        self.assertEqual(entry_block_reason(r), "RISK_METRICS_INCOMPLETE")
        self.assertIn("EKSİK VERİ", format_warning(shadow))
        self.assertEqual(entry_block_reason(candidate()),
                         "RISK_ASSESSMENT_MISSING")

    def test_no_future_leak_and_outcomes_have_separate_counts(self):
        r = candidate()
        self.risk(r)
        r.update(tracking_status="OPEN", entered_at=1_000_010)
        tp = candidate(symbol="BUSDT")
        self.risk(tp)
        tp.update(tracking_status="TP", entered_at=1_000_010)
        timeout = candidate(symbol="CUSDT", green=.1, volume=1.0)
        self.risk(timeout)
        timeout.update(tracking_status="TIMEOUT", entered_at=1_000_020)
        result = summarize([r, tp, timeout])
        self.assertEqual(result["groups"]["all"]["signals"], 3)
        self.assertEqual(result["groups"]["all"]["tp"], 1)
        self.assertEqual(result["groups"]["all"]["timeout"], 1)
        self.assertEqual(result["groups"]["all"]["open_unresolved"], 1)
        self.assertEqual(result["groups"]["weak_volume"]["timeout"], 1)
        rows = list(csv.DictReader(io.StringIO(export_csv([r, tp, timeout]))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["weak_volume"], "True")

    def test_entry_safety_guard_blocks_post_and_does_not_count_risky(self):
        with tempfile.TemporaryDirectory() as directory:
            broker = FakeBroker()
            clock = timestamp("2026-11-03T20:00:00")
            executor = NightExecutor(directory, broker, Mock(return_value={"ok":True}),
                                     Mock(), lambda:clock, enabled=True, max_daily=20)
            self.addCleanup(executor.guard.close)
            r = candidate()
            r["signal_id"] = "risk-1"
            r["created_at"] = clock - 1
            self.risk(r)
            clean = dict(r)
            clean["signal_id"] = "clean-1"
            risky = candidate(green=.1, volume=1.1)
            risky.update(created_at=clock-1, signal_id="risky-2")
            self.risk(risky)
            executor.enter(risky, clock)
            self.assertEqual(len(broker.buys), 0)
            self.assertEqual(executor.daily_count(), 0)
            executor.enter(clean, clock)
            self.assertEqual(len(broker.buys), 1)
            self.assertEqual(executor.daily_count(), 1)

    def test_missing_assessment_blocks_live_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            broker = FakeBroker()
            clock = timestamp("2026-11-03T20:00:00")
            executor = NightExecutor(directory, broker, Mock(return_value={"ok":True}),
                                     Mock(), lambda:clock, enabled=True)
            self.addCleanup(executor.guard.close)
            r = candidate()
            r["signal_id"] = "old-unassessed"
            executor.enter(r, clock)
            self.assertEqual(len(broker.buys), 0)
            self.assertEqual(executor.daily_count(), 0)


if __name__ == "__main__":
    unittest.main()
