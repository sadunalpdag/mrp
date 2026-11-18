"""Offline safety, look-ahead, and accounting checks for BTC paper range observer."""
import math
import os
import unittest
from unittest.mock import patch

import btc_range_paper as m


def oscillating_window():
    bars = []
    for i in range(96):
        price = 101 + math.cos(i * math.pi / 8)
        bars.append([i * m.FIFTEEN, price, price + .08, price - .08, price, 10.0])
    bars[-1] = [95 * m.FIFTEEN, 100.12, 100.2, 100.0, 100.12, 10.0]
    return bars


def record(created, plan):
    return {"timestamp_paris": "2026-11-18T12:00:00+01:00",
            "range_paper": dict(plan, created_ms=created, status="PENDING",
                                last_checked_open_ms=created - 1)}


class RangePaperTests(unittest.TestCase):
    def test_range_near_support_generates_only_virtual_limit(self):
        snapshot = m.detect_range(oscillating_window(), 12)
        self.assertEqual(snapshot["regime"], "RANGE")
        self.assertEqual(snapshot["candidate"]["side"], "PAPER_LONG")
        self.assertGreater(snapshot["candidate"]["target"], snapshot["candidate"]["entry"])

    def test_high_adx_suppresses_setup(self):
        snapshot = m.detect_range(oscillating_window(), 31)
        self.assertEqual(snapshot["regime"], "NO_RANGE")
        self.assertNotIn("candidate", snapshot)

    def test_inadequate_data_suppresses_setup(self):
        self.assertEqual(m.detect_range(oscillating_window()[:40], 12)["regime"],
                         "INSUFFICIENT_DATA")

    def test_plan_cannot_be_filled_by_precreation_candle(self):
        created = 10 * m.FIFTEEN
        plan = {"entry": 100, "target": 102, "invalidation": 99.5}
        records = {"a": record(created, plan)}
        bars = [[created - m.FIFTEEN, 100, 103, 99, 101, 20]]
        self.assertEqual(m.evaluate_positions(records, bars), set())
        self.assertEqual(records["a"]["range_paper"]["status"], "PENDING")

    def test_entry_then_target_only_on_later_candle(self):
        created = 10 * m.FIFTEEN
        plan = {"entry": 100, "target": 102, "invalidation": 99.5}
        records = {"a": record(created, plan)}
        bars = [[created, 100.5, 103, 99.9, 102, 20],
                [created + m.FIFTEEN, 101, 102.1, 100.1, 102, 15]]
        m.evaluate_positions(records, bars)
        p = records["a"]["range_paper"]
        self.assertEqual(p["status"], "CLOSED")
        self.assertEqual(p["result"], "TARGET")
        self.assertAlmostEqual(p["net_pct"], 1.92)
        self.assertEqual(m.evaluate_positions(records, bars), set())

    def test_ambiguous_candle_chooses_invalidation(self):
        created = 10 * m.FIFTEEN
        plan = {"entry": 100, "target": 102, "invalidation": 99.5}
        records = {"a": record(created, plan)}
        bars = [[created, 100.1, 103, 99.4, 102, 20]]
        m.evaluate_positions(records, bars)
        self.assertEqual(records["a"]["range_paper"]["result"], "INVALIDATION")
        self.assertLess(records["a"]["range_paper"]["net_pct"], 0)

    def test_pending_expires_after_six_hours(self):
        created = 10 * m.FIFTEEN
        plan = {"entry": 100, "target": 102, "invalidation": 99.5}
        records = {"a": record(created, plan)}
        bars = [[created + 6 * m.HOUR, 101, 103, 100.5, 102, 15]]
        m.evaluate_positions(records, bars)
        self.assertEqual(records["a"]["range_paper"]["status"], "EXPIRED")
        self.assertNotIn("net_pct", records["a"]["range_paper"])

    def test_reference_is_30_days_of_15m_and_no_future(self):
        self.assertEqual(m.REFERENCE_BARS, 2880)
        bars = [[i * m.FIFTEEN, 100, 101, 99, 100, 10] for i in range(2880)]
        compact = m._compact(bars)
        self.assertEqual(len(compact), 2880)
        self.assertEqual(compact[-1][0], 2879)

    def test_hourly_step_never_accesses_private_order_api(self):
        # Only public 15m candles can produce a paper plan.
        now = 192 * m.FIFTEEN
        prior = oscillating_window()
        bars = [[i * m.FIFTEEN] + prior[i % 96][1:] for i in range(192)]
        record_now = {
            "timestamp_paris": "2026-11-18T12:00:00+01:00",
            "indicators": {"adx_1h": 12},
        }
        records = {"now": record_now}
        with patch.object(m, "_fetch_15m", return_value=bars), \
             patch.object(m, "ensure_reference", return_value={
                 "created_ms": 0, "gemini": {"status": "DISABLED"}}):
            m.step(records, now, lambda: False, record_now)
        self.assertEqual(record_now["range_paper"]["status"], "PENDING")
        self.assertEqual(record_now["range_paper"]["source"], "BINANCE_CLOSED_CANDLES")
        self.assertEqual(record_now["range_observation"]["virtual_action"],
                         "PAPER_LIMIT_BUY")


if __name__ == "__main__":
    unittest.main()
