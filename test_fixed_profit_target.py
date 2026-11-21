import unittest
from datetime import datetime, timedelta, timezone

from fixed_profit_target import (
    initial_state, observe, closing_verified, roll_after_midnight
)

def at(day, hour=12):
    return datetime(2026, 11, day, hour, tzinfo=timezone.utc)

class FixedProfitTargetTests(unittest.TestCase):
    def test_missed_target_does_not_move_across_days(self):
        s = initial_state("800", "10", at(20))
        for day, equity in [(20, "870"), (21, "850"), (22, "879")]:
            s, hit = observe(s, equity, at(day))
            self.assertFalse(hit)
            self.assertEqual(s["target"], "880.0")
        s, hit = observe(s, "880", at(23))
        self.assertTrue(hit)
        self.assertEqual(s["phase"], "CLOSING")
        s, hit = observe(s, "890", at(23))
        self.assertFalse(hit)

    def test_incomplete_close_survives_midnight(self):
        s, _ = observe(initial_state(800, 10, at(20)), 880, at(20))
        s = closing_verified(s, True, False, True, at(20))
        self.assertEqual(s["phase"], "CLOSING")
        s = roll_after_midnight(s, 890, 10, at(21), True, True, True)
        self.assertEqual(s["phase"], "CLOSING")

    def test_completed_cycle_immediate_rebase(self):
        s, _ = observe(initial_state(800, 10, at(20)), 880, at(20))
        s = closing_verified(s, True, True, True, at(20))
        self.assertEqual(s["phase"], "WAIT_REBASE")
        s = roll_after_midnight(s, 890, 10, at(20), True, True, True)
        self.assertEqual(s["target"], "979.0")
        self.assertEqual(s["phase"], "WAIT_TARGET")

    def test_rebase_requires_all_clear(self):
        s, _ = observe(initial_state(800, 10, at(20)), 880, at(20))
        s = closing_verified(s, True, True, True, at(20))
        blocked = roll_after_midnight(s, 890, 10, at(20), False, True, True)
        self.assertEqual(blocked["phase"], "WAIT_REBASE")

    def test_no_automatic_one_percent_rebase(self):
        s = initial_state(800, 10, at(20))
        s, _ = observe(s, 810, at(21))
        self.assertEqual(s["pct"], "10")
        self.assertEqual(s["target"], "880.0")

    def test_rejects_invalid_and_naive_inputs(self):
        with self.assertRaises(ValueError):
            initial_state(800, 10, datetime(2026, 11, 20))
        with self.assertRaises(ValueError):
            initial_state("NaN", 10, at(20))

if __name__ == "__main__":
    unittest.main()
