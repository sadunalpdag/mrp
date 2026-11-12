"""Offline unit tests: no network, no real orders, no Telegram."""
import json
import os
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import daily_profit_lock_paper as p

PARIS = ZoneInfo("Europe/Paris")


def dt(day, hour=12):
    return datetime.fromisoformat(day + "T%02d:00:00" % hour).replace(tzinfo=PARIS)


class Reader:
    def __init__(self, eq, positions=None, bid="0", ask="100000"):
        self.eq = Decimal(str(eq))
        self.ps = list(positions or [])
        self.bid, self.ask = bid, ask
        self.position_calls = 0

    def equity(self):
        return self.eq, self.eq, Decimal("0")

    def positions(self):
        self.position_calls += 1
        return self.ps

    def best_book(self, symbol):
        return {"bid": Decimal(self.bid), "ask": Decimal(self.ask)}


class PaperTests(unittest.TestCase):
    def test_first_partial_day_bootstraps_without_fabricated_yesterday(self):
        state, triggered, rolled = p.tick(None, Reader(1000), dt("2026-11-12"))
        self.assertFalse(triggered)
        self.assertIsNone(state["target"])
        self.assertEqual(state["reference_status"], "BOOTSTRAP")
        self.assertEqual(state["all_time_peak"], "1000")

    def test_previous_peak_never_falls_and_threshold_freezes(self):
        st, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
        st, tr, rollover = p.tick(st, Reader(950), dt("2026-11-12"))
        self.assertTrue(rollover)
        self.assertFalse(tr)
        self.assertEqual(Decimal(st["target"]), Decimal("1050"))
        self.assertEqual(Decimal(st["all_time_peak"]), Decimal("1000"))
        st, _, _ = p.tick(st, Reader(920), dt("2026-11-13"))
        self.assertEqual(Decimal(st["target"]), Decimal("1050"))

    def test_1040_observed_under_1050_makes_next_day_1092(self):
        st, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
        st, *_ = p.tick(st, Reader(1000), dt("2026-11-12"))
        st, fired, _ = p.tick(st, Reader(1040), dt("2026-11-12", 18))
        self.assertFalse(fired)
        self.assertEqual(Decimal(st["target"]), Decimal("1050"))
        st, fired, _ = p.tick(st, Reader(940), dt("2026-11-13"))
        self.assertFalse(fired)
        self.assertEqual(Decimal(st["target"]), Decimal("1092"))

    def test_one_trigger_and_immutable_price_snapshot(self):
        st, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
        st, *_ = p.tick(st, Reader(1000), dt("2026-11-12"))
        pos = [{"symbol": "BTCUSDT", "positionSide": "BOTH",
                "side": "LONG", "qty": "1", "snapshot_price": "100",
                "hypothetical_close": "SELL LIMIT",
                "paper_status": "WAIT", "paper_fill_at": None}]
        reader = Reader(1050, pos, bid="90")
        st, fired, _ = p.tick(st, reader, dt("2026-11-12", 13))
        self.assertTrue(fired)
        self.assertEqual(reader.position_calls, 1)
        self.assertTrue(st["active_lock"])
        self.assertEqual(st["snapshots"][0]["positions"][0]["snapshot_price"], "100")
        st, fired, _ = p.tick(st, Reader(950, pos, bid="95"), dt("2026-11-12", 14))
        self.assertFalse(fired)
        self.assertEqual(st["active_lock"]["positions"][0]["snapshot_price"], "100")
        self.assertEqual(len(st["snapshots"]), 1)

    def test_quote_cross_paper_only_and_restart_history_consistent(self):
        st, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
        st, *_ = p.tick(st, Reader(1000), dt("2026-11-12"))
        pos = [{"symbol": "BTCUSDT", "positionSide": "BOTH",
                "side": "LONG", "qty": "1", "snapshot_price": "100",
                "hypothetical_close": "SELL LIMIT",
                "paper_status": "WAIT", "paper_fill_at": None}]
        st, *_ = p.tick(st, Reader(1050, pos, bid="90"), dt("2026-11-12", 13))
        st = json.loads(json.dumps(st))  # Simulate process restart.
        st, fired, _ = p.tick(st, Reader(990, pos, bid="101"), dt("2026-11-12", 14))
        self.assertFalse(fired)
        self.assertIsNone(st["active_lock"])
        self.assertEqual(st["snapshots"][-1]["positions"][0]["paper_status"],
                         "PAPER_QUOTE_CROSSED")
        self.assertTrue(st["snapshots"][-1]["all_paper_crossed"])

    def test_waiting_lock_persists_past_midnight(self):
        st, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
        st, *_ = p.tick(st, Reader(1000), dt("2026-11-12"))
        pos = [{"symbol": "BTCUSDT", "positionSide": "BOTH",
                "side": "LONG", "qty": "1", "snapshot_price": "100",
                "hypothetical_close": "SELL LIMIT",
                "paper_status": "WAIT", "paper_fill_at": None}]
        st, *_ = p.tick(st, Reader(1050, pos, bid="90"), dt("2026-11-12", 13))
        st, _, rolled = p.tick(st, Reader(950, pos, bid="85"), dt("2026-11-13", 1))
        self.assertTrue(rolled)
        self.assertIsNotNone(st["active_lock"])

    def test_telegram_target_is_persistent_and_only_tomorrow_applies(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(p, "STATE_FILE", Path(root) / "state.json"), \
                 patch.object(p, "CONFIG_FILE", Path(root) / "settings.json"):
                st, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
                st, *_ = p.tick(st, Reader(950), dt("2026-11-12"))
                p.persist(st)
                self.assertEqual(Decimal(st["target"]), Decimal("1050"))
                response = p.control("paper_target:7")
                self.assertIn("%7", response)
                settings = p.load_settings()
                self.assertEqual(settings["target_pct"], "7")
                self.assertEqual(Decimal(p.load_state()["target"]), Decimal("1050"))
                st, fired, _ = p.tick(st, Reader(940), dt("2026-11-13"),
                                       settings=p.load_settings())
                self.assertFalse(fired)
                self.assertEqual(Decimal(st["target"]), Decimal("1070"))
                self.assertEqual(st["current_pct"], "7")

    def test_telegram_off_disables_paper_trigger_but_keeps_peak(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(p, "STATE_FILE", Path(root) / "state.json"), \
                 patch.object(p, "CONFIG_FILE", Path(root) / "settings.json"):
                st, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
                st, *_ = p.tick(st, Reader(1000), dt("2026-11-12"))
                p.persist(st)
                self.assertIn("KAPALI", p.control("paper_off"))
                reader = Reader(1100)
                st, fired, _ = p.tick(st, reader, dt("2026-11-12", 13),
                                       settings=p.load_settings())
                self.assertFalse(fired)
                self.assertEqual(reader.position_calls, 0)
                self.assertEqual(st["all_time_peak"], "1100")
                p.persist(st)
                self.assertIn("AÇIK", p.control("paper_on"))
                self.assertIn("1100", p.control("paper_status"))

    def test_invalid_telegram_targets_do_not_change_settings(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(p, "STATE_FILE", Path(root) / "state.json"), \
                 patch.object(p, "CONFIG_FILE", Path(root) / "settings.json"):
                for target in ("0.1", "21", "-1", "NaN", "xyz", "1e100", "7%"):
                    self.assertIn("Kullanım", p.control("paper_target:" + target))
                    self.assertEqual(p.load_settings()["target_pct"], "5")
                self.assertIn("%7.5", p.control("paper_target:7,5"))
                self.assertEqual(p.load_settings()["target_pct"], "7.5")

    def test_signed_get_disallows_any_order_endpoint(self):
        reader = p.MarketReader(session=object())
        with self.assertRaisesRegex(ValueError, "READ_ONLY_ENDPOINT"):
            reader.signed_get("/fapi/v1/order", {"symbol": "BTCUSDT"})

    def test_persistent_json_round_trip(self):
        with tempfile.TemporaryDirectory() as root, patch.object(p, "STATE_FILE", Path(root)/"paper.json"):
            original, *_ = p.tick(None, Reader(1000), dt("2026-11-11"))
            p.persist(original)
            self.assertEqual(p.load_state(), original)


if __name__ == "__main__":
    unittest.main()
