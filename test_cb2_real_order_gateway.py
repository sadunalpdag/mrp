"""Offline regression tests for the isolated CB2 marketable LIMIT preflight."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cb2_real_order_gateway import (PreflightBlocked, gate_shared_capacity,
    make_read_only_proposal, plan_ask_limit, read_crash_snapshot,
    validate_fresh_momentum)


def market(symbol="TESTUSDT"):
    return {"symbol": symbol, "status": "TRADING", "contractType": "PERPETUAL",
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01",
                         "minPrice": "0.01", "maxPrice": "999999"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001",
                         "minQty": "0.001", "maxQty": "100000"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"}]}


class CB2OrderGatewayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = datetime(2026, 11, 17, 4, 30, tzinfo=timezone.utc).timestamp()
        self.day = datetime.fromtimestamp(self.now).date().isoformat()
        self.state = {"halt": None, "paused": False, "orders": {},
            "trade_settings": {"max_daily": 50, "max_symbols": 28,
                               "max_open_positions": 11,
                               "max_pending_entries": 14, "position_usdt": "314"}}
        self._write_state()
        self.item = {
            "id": "a1", "symbol": "TESTUSDT", "sqs": 78,
            "observed_at": datetime.fromtimestamp(self.now-20, timezone.utc).isoformat(),
            "filters_passed": True,
            "indicators": {"adx14": 22, "change24_pct": 11,
                           "macd_histogram": 0.04}}
    def tearDown(self):
        self.tmp.cleanup()
    def _write_state(self):
        (self.root / "crash_night_state.json").write_text(json.dumps(self.state))
    def test_exact_ask_rounded_up_to_tick_no_pullback(self):
        plan = plan_ask_limit("TESTUSDT", "123.456", market(), 314)
        self.assertEqual(plan["price"], "123.46")
        self.assertEqual(plan["target_tp"], "124.70")
        self.assertEqual(plan["quantity"], "2.543")
        self.assertEqual(plan["timeInForce"], "GTC")
        self.assertEqual(plan["mode"], "PREFLIGHT_ONLY_NO_BINANCE_ORDER")
    def test_plan_accepts_real_current_limits_without_order(self):
        p = make_read_only_proposal(self.item, self.now, self.root, [],
                                     set(), set(), "123.456", market())
        self.assertEqual(p["daily_usage_before"], 0)
        self.assertEqual(p["max_daily"], 50)
        self.assertEqual(p["price"], "123.46")
    def test_reject_stale_and_failed_filters(self):
        self.item["observed_at"] = datetime.fromtimestamp(
            self.now-901, timezone.utc).isoformat()
        with self.assertRaisesRegex(PreflightBlocked, "SIGNAL_TOO_OLD"):
            validate_fresh_momentum(self.item, self.now)
        self.item["observed_at"] = datetime.fromtimestamp(
            self.now-5, timezone.utc).isoformat()
        self.item["indicators"]["adx14"] = 18
        with self.assertRaisesRegex(PreflightBlocked, "THREE_FILTERS_NOT_MET"):
            validate_fresh_momentum(self.item, self.now)
    def test_daily_combined_gate_and_settings(self):
        self.state["orders"] = {"cr": {"day": self.day, "phase": "PROTECTED"}}
        self.state["trade_settings"]["max_daily"] = 2
        self._write_state()
        snap = read_crash_snapshot(self.root, self.now)
        self.assertTrue(gate_shared_capacity(
            snap, [{"day": self.day}], set(), set(), "TESTUSDT"))
    def test_global_position_and_pending_caps(self):
        snap = read_crash_snapshot(self.root, self.now)
        with self.assertRaisesRegex(PreflightBlocked, "MAX_OPEN_POSITIONS"):
            gate_shared_capacity(snap, [], set("P"+str(i) for i in range(11)),
                                 set(), "TESTUSDT")
        with self.assertRaisesRegex(PreflightBlocked, "MAX_PENDING_ENTRIES"):
            gate_shared_capacity(snap, [], set(),
                                 set("P"+str(i) for i in range(14)), "TESTUSDT")
        with self.assertRaisesRegex(PreflightBlocked, "SYMBOL_ALREADY_ACTIVE"):
            gate_shared_capacity(snap, [], {"TESTUSDT"}, set(), "TESTUSDT")
    def test_fail_closed_when_crash_paused_or_missing(self):
        self.state["paused"] = True
        self._write_state()
        with self.assertRaisesRegex(PreflightBlocked, "CRASH_AUTO_PAUSED"):
            read_crash_snapshot(self.root, self.now)
        (self.root / "crash_night_state.json").unlink()
        with self.assertRaisesRegex(PreflightBlocked, "CRASH_STATE_UNAVAILABLE"):
            read_crash_snapshot(self.root, self.now)
    def test_no_trade_request_ever_created(self):
        self.assertFalse(hasattr(__import__("cb2_real_order_gateway"), "buy"))
        self.assertFalse(hasattr(__import__("cb2_real_order_gateway"), "submit"))


if __name__ == "__main__":
    unittest.main()
