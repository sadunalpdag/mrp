"""Offline tests for CoinBeacon BTC detector ingestion (no paid API calls)."""
import os
import unittest
from unittest.mock import patch, Mock

import btc_coinbeacon_levels as cb
import btc_range_paper as paper
from test_btc_range_paper import oscillating_window


HOUR = 1_797_000_000_000


def row(label, price, *, timeframe="1h", computed=None, available=None, source_price=100.12):
    if computed is None:
        computed = HOUR - 1
    if available is None:
        available = HOUR - 1
    return {"symbol": "BTCUSDT", "exchange": "binance-futures", "tf": timeframe,
            "price": source_price, "computedAt": computed,
            "item": {"type": "pivot", "label": label, "price": price,
                     "kind": "resistance", "availableAt": available,
                     "period": {"validFrom": HOUR, "validTo": HOUR + cb.FRAME_MS[timeframe]}}}


def payload(rows, tf="1h"):
    return {"kind": "levels", "exchange": "binance-futures",
            "timeframe": tf, "count": len(rows), "total": len(rows), "items": rows}


class CoinBeaconRangeTests(unittest.TestCase):
    def test_normalize_only_same_venue_symbol_and_known_closed_level(self):
        right = row("Pivot R1", 100.0)
        future = row("Pivot R2", 102.0, available=HOUR + 100)
        stale = row("Pivot P", 101, computed=HOUR - 3 * cb.HOUR_MS)
        other = dict(right, symbol="ETHUSDT")
        levels = cb.normalize(payload([right, future, stale, other]), HOUR, 100.12, "1h")
        self.assertEqual([x["label"] for x in levels], ["Pivot R1"])

    def test_expired_period_is_rejected(self):
        x = row("Pivot R1", 100.0)
        x["item"]["period"]["validTo"] = HOUR
        self.assertEqual(cb.normalize(payload([x]), HOUR, 100.12, "1h"), [])

    def test_wrong_market_and_future_compute_are_rejected(self):
        with self.assertRaises(ValueError):
            cb.normalize(payload([], "4h"), HOUR, 100.12, "1h")
        self.assertEqual(cb.normalize(payload([row("Pivot R1", 100, computed=HOUR+1)]),
                                      HOUR, 100.12, "1h"), [])

    def test_coinbeacon_levels_trigger_paper_only_if_repeated_and_range(self):
        bars = oscillating_window()
        original = paper.detect_range(bars, 12)
        feed = {"status": "LEVELS_AVAILABLE", "frames": {"1h": "BTC_FOUND"},
                "levels": [cb.normalize(payload([row("Pivot R1", 100.0),
                                                  row("Pivot S1", 102.0)]),
                                        HOUR, 100.12, "1h")[i] for i in range(2)]}
        result = cb.evaluate(original, bars, feed)
        self.assertEqual(result["status"], "CONFIRMED_PAIR")
        self.assertEqual(result["support"], 100)
        # Name R1 below price becomes support; S1 above price becomes resistance.
        with patch.object(cb, "fetch_levels", return_value=feed):
            snapshot = cb.apply(original, bars, HOUR, lambda: False)
        self.assertEqual(snapshot["level_source"], "COINBEACON_VERIFIED_CLOSED_CANDLES")
        self.assertEqual(snapshot["candidate"]["plan"], "COINBEACON_SUPPORT_TO_RESISTANCE")
        self.assertGreater(snapshot["candidate"]["target"], snapshot["candidate"]["entry"])

    def test_missing_btc_does_not_fabricate_level(self):
        bars = oscillating_window()
        original = paper.detect_range(bars, 12)
        with patch.object(cb, "fetch_levels", return_value={
                "status": "NO_VALID_BTC_LEVELS", "frames": {"1h": "BTC_MISSING_OR_STALE"},
                "levels": []}):
            result = cb.apply(original, bars, HOUR, lambda: False)
        self.assertEqual(result["level_source"], "BINANCE_CLOSED_CANDLES")
        self.assertEqual(result["coinbeacon"]["status"], "NO_VALID_BTC_LEVELS")
        self.assertNotIn("coinbeacon", result["candidate"]["plan"].lower())

    def test_error_or_429_does_not_make_second_paid_call(self):
        response = Mock(status_code=429)
        with patch.dict(os.environ, {"api2": "unit-test-key"}), \
             patch.object(cb.requests, "get", return_value=response) as request:
            result = cb.fetch_levels(HOUR, 100.12, lambda: False)
        self.assertEqual(result["status"], "RATE_LIMIT")
        self.assertEqual(request.call_count, 1)

    def test_no_key_does_not_call_coinbeacon(self):
        with patch.dict(os.environ, {"api2": "", "API2": "", "COINBEACON_API_KEY": ""}), \
             patch.object(cb.requests, "get") as request:
            result = cb.fetch_levels(HOUR, 100.12, lambda: False)
        self.assertEqual(result["status"], "NO_API_KEY")
        request.assert_not_called()

    def test_trend_not_range_remains_non_tradable(self):
        bars = oscillating_window()
        original = paper.detect_range(bars, 35)
        with patch.object(cb, "fetch_levels", return_value={"status": "LEVELS_AVAILABLE",
                        "levels": [dict(price=100.0, label="Pivot R1", timeframe="1h"),
                                   dict(price=102.0, label="Pivot S1", timeframe="1h")]}):
            result = cb.apply(original, bars, HOUR, lambda: False)
        self.assertEqual(result["regime"], "NO_RANGE")
        self.assertNotIn("candidate", result)


if __name__ == "__main__":
    unittest.main()
