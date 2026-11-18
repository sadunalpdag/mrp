"""Offline CoinBeacon BTC OHLCV pagination, data safety and rollup tests."""
import unittest
from unittest.mock import Mock, patch

import btc_coinbeacon_ohlcv as cb


def raw(opened):
    return {"openTime": opened, "closeTime": opened + cb.FIFTEEN - 1,
            "open": 100, "high": 101, "low": 99, "close": 100.25,
            "volume": 4.5}


class FakeMarket:
    key = "test-key"

    def __init__(self, total):
        self.bars = [raw(i * cb.FIFTEEN) for i in range(total)]
        self.calls = []

    def cb(self, path, cost, params):
        self.calls.append((path, cost, dict(params)))
        to = params["to"]
        bars = [b for b in self.bars if b["openTime"] <= to]
        return {"exchange": "binance-futures", "symbol": "BTCUSDT",
                "timeframe": "15m", "bars": bars[-params["limit"]:]}


class CoinBeaconOHLCVTests(unittest.TestCase):
    def setUp(self):
        cb._CACHE.clear()
        self.old = cb._CLIENT

    def tearDown(self):
        cb._CLIENT = self.old
        cb._CACHE.clear()

    def test_one_coinbeacon_request_for_hourly_and_paper(self):
        cb._CLIENT = FakeMarket(1000)
        before = 1000 * cb.FIFTEEN
        with patch.object(cb.requests, "get") as http:
            bars = cb.fetch_closed_15m(before, 1000)
            self.assertEqual(len(cb._CLIENT.calls), 1)
            self.assertEqual(cb._CLIENT.calls[0][1], 1)
            self.assertEqual(cb._CLIENT.calls[0][2]["to"], before - 1)
            self.assertEqual(cb._CLIENT.calls[0][2]["limit"], 1000)
            self.assertEqual(len(cb.aggregate(bars, "1h", before)), 250)
            self.assertGreaterEqual(len(cb.aggregate(bars, "4h", before)), 55)
            self.assertEqual(cb.fetch_closed_15m(before, 1000), bars)
            http.assert_not_called()
        self.assertEqual(len(cb._CLIENT.calls), 1)

    def test_thirty_days_two_backward_pages(self):
        cb._CLIENT = FakeMarket(2880)
        end = 2880 * cb.FIFTEEN
        result = cb.fetch_closed_15m(end, 2880)
        self.assertEqual(len(result), 2880)
        self.assertEqual(len(cb._CLIENT.calls), 2)
        self.assertEqual(cb._CLIENT.calls[0][2]["limit"], 1500)
        self.assertEqual(cb._CLIENT.calls[1][2]["limit"], 1380)
        self.assertEqual(cb._CLIENT.calls[1][2]["to"],
                         (2880 - 1500) * cb.FIFTEEN - 1)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[-1][0], end - cb.FIFTEEN)

    def test_cache_shallow_fails_no_fake_history(self):
        cb._CLIENT = FakeMarket(600)
        with self.assertRaises(cb.CandleUnavailable):
            cb.fetch_closed_15m(1000 * cb.FIFTEEN, 1000)
        self.assertNotIn((1000 * cb.FIFTEEN, 1000), cb._CACHE)

    def test_missing_one_candle_is_not_silently_filled(self):
        market = FakeMarket(1000)
        market.bars.pop(987)
        cb._CLIENT = market
        with self.assertRaises(cb.CandleUnavailable):
            cb.fetch_closed_15m(1000 * cb.FIFTEEN, 1000)

    def test_reject_future_and_invalid_ohlcv(self):
        before = 100 * cb.FIFTEEN
        self.assertIsNone(cb._validate_bar(raw(100 * cb.FIFTEEN), before))
        item = raw(99 * cb.FIFTEEN)
        item["high"] = 1
        self.assertIsNone(cb._validate_bar(item, before))

    def test_wrong_exchange_fails_closed(self):
        market = FakeMarket(1000)
        def bad(path, cost, params):
            return {"exchange": "binance", "symbol": "BTCUSDT",
                    "timeframe": "15m", "bars": market.bars[-1000:]}
        market.cb = bad
        cb._CLIENT = market
        with self.assertRaisesRegex(cb.CandleUnavailable, "WRONG_MARKET"):
            cb.fetch_closed_15m(1000 * cb.FIFTEEN, 1000)

    def test_priority_yields_without_http(self):
        cb._CLIENT = FakeMarket(1000)
        with self.assertRaisesRegex(cb.CandleUnavailable, "LIVE_CRASH_PRIORITY"):
            cb.fetch_closed_15m(1000 * cb.FIFTEEN, 1000, lambda: True)
        self.assertEqual(cb._CLIENT.calls, [])


if __name__ == "__main__":
    unittest.main()
