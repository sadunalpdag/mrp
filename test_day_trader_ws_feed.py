"""Offline-only verification: no requests, no Binance network, no order side effects."""
import json
import tempfile
import unittest

from day_trader_ws_feed import DayTraderWsFeed


class DayTraderWsFeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = 2_000_000_000.0
        self.feed = DayTraderWsFeed(self.temp.name, clock=lambda: self.now)
        self.feed.start()
        self.step = 900_000
        self.first = (int(self.now * 1000) // self.step - 1) * self.step

    def row(self, t=None, **values):
        t = self.first if t is None else t
        payload = [t, "1", "1.2", "0.9", "1.1", "10", t + self.step - 1]
        for index, key in ((1, "o"), (2, "h"), (3, "l"), (4, "c"), (5, "v")):
            if key in values:
                payload[index] = values[key]
        return payload

    def saved(self, sym="UAIUSDT"):
        with open(self.temp.name + "/" + sym + ".json", encoding="utf-8") as handle:
            return json.load(handle)

    def test_only_closed_ws_bar_persisted_with_explicit_source(self):
        self.assertTrue(self.feed.submit("UAIUSDT", self.row()))
        self.feed.events.join()
        doc = self.saved()
        self.assertEqual(doc["schema"], "crash_day_trader_ws_v1")
        self.assertEqual(doc["source"], "BINANCE_USDT_M_WEBSOCKET")
        self.assertEqual(doc["interval"], "15m")
        self.assertTrue(doc["closed_only"])
        self.assertEqual(doc["candle_count"], 1)
        self.assertEqual(doc["candles"][0]["volume"], "10")

    def test_invalid_symbol_ohlc_and_stale_or_open_bar_rejected(self):
        self.assertFalse(self.feed.submit("../../abc", self.row()))
        self.assertFalse(self.feed.submit("UAIUSDT", self.row(h="0.8")))
        self.assertFalse(self.feed.submit("UAIUSDT", self.row(l="1.15")))
        self.assertFalse(self.feed.submit("UAIUSDT", self.row(c="NaN")))
        self.assertFalse(self.feed.submit("UAIUSDT", self.row(v="-2")))
        self.assertFalse(self.feed.submit("UAIUSDT", self.row(t=self.first + 1000)))
        self.assertFalse(self.feed.submit("UAIUSDT", self.row(t=self.first + self.step)))
        self.assertFalse(self.feed.submit("UAIUSDT", self.row(t=self.first - self.step * 5)))
        self.assertEqual(self.feed.events.qsize(), 0)

    def test_dedup_missing_slots_not_fabricated_and_restarts(self):
        self.assertTrue(self.feed.submit("UAIUSDT", self.row()))
        self.feed.events.join()
        self.assertTrue(self.feed.submit("UAIUSDT", self.row()))
        self.feed.events.join()
        self.assertEqual(self.saved()["candle_count"], 1)
        self.first += self.step * 2
        self.now += 1800
        self.assertTrue(self.feed.submit("UAIUSDT", self.row()))
        self.feed.events.join()
        doc = self.saved()
        self.assertEqual(doc["candle_count"], 2)
        self.assertEqual(doc["gap_slots"], 1)
        self.assertEqual(len(doc["candles"]), 2)
        again = DayTraderWsFeed(self.temp.name, clock=lambda: self.now)
        again.start()
        self.assertTrue(again.submit("UAIUSDT", self.row()))
        again.events.join()
        self.assertEqual(self.saved()["candle_count"], 2)

    def test_bounded_real_bars_and_no_external_data_import(self):
        feed = DayTraderWsFeed(self.temp.name, clock=lambda: self.now, max_bars=16)
        feed.start()
        for i in range(20):
            t = self.first + i * self.step
            self.now = (t + self.step + 1000) / 1000
            self.assertTrue(feed.submit("BTCUSDT", self.row(t=t)))
        feed.events.join()
        doc = self.saved("BTCUSDT")
        self.assertEqual(doc["candle_count"], 16)
        self.assertEqual(doc["gap_slots"], 0)
        self.assertEqual(doc["candles"][0]["open_time"], self.first + 4 * self.step)


if __name__ == "__main__":
    unittest.main()
