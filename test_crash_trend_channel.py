import math
import unittest

from crash_trend_channel import analyze, STEP_MS
from crash_sheets_dashboard import build_rows, HEADERS


class TrendChannelTests(unittest.TestCase):
    now = 1794050000

    def rows(self, variant="inside"):
        start = (int(self.now * 1000) // STEP_MS - 64) * STEP_MS
        result = []
        for i in range(64):
            upper = 110 - 0.35 * i
            close = upper - 5 + 4 * math.sin(2 * math.pi * i / 10)
            if variant == "break" and i in (62, 63):
                close = upper + 1.5
            elif variant == "fake" and i in (60, 61):
                close = upper + 1.5
            elif variant == "fake" and i in (62, 63):
                close = upper - 2
            elif variant == "near" and i == 63:
                close = upper - 0.75
            elif variant == "pending" and i == 63:
                close = upper - 0.15
            hi, low = close + 0.8, close - 0.8
            opened = start + i * STEP_MS
            result.append([opened, str(close), str(hi), str(low),
                           str(close), "1000", opened + STEP_MS - 1])
        return result

    def test_closes_above_channel_are_confirmed(self):
        outcome = analyze(self.rows("break"), self.now)
        self.assertEqual(outcome["status"], "KIRDI", outcome)
        self.assertGreater(outcome["distance_pct"], 0)
        self.assertIsNotNone(outcome["confirmed_at"])

    def test_break_then_fall_back_is_fake(self):
        outcome = analyze(self.rows("fake"), self.now)
        self.assertEqual(outcome["status"], "SAHTE KIRILIM", outcome)

    def test_near_upper_trendline_is_yellow(self):
        outcome = analyze(self.rows("near"), self.now)
        self.assertEqual(outcome["status"], "YAKLAŞIYOR", outcome)

    def test_unconfirmed_above_channel_is_pending(self):
        outcome = analyze(self.rows("pending"), self.now)
        self.assertEqual(outcome["status"], "KIRILIM BEKLEME", outcome)
        self.assertIsNone(outcome["confirmed_at"])

    def test_missing_and_stale_candles_are_unknown(self):
        self.assertEqual(analyze([], self.now)["status"], "VERİ YOK")
        self.assertEqual(analyze(self.rows()[:12], self.now)["status"], "VERİ YOK")
        self.assertEqual(analyze(self.rows(), self.now + 3 * 3600)["status"], "VERİ YOK")

    def test_closed_candle_only_no_peek(self):
        initial = self.rows()
        unfinished = initial[-1].copy()
        unfinished[0] += STEP_MS
        unfinished[6] += STEP_MS
        unfinished[4] = "999999"
        unfinished[2] = "999999"
        with_unfinished = analyze(initial + [unfinished], self.now)
        self.assertEqual(with_unfinished, analyze(initial, self.now))

    def test_dashboard_uses_read_only_candle_callback_and_keeps_risk(self):
        symbols = {"TESTUSDT": {"signal_id": "1", "created_at": self.now - 60,
                    "v2_status": "STRONG_V2", "entry": 1, "tp": 1.01,
                    "failure_shadow": {"flags": {"weak_volume": False,
                        "falling_rsi_ema": True, "repeat_lower_entry": False, "low_atr_15m": False}}}}
        calls = []
        def cached(symbol):
            calls.append(symbol)
            return self.rows("break")
        _, table, colors, _ = build_rows({"symbols": symbols, "history": []},
            [{"symbol":"TESTUSDT", "priceChangePercent":"-20", "lastPrice":"1"}],
            {}, {}, self.now, candle_get=cached)
        self.assertEqual(calls, ["TESTUSDT"])
        self.assertEqual(table[1][13], 1)
        self.assertEqual(table[1][22], "⚪ VERİ YOK")
        self.assertEqual(table[1][26], "64/480")  # Short WS-only history cannot claim a 5d break
        self.assertEqual(len(table[1]), len(HEADERS))
        self.assertEqual(colors[0], ("green", "green"))


if __name__ == "__main__":
    unittest.main()
