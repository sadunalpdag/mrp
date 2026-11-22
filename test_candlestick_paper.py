import os
import tempfile
import time
import unittest
from unittest.mock import patch

import candlestick_paper as paper


class CandlePaperTests(unittest.TestCase):
    def test_pattern_rejects_setup_and_accepts_complete(self):
        now = 1_800_000_000_000
        row = {"symbol": "BTCUSDT", "exchange": "binance-futures",
               "tf": "15m", "price": 100,
               "computedAt": now - 60_000,
               "item": {"name": "Hammer", "state": "setup", "direction": "bull"}}
        self.assertIsNone(paper._pattern(row, "15m", now))
        row["item"]["state"] = "complete"
        self.assertEqual(paper._pattern(row, "15m", now)[2], "Hammer")
        row["item"]["name"] = "Morning Star"
        self.assertEqual(paper._pattern(row, "15m", now)[2], "Morning Star")

    def test_capped_at_ten_and_deduplicated(self):
        now = int(time.time() * 1000)
        rows = [
            {"symbol": f"T{i}USDT", "exchange": "binance-futures", "tf": "15m",
             "price": 100, "computedAt": now - 60_000,
             "item": {"name": "Hammer", "state": "complete", "direction": "bull"}}
            for i in range(12)
        ]
        def fake_api(path, params=None):
            if path == "/public/usage":
                return {"today": {"creditsLeft": 1200},
                        "thisMinute": {"creditsLeft": 100}}
            return {"kind": "candlesticks", "timeframe": params["timeframe"],
                    "items": rows, "total": len(rows)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DATA_DIR": directory}), \
             patch.object(paper, "_api", side_effect=fake_api), \
             patch.object(paper, "_sheet"), patch.object(paper, "_archive"), \
             patch.object(paper, "_notify"):
            paper._USAGE = None
            paper._USAGE_AT = 0.0
            paper.step()
            with paper._db() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM signals").fetchone()[0], 10)
            paper.step()
            with paper._db() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM signals").fetchone()[0], 10)

    def test_duplicate_identity_ignores_price(self):
        now = 1_800_000_000_000
        row = {"symbol": "BTCUSDT", "exchange": "binance-futures", "tf": "15m",
               "price": 101, "computedAt": now - 300_000,
               "item": {"name": "Hammer", "state": "complete", "direction": "bull"}}
        a = paper._pattern(row, "15m", now)
        row["price"] = 110
        b = paper._pattern(row, "15m", now)
        self.assertEqual(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
