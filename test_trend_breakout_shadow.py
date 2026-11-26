"""Trend breakout shadow: strict watched-symbol gate and 72h research."""
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import trend_breakout_shadow as shadow


class TrendShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = datetime(2026, 11, 11, 12, 1, tzinfo=timezone.utc).timestamp()
        self.lines = []
        self.uploads = []
        self.agent = shadow.TrendBreakoutShadow(
            self.tmp.name, log=self.lines.append,
            now_fn=lambda: self.now,
            uploader=lambda json_file, zip_file, day:
            self.uploads.append((json_file, zip_file, day)) or ["source", "backup"],
        )

    def candle(self):
        # Fully closed 11:45-12:00 UTC bar: normal 15m exchange timing.
        open_ms = int(self.now // 900 - 1) * shadow.STEP_MS
        return [open_ms, "100", "102", "97.5", "100",
                "10000", open_ms + shadow.STEP_MS - 1]

    def breakout(self):
        return {
            "status": "KIRDI", "direction": "YUKARI",
            "confirmed_at": self.now - 90, "distance_pct": 1.1,
            "coverage": "480/480", "support": 95, "resistance": 105,
            "fib_band": "38.2–50"
        }

    def test_only_already_watched_symbols_never_50_extra(self):
        tickers = [{"symbol": f"C{i:02}USDT",
                    "priceChangePercent": str(-20 - i)}
                   for i in range(1, 51)]
        allowed = {f"C{i:02}USDT" for i in range(1, 21)}
        candle_seen = []
        def load(directory, symbol, now, getter):
            candle_seen.append(symbol)
            return [self.candle()]
        with patch.object(shadow, "_joined_rows", side_effect=load), \
             patch("crash_levels_5d.analyze", return_value=self.breakout()):
            self.agent._scan(tickers, lambda _: [], [], self.now, allowed)
        self.assertEqual(len(candle_seen), 20)
        self.assertEqual(set(candle_seen), allowed)
        self.assertEqual(len(self.agent.state["events"]), 20)
        self.assertTrue(all(e["symbol"] in allowed for e in self.agent.state["events"]))
        self.assertFalse(self.uploads)
        self.assertTrue(all(e["signal_context"] == {} for e in self.agent.state["events"]))
        self.assertTrue(all(e["phase"] == "TRACKING" for e in self.agent.state["events"]))

    def test_unwatched_ticker_cannot_create_event(self):
        tickers = [{"symbol": "BADUSDT", "priceChangePercent": "-50"}]
        with patch.object(shadow, "_joined_rows") as load:
            self.agent._scan(tickers, lambda _: [], [], self.now, {"GOODUSDT"})
        self.assertEqual(self.agent.state["events"], [])
        load.assert_not_called()

    def test_15m_high_low_tracking_24_48_72_and_unknown_intrabar_order(self):
        at = 1000000.0
        event = dict(
            reference_price=100, observed_at=at, last_bar_at=at,
            max_up_pct=0, max_down_pct=0, last_close_pct=0,
            first_plus_1pct_at=None, first_minus_1pct_at=None,
            bars_observed=0, coverage_gaps=0, phase="TRACKING",
            horizons={str(h) + "h": None for h in shadow.HORIZONS},
            intrabar_order="UNKNOWN")
        for i in range(1, 289):
            self.assertTrue(shadow._add_bar(event, at + i * 900, 102, 97, 99))
        self.assertEqual(event["phase"], "MATURED")
        self.assertEqual(event["bars_observed"], 288)
        self.assertEqual(event["coverage_gaps"], 0)
        self.assertEqual(event["max_up_pct"], 2.0)
        self.assertEqual(event["max_down_pct"], -3.0)
        for hrs in (24, 48, 72):
            self.assertEqual(event["horizons"][str(hrs) + "h"]["close_pct"], -1.0)
        self.assertEqual(event["intrabar_order"], "UNKNOWN")
        self.assertFalse(shadow._add_bar(event, at + 901, 200, 1, 199))

    def test_missing_15m_gaps_are_detected(self):
        e = dict(reference_price=100, observed_at=1000,
                 last_bar_at=1000, max_up_pct=0, max_down_pct=0,
                 first_plus_1pct_at=None, first_minus_1pct_at=None,
                 last_close_pct=0, bars_observed=0, coverage_gaps=0,
                 phase="TRACKING",
                 horizons={str(h) + "h": None for h in shadow.HORIZONS})
        shadow._add_bar(e, 1900, 101, 99, 100)
        shadow._add_bar(e, 3700, 101, 99, 100)
        self.assertEqual(e["coverage_gaps"], 1)
        self.assertEqual(e["bars_observed"], 2)

    def test_daily_json_zip_contain_same_valid_ai_ready_records(self):
        self.agent.state["events"].append({
            "event_id": "ABCUSDT-YUKARI-11", "symbol": "ABCUSDT",
            "direction": "YUKARI", "observed_at": self.now - 3600,
            "reference_price": 10, "horizons": {"24h": None}})
        day = shadow._date(self.now)
        jp, zp, count = self.agent._export(day, self.now)
        self.assertEqual(count, 1)
        data = json.loads(Path(jp).read_text(encoding="utf-8"))
        self.assertEqual(data["sampling"], "max_20_of_existing_crash_bounce_watched_symbols")
        self.assertEqual(data["event_count"], 1)
        self.assertTrue(data["research_only"])
        self.assertTrue(data["no_extra_websocket_subscriptions"])
        with zipfile.ZipFile(zp) as z:
            self.assertEqual(json.loads(z.read(Path(jp).name)), data)

    def test_daily_upload_done_once_no_local_files_left(self):
        now = datetime(2026, 11, 11, 21, 16,
                       tzinfo=shadow.PARIS).timestamp()
        self.agent._upload(now)
        self.agent._upload(now)
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(self.agent.state["last_uploaded_day"], "2026-11-11")
        self.assertTrue(all(not Path(x).exists() for x in self.uploads[0][:2]))
        saved = json.loads(Path(self.agent.state_file).read_text(encoding="utf-8"))
        self.assertEqual(saved["uploaded_keys"], ["source", "backup"])

    def test_no_worker_thread_before_next_scan_and_before_upload(self):
        self.agent.next_scan = self.now + 900
        with patch.object(self.agent, "_scan") as scan, \
             patch.object(self.agent, "_upload") as upload:
            self.agent.tick([], lambda _: [], [], {"ABCUSDT"})
            scan.assert_not_called()
            upload.assert_not_called()
        self.assertFalse(self.agent.running)
        self.assertFalse(self.agent.lock.locked())

    def test_no_report_upload_when_earlier_than_2115_paris(self):
        early = datetime(2026, 11, 11, 20, 59,
                         tzinfo=shadow.PARIS).timestamp()
        self.agent._upload(early)
        self.assertFalse(self.uploads)

    def test_absent_r2_configuration_cannot_claim_uploaded(self):
        with patch.dict(shadow.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "R2_NOT_CONFIGURED"):
                shadow._r2_upload_files("/tmp/x.json", "/tmp/y.zip", "2026-11-11")


if __name__ == "__main__":
    unittest.main()
