"""Offline deterministic tests for isolated Gemini research; no network/orders."""
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import coinbeacon_ai_learning as ai
import coinbeacon_ai_daily_archive as archive


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"DATA_DIR": self.directory.name,
                                           "CB_AI_LEARNING_ENABLED": "0"}, clear=False)
        self.env.start()
        self.now = (int(time.time() * 1000) // ai.INTERVAL) * ai.INTERVAL - 1
        last_open = self.now - ai.INTERVAL + 1
        self.bars = []
        for index in range(480):
            opened = last_open - (479 - index) * ai.INTERVAL
            self.bars.append({"open_ms": opened, "close_ms": opened+ai.INTERVAL-1,
                              "open": 100.0, "high": 100.6, "low": 99.5, "close": 100.2})
        self.item = {"id": "0123456789abcdef01234567", "symbol": "TESTUSDT",
                     "archetype": "pattern_breakout", "sqs": 75, "observed_ms": self.now,
                     "source_signal_at": "2026-11-17T04:00:00Z",
                     "reference_price": 100.1, "indicators": {"adx14": 22}}

    def tearDown(self):
        self.env.stop()
        self.directory.cleanup()

    def insert(self):
        self.assertTrue(ai._ingest(self.item, self.bars, self.now))
        rec = ai._get(self.item["id"])
        self.assertEqual(len(rec["initial_candles"]), 480)
        return rec

    def test_paris_01_to_01_dispatch_counts_unique_symbols(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        paris = ZoneInfo("Europe/Paris")
        now = int(datetime(2026, 11, 18, 0, 30, tzinfo=paris).timestamp()*1000)
        previous = int(datetime(2026, 11, 17, 0, 30, tzinfo=paris).timestamp()*1000)
        with ai._database() as db:
            for ident, symbol, when in (("a","BTCUSDT",now),("b","BTCUSDT",now),
                                         ("c","ETHUSDT",now),("old","SOLUSDT",previous)):
                db.execute("INSERT INTO gemini_dispatches(case_id,symbol,phase,sent_ms) VALUES(?,?,?,?)",
                           (ident,symbol,"initial",when))
            db.commit()
        result = ai.daily_dispatch_stats(now)
        self.assertEqual(result["signals"],3)
        self.assertEqual(result["coins"],2)
        self.assertEqual(result["start"],"2026-11-17 01:00")

    def test_duplicate_is_not_queued_again(self):
        self.insert()
        self.assertFalse(ai._ingest(self.item, self.bars, self.now))
        self.assertEqual(len(ai._all()), 1)

    def test_open_candle_or_stale_history_rejected(self):
        bad = [dict(x) for x in self.bars]
        bad[-1]["close_ms"] += ai.INTERVAL
        bad = bad[:39] + [bad[-1]]
        self.assertFalse(ai._ingest(self.item, bad, self.now))
        self.assertEqual(len(ai._all()), 0)

    def test_disabled_does_not_enqueue(self):
        with patch.object(ai, "start") as start:
            self.assertFalse(ai.submit_candidate(self.item, self.bars))
            start.assert_not_called()

    def test_plan_validation_prevents_market_crossing_and_bad_prices(self):
        raw = {"direction": "LONG", "order_type": "STOP", "entry_price": 100.4,
               "confidence": .8, "rationale": "breakout"}
        self.assertEqual(ai._validate_plan(raw, 100)["direction"], "LONG")
        raw["entry_price"] = 99.0
        self.assertEqual(ai._validate_plan(raw, 100)["direction"], "WAIT")
        raw["entry_price"] = 150.0
        self.assertEqual(ai._validate_plan(raw, 100)["order_type"], "NONE")
        self.assertEqual(ai._validate_plan({"direction": "WAIT"}, 100)["direction"], "WAIT")

    def test_first_gemini_prediction_frozen_with_future_tracking_start(self):
        rec = self.insert()
        fake = {"direction": "LONG", "order_type": "STOP", "entry_price": 101.0,
                "confidence": .7, "rationale": "resistance"}
        future = self.now + 180000
        with patch.object(ai, "_ask_gemini", return_value=fake) as call:
            with patch.object(ai.time, "time", return_value=future/1000):
                ai._initial(rec)
        saved = ai._get(rec["id"])
        self.assertEqual(saved["state"], "TRACKING")
        self.assertEqual(saved["initial_plan"]["entry_price"], 101.0)
        self.assertGreaterEqual(saved["first_full_open_ms"], saved["planned_ms"])
        self.assertEqual(saved["deadline_ms"]-saved["planned_ms"], ai.HORIZON)
        self.assertEqual(call.call_count, 1)
        self.assertFalse(saved["future_candles"])

    def test_entry_candle_tp_is_ambiguous_not_confirmed(self):
        rec = self.insert()
        first = 50*ai.INTERVAL
        rec.update(initial_plan={"direction": "LONG", "order_type": "LIMIT", "entry_price": 100.0},
                   first_full_open_ms=first, deadline_ms=first+2*ai.INTERVAL,
                   future_candles=[{"t": first, "o": 100.0, "h": 103.0, "l": 99.0, "c": 102.0}])
        result = ai._update_outcome(rec)
        self.assertTrue(result["entry_triggered"])
        self.assertTrue(result["tp_ambiguous_in_entry_candle"])
        self.assertFalse(result["tp_observed"])
        self.assertEqual(result["label"], "OPEN_OR_UNCERTAIN")
        rec["future_candles"].append({"t": first+ai.INTERVAL, "o": 100,
                                      "h": 101.3, "l": 99.8, "c": 101.2})
        result = ai._update_outcome(rec)
        self.assertEqual(result["label"], "TP_OBSERVED")
        self.assertTrue(result["coverage_complete"])

    def test_missing_bars_are_explicit_not_wins(self):
        rec = self.insert()
        first = 50*ai.INTERVAL
        rec.update(initial_plan={"direction": "LONG", "order_type": "STOP", "entry_price": 101},
                   first_full_open_ms=first, deadline_ms=first+3*ai.INTERVAL,
                   future_candles=[{"t": first+2*ai.INTERVAL, "o": 99, "h": 99.5, "l": 98, "c": 99}])
        out = ai._update_outcome(rec)
        self.assertEqual(out["gap_count"], 2)
        self.assertEqual(out["label"], "NOT_TRIGGERED_DATA_GAP")

    def test_expiration_stops_stream_tracking_and_prepares_review(self):
        rec = self.insert()
        rec.update(state="TRACKING", planned_ms=self.now,
                   first_full_open_ms=self.now+1, deadline_ms=self.now+ai.HORIZON,
                   initial_plan={"direction": "WAIT", "order_type": "NONE", "entry_price": None})
        ai._save(rec)
        with patch.object(ai, "_rest_gap_fill"):
            ai._expiry(rec)
        self.assertEqual(ai._get(rec["id"])["state"], "REVIEW_PENDING")
        self.assertEqual(ai._get(rec["id"])["outcome"]["label"], "WAIT")

    def test_final_review_is_separate_from_immutable_prediction(self):
        rec = self.insert()
        rec["initial_plan"] = {"direction": "WAIT", "order_type": "NONE", "entry_price": None}
        rec["outcome"] = {"label": "WAIT", "gap_count": 0}
        rec["state"] = "REVIEW_PENDING"
        ai._save(rec)
        with patch.object(ai, "_ask_gemini", return_value={"result_summary": "No entry.",
              "initial_thesis_correct": None, "learning_tags": ["low_volume"],
              "failure_modes": [], "suggested_filter_changes": []}):
            ai._final(rec)
        saved = ai._get(rec["id"])
        self.assertEqual(saved["state"], "COMPLETE")
        self.assertEqual(saved["initial_plan"]["direction"], "WAIT")
        self.assertEqual(saved["final_review"]["learning_tags"], ["low_volume"])


    def test_one_daily_archive_preserves_480_candles_and_late_result(self):
        rec = self.insert()
        day = archive.research_day(rec["observed_ms"])
        with ai._database() as db:
            first = archive.build_payload(db, day)
            revision_before = db.execute(
                "SELECT revision FROM daily_archives WHERE research_day=?", (day,)).fetchone()[0]
        self.assertEqual(first["summary"]["signals"], 1)
        self.assertEqual(first["summary"]["unique_coins"], 1)
        self.assertEqual(len(first["cases"][0]["initial_candles"]), 480)
        self.assertEqual(first["summary"]["gemini_decisions"]["NO_RESPONSE"], 1)
        expected_key = archive.archive_key(day)
        rec["state"] = "COMPLETE"
        rec["initial_plan"] = {"direction": "LONG", "entry_price": 100, "tp_pct": 1}
        rec["outcome"] = {"label": "TP_OBSERVED", "coverage_complete": True,
                          "tp_observed": True, "mae_pct": -0.4, "mfe_pct": 1.1}
        rec["future_candles"] = [{"t": self.now + ai.INTERVAL, "o": 100,
                                  "h": 101.1, "l": 99.6, "c": 101}]
        rec["final_review"] = {"learning_tags": ["breakout_retest"],
                                "suggested_filter_changes": ["look at retest"]}
        ai._save(rec)
        with ai._database() as db:
            updated = archive.build_payload(db, day)
            revision_after = db.execute(
                "SELECT revision FROM daily_archives WHERE research_day=?", (day,)).fetchone()[0]
        self.assertEqual(archive.archive_key(day), expected_key)
        self.assertGreater(revision_after, revision_before)
        self.assertEqual(updated["summary"]["outcome_labels"]["TP_OBSERVED"], 1)
        self.assertEqual(updated["summary"]["learning_tags"]["breakout_retest"], 1)
        self.assertEqual(updated["cases"][0]["outcome"]["mae_pct"], -0.4)
        self.assertEqual(len(updated["cases"][0]["future_candles"]), 1)

    def test_day_boundary_uses_paris_01_00_even_on_dst_transition(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        p = ZoneInfo("Europe/Paris")
        before = datetime(2026, 11, 18, 0, 59, tzinfo=p)
        after = datetime(2026, 11, 18, 1, 0, tzinfo=p)
        self.assertEqual(archive.research_day(int(before.timestamp()*1000)), "2026-11-17")
        self.assertEqual(archive.research_day(int(after.timestamp()*1000)), "2026-11-18")
        s, e = archive.bounds("2026-10-24")
        self.assertEqual((e.timestamp() - s.timestamp()) / 3600, 25)

    def test_archive_reuploads_same_key_after_late_update(self):
        rec = self.insert()
        captured = []
        now = int(time.time()*1000)
        def send(key, body):
            import json
            captured.append((key, json.loads(body)))
        with patch("crash_report_storage.r2_configured", return_value=True), \
             patch.object(archive, "_put_r2", side_effect=send):
            self.assertTrue(archive.sync_due(ai._database, now_ms=now))
            self.assertFalse(archive.sync_due(ai._database, now_ms=now+1000))
            rec["state"] = "COMPLETE"
            rec["outcome"] = {"label": "NO_TP_8H", "coverage_complete": True}
            ai._save(rec)
            self.assertTrue(archive.sync_due(ai._database,
                                             now_ms=now+archive.MIN_REUPLOAD_MS+1))
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0][0], captured[1][0])
        self.assertEqual(captured[0][1]["summary"]["states"], {"QUEUED": 1})
        self.assertEqual(captured[1][1]["summary"]["states"], {"COMPLETE": 1})

    def test_failed_daily_upload_retries_without_data_loss(self):
        self.insert()
        now = int(time.time()*1000)
        with patch("crash_report_storage.r2_configured", return_value=True):
            with patch.object(archive, "_put_r2", side_effect=OSError("offline")):
                self.assertFalse(archive.sync_due(ai._database, now_ms=now))
            with ai._database() as db:
                self.assertEqual(db.execute(
                    "SELECT uploaded_revision FROM daily_archives").fetchone()[0], 0)
            with patch.object(archive, "_put_r2", return_value=None):
                self.assertTrue(archive.sync_due(
                    ai._database, now_ms=now+archive.MIN_REUPLOAD_MS+1))

    def test_r2_missing_does_not_fake_uploaded_success(self):
        rec = self.insert()
        with patch("crash_report_storage.r2_configured", return_value=False):
            self.assertFalse(ai._r2_upload(rec))
        with ai._database() as db:
            row = db.execute("SELECT uploaded FROM cases WHERE id=?", (rec["id"],)).fetchone()
            self.assertEqual(row[0], 0)


if __name__ == "__main__":
    unittest.main()
