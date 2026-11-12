"""Regression coverage for non-destructive 10-day disk housekeeping."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from crash_disk_retention import cleanup_if_due, purge_disabled_risk_v1_raw, RETENTION_DAYS, SWEEP_INTERVAL_SECONDS


def epoch(year=2026, month=11, day=11, hour=18):
    return datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp()


class DiskRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "crash_research"
        self.root.mkdir()
        self.now = epoch()
        self.logs = []

    def write(self, path, age_days=None, content="old test"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if age_days is not None:
            stamp = self.now - age_days * 86400
            os.utime(path, (stamp, stamp))
        return path

    def sweep(self, now=None, pending=None):
        return cleanup_if_due(self.base, self.now if now is None else now,
                              pending=pending, log=self.logs.append)

    def test_risk_v1_purge_requires_explicit_flag_and_disabled_collector(self):
        raw = self.write(self.root / "risk_v1" / "2026-11-09.jsonl", content="huge shadow test")
        with patch.dict(os.environ, {"CRASH_RISK_V1_ENABLED": "0",
                                    "CRASH_RISK_V1_PURGE_LOGS": "0"}):
            self.assertIsNone(purge_disabled_risk_v1_raw(self.base))
        self.assertTrue(raw.exists())
        with patch.dict(os.environ, {"CRASH_RISK_V1_ENABLED": "1",
                                    "CRASH_RISK_V1_PURGE_LOGS": "1"}):
            self.assertIsNone(purge_disabled_risk_v1_raw(self.base))
        self.assertTrue(raw.exists())

    def test_risk_v1_purge_only_raw_jsonl_preserves_summaries_and_orders(self):
        risk = self.root / "risk_v1"
        raw_old = self.write(risk / "2026-11-09.jsonl", content="a" * 97)
        raw_today = self.write(risk / "2026-11-12.jsonl", content="b" * 17)
        keep = [self.write(risk / n) for n in
                ("summary.json", "state.json", "experiment.jsonl", "2026-11-09.jsonl.gz")]
        keep += [self.write(self.root / "2026-11-09.jsonl"),
                 self.write(self.base / "crash_night_state.json")]
        external = self.write(self.base / "external.jsonl")
        (risk / "2026-11-08.jsonl").symlink_to(external)
        with patch.dict(os.environ, {"CRASH_RISK_V1_ENABLED": "0",
                                    "CRASH_RISK_V1_PURGE_LOGS": "1"}):
            result = purge_disabled_risk_v1_raw(self.base, log=self.logs.append)
        self.assertEqual(result["deleted_files"], 2)
        self.assertEqual(result["freed_bytes"], 114)
        self.assertFalse(raw_old.exists())
        self.assertFalse(raw_today.exists())
        self.assertTrue(all(x.exists() for x in keep))
        self.assertTrue(external.exists())
        self.assertTrue((risk / "2026-11-08.jsonl").is_symlink())

    def test_first_sweep_keeps_latest_ten_utc_dates(self):
        day = datetime.fromtimestamp(self.now, timezone.utc).date()
        paths = [self.write(self.root / f"{day-timedelta(days=i)}.jsonl")
                 for i in range(13)]
        result = self.sweep()
        self.assertEqual(result["deleted_files"], 3)
        self.assertTrue(all(path.exists() for path in paths[:10]))
        self.assertTrue(all(not path.exists() for path in paths[10:]))
        self.assertEqual(result["errors"], 0)

    def test_second_sweep_only_after_ten_days(self):
        path = self.write(self.root / "2026-10-10.jsonl")
        self.sweep()
        self.assertFalse(path.exists())
        new = self.write(self.root / "2026-11-01.jsonl")
        self.assertIsNone(self.sweep(self.now + 9 * 86400))
        self.assertTrue(new.exists())
        result = self.sweep(self.now + SWEEP_INTERVAL_SECONDS)
        self.assertEqual(result["deleted_files"], 1)
        self.assertFalse(new.exists())
        self.assertEqual(result["interval_days"], 10)

    def test_never_delete_trading_state_or_active_candles(self):
        saved = [self.write(self.base / name, 25) for name in
                 ("crash_bounce_state.json", "state.json",
                  "crash_bounce_ws_candles.json", "log.txt")]
        saved += [self.write(self.root / name, 25) for name in
                  ("pending.json", "download_schedule.json", "strange.jsonl",
                   "2026-01-01.json", "unrecognized.zip")]
        live = self.write(self.root / "history_cache" / "BTCUSDT.json", 3)
        self.sweep()
        self.assertTrue(all(p.exists() for p in saved))
        self.assertTrue(live.exists())

    def test_stale_caches_only_and_never_pending_schedule(self):
        stale = [self.write(self.root / name / "BTCUSDT.json", 11)
                 for name in ("history_cache", "fallback_cache", "bybit_reference",
                              "coinbase_reference", "urgent_bybit")]
        protected = [self.write(self.root / "history_cache" / f, 25)
                     for f in ("pending.json", "download_schedule.json")]
        recent = self.write(self.root / "history_cache" / "ETHUSDT.json", 9)
        result = self.sweep()
        self.assertEqual(result["deleted_files"], 5)
        self.assertTrue(all(not f.exists() for f in stale))
        self.assertTrue(all(p.exists() for p in protected))
        self.assertTrue(recent.exists())

    def test_unfinished_report_and_its_parts_are_protected(self):
        orig = self.write(self.root / "crash_data_123.zip", 20)
        part = self.write(self.root / "crash_data_123.part001.zip", 20)
        temp = self.write(self.root / "crash_data_123.zip.tmp", 20)
        other = self.write(self.root / "crash_data_987.zip", 20)
        result = self.sweep(pending={"path": str(orig), "parts": [str(part)]})
        self.assertEqual(result["deleted_files"], 1)
        self.assertTrue(all(p.exists() for p in (orig, part, temp)))
        self.assertFalse(other.exists())

    def test_orphaned_report_parts_and_temps_expire(self):
        files = [self.write(self.root / n, 15) for n in (
            "crash_data_123.zip", "crash_data_123.part001.zip",
            "crash_data_123.part002.zip.tmp")]
        self.sweep()
        self.assertTrue(all(not p.exists() for p in files))

    def test_symlink_and_unrelated_files_untouched(self):
        external = self.write(self.base / "external.json", 50)
        d = self.root / "history_cache"
        d.mkdir()
        link = d / "BTCUSDT.json"
        link.symlink_to(external)
        self.sweep()
        self.assertTrue(link.is_symlink())
        self.assertTrue(external.exists())

    def test_invalid_or_missing_research_does_not_create_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(cleanup_if_due(d, self.now))
            self.assertFalse((Path(d) / "crash_research").exists())

    def test_bad_time_rejected_before_cleanup(self):
        with self.assertRaises(ValueError):
            self.sweep(now=float("nan"))
        with self.assertRaises(ValueError):
            self.sweep(now=-3)
        self.assertFalse((self.root / ".disk_retention.json").exists())

    def test_success_persists_marker_and_storage_is_still_local_only(self):
        old = self.write(self.root / "2026-10-10.jsonl", 20)
        result = self.sweep()
        marker = json.loads((self.root / ".disk_retention.json").read_text())
        self.assertEqual(marker["last_run_at"], self.now)
        self.assertEqual(marker["freed_bytes"], len("old test"))
        self.assertEqual(result["deleted_files"], 1)
        self.assertTrue(any("orders_and_R2_untouched=1" in s for s in self.logs))
        self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
