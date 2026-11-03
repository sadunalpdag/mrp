"""Offline tests for date-partitioned, append-only research records."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from research_archive import append_research_runtime, archive_location_ok


class ArchiveTests(unittest.TestCase):
    def test_append_partition_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(append_research_runtime(d,'[CRASH SCAN COMPLETE] eligible=1',now=1762128000))
            self.assertTrue(append_research_runtime(d,'[BTC HOURLY] COMPLETE',now=1762128600))
            self.assertTrue(append_research_runtime(d,'[TOP GAINERS] COMPLETE',now=1762214400))
            directory=Path(d)/'research_archive'/'runtime'
            files=sorted(directory.glob('*.jsonl'))
            self.assertEqual(len(files),2)
            rows=[json.loads(x) for x in files[0].read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertIn('eligible=1',rows[0]['message'])
            self.assertEqual(rows[1]['message'],'[BTC HOURLY] COMPLETE')
            self.assertTrue(rows[0]['ts_utc'].endswith('+00:00'))
    def test_no_secrets_or_unrelated_messages_saved(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(append_research_runtime(d,'[CRASH] BOT_TOKEN=secret',now=1762128000))
            self.assertFalse(append_research_runtime(d,'misc token',now=1762128000))
            self.assertFalse((Path(d)/'research_archive').exists())
    def test_mount_path_is_not_confused_with_ephemeral_repo(self):
        self.assertTrue(archive_location_ok('/data','/data'))
        self.assertTrue(archive_location_ok('/data/crash_research','/data'))
        self.assertFalse(archive_location_ok('/opt/render/project/src/data','/data'))


if __name__=='__main__':
    unittest.main()
