"""Offline Drive backup queue/retry tests; no Google network calls."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

from crash_drive_backup import DailyDriveBackup


class Tests(unittest.TestCase):
    def test_off_by_default(self):
        with tempfile.TemporaryDirectory() as path:
            b=DailyDriveBackup(path,Mock(),env={})
            self.assertFalse(b.enqueue("foo.zip",100))
            self.assertFalse(b.process_once())

    def test_retry_nonblocking_and_restart_persistence(self):
        now=[100000.0]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"report.zip"
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("failure_shadow.csv","symbol,risk\nAAA,weak\n")
            notify=Mock()
            env={"CRASH_DRIVE_BACKUP_ENABLED":"1"}
            b=DailyDriveBackup(root,notify,env=env,clock=lambda:now[0])
            self.assertTrue(b.enqueue(path,now[0]))
            self.assertFalse(b.enqueue(path,now[0]))
            self.assertTrue(b.process_once())  # no credentials -> queued retry
            self.assertEqual(b.tasks[path.name]["attempts"],1)
            self.assertGreater(b.tasks[path.name]["next_attempt"],now[0])
            b2=DailyDriveBackup(root,notify,env=env,clock=lambda:now[0])
            self.assertEqual(b2.tasks[path.name]["attempts"],1)
            self.assertFalse(b2.process_once())

    def test_uploads_zip_and_csv_once(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"report.zip"
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("failure_shadow.csv","symbol,risk\nAAA,weak\n")
            notified=Mock()
            b=DailyDriveBackup(root,notified,
                env={"CRASH_DRIVE_BACKUP_ENABLED":"1"},clock=lambda:123456.0)
            b.enqueue(path,123456.0)
            b._session=Mock(return_value=object())
            b._folder=Mock(side_effect=["folder","month"])
            b._upload=Mock(side_effect=["zip-id","csv-id"])
            self.assertTrue(b.process_once())
            self.assertEqual(b._upload.call_count,2)
            self.assertTrue(b.tasks[path.name]["done"])
            self.assertFalse(b.process_once())
            notified.assert_called()


if __name__=="__main__":
    unittest.main()
