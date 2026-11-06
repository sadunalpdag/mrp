"""Offline tests for the opt-in EMA Filter reset utility (no Google calls)."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import reset_ema_filter_sheets as reset


class FakeTab:
    title = "Eski Veriler"
    id = 123
    row_count = 100
    col_count = 5

    def get_all_values(self, value_render_option=None):
        assert value_render_option == "FORMULA"
        return [["Coin", "Değer"], ["TESTUSDT", "=A2"]]


class TestEmaFilterReset(unittest.TestCase):
    def test_rejects_unrelated_spreadsheet(self):
        with self.assertRaises(ValueError):
            reset.guard_target("a" * 35, "Other Portfolio")
        with self.assertRaises(ValueError):
            reset.guard_target("", "EMA Filter")
        reset.guard_target("a" * 35, "EMA Filter")

    def test_backup_contains_values_and_formulas(self):
        sheet = Mock(id="a" * 35, title="EMA Filter")
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "backup.json.gz"
            reset.archive_sheet(sheet, [FakeTab()], dest)
            self.assertTrue(dest.is_file())
            with gzip.open(dest, "rt", encoding="utf-8") as fh:
                backup = json.load(fh)
            self.assertEqual(backup["title"], "EMA Filter")
            self.assertEqual(backup["tabs"][0]["values_and_formulas"][1][1], "=A2")

    def test_dry_run_never_mutates_spreadsheet(self):
        workbook = Mock(id="a" * 35, title="EMA Filter")
        workbook.worksheets.return_value = [FakeTab()]
        client = Mock()
        client.open_by_key.return_value = workbook
        env = {"GOOGLE_SHEETS_ID": "a" * 35, "GOOGLE_SERVICE_ACCOUNT_JSON": "{}"}
        with patch.object(reset, "auth_client", return_value=(client, "a" * 35)), \
             patch.object(reset, "archive_sheet") as archive, \
             patch.object(reset, "reset_sheet") as perform_reset:
            result = reset.main([], env=env)
        self.assertEqual(result, 0)
        archive.assert_not_called()
        perform_reset.assert_not_called()
        workbook.batch_update.assert_not_called()

    def test_backup_failure_never_resets(self):
        workbook = Mock(id="a" * 35, title="EMA Filter")
        workbook.worksheets.return_value = [FakeTab()]
        client = Mock()
        client.open_by_key.return_value = workbook
        env = {"GOOGLE_SHEETS_ID": "a" * 35, "GOOGLE_SERVICE_ACCOUNT_JSON": "{}"}
        with patch.object(reset, "auth_client", return_value=(client, "a" * 35)), \
             patch.object(reset, "archive_sheet", side_effect=OSError("disk full")), \
             patch.object(reset, "reset_sheet") as perform_reset:
            with self.assertRaises(OSError):
                reset.main(["--apply"], env=env)
        perform_reset.assert_not_called()

    def test_replacement_preserves_document_and_creates_two_tabs(self):
        status = Mock(id=88)
        tracking = Mock(id=99)
        workbook = Mock()
        workbook.add_worksheet.return_value = Mock(id=12)
        workbook.worksheet.side_effect = [status, tracking]
        result = reset.reset_sheet(workbook)
        self.assertEqual(result, ["BOT_STATUS", "COIN_TRACKING"])
        calls = workbook.batch_update.call_args_list
        self.assertEqual(len(calls), 2)
        operations = calls[0].args[0]["requests"]
        self.assertTrue(any("addSheet" in o for o in operations))
        self.assertTrue(any("deleteSheet" in o for o in operations))
        self.assertEqual(workbook.values_batch_update.call_count, 1)


if __name__ == "__main__":
    unittest.main()
