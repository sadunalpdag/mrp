"""Offline tests for R2 report archive and Telegram 49 MB ZIP fallback."""
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from crash_report_parts import report_zip_parts
from crash_report_storage import (
    LINK_EXPIRY_SECONDS, r2_configured, upload_private_report
)


class ReportPartsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "crash_data_123.zip"

    def test_small_report_keeps_single_original_file(self):
        with zipfile.ZipFile(self.path, "w") as z:
            z.writestr("report.json", '{"ok":true}')
        self.assertEqual(report_zip_parts(str(self.path), limit=250_000),
                         [str(self.path)])

    def test_oversized_report_makes_valid_independent_archives_with_manifest(self):
        original = {
            "report.json": b'{"metrics":"complete"}',
            "observations/large.jsonl": os.urandom(880_000),
            "signals_and_outcomes.json": b"[]" * 300,
        }
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as source:
            for name, data in original.items():
                source.writestr(name, data)
        parts = report_zip_parts(str(self.path), limit=250_000,
                                 chunk_bytes=80_000)
        self.assertGreater(len(parts), 3)
        stored = {}
        for name in parts:
            self.assertLessEqual(os.path.getsize(name), 250_000)
            with zipfile.ZipFile(name) as archive:
                self.assertIsNone(archive.testzip())
                for member in archive.namelist():
                    self.assertNotIn(member, stored)
                    stored[member] = archive.read(member)
        self.assertIn("MULTIPART_README.txt", stored)
        manifest = json.loads(stored["MULTIPART_README.txt"].decode("utf-8")
                              .split("\n\n")[-1])
        self.assertEqual(len(manifest["zip_parts"]), len(parts))
        self.assertEqual(set(original),
                         {item["name"] for item in manifest["original_files"]})
        for entry in manifest["original_files"]:
            reconstructed = b"".join(stored[p] for p in entry["files"])
            self.assertEqual(reconstructed, original[entry["name"]])

    def test_invalid_part_limit_never_makes_public_archive(self):
        self.path.write_bytes(os.urandom(250_000))
        with self.assertRaises(ValueError):
            report_zip_parts(str(self.path), limit=90_000)


class R2ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "crash_data_123.zip"
        self.path.write_bytes(b"PK0000"*10000)
        self.env = {
            "CRASH_REPORT_R2_ACCOUNT_ID": "a"*32,
            "CRASH_REPORT_R2_BUCKET": "crash-reports",
            "CRASH_REPORT_R2_ACCESS_KEY_ID": "key",
            "CRASH_REPORT_R2_SECRET_ACCESS_KEY": "secret",
        }

    def test_incomplete_r2_config_fails_closed(self):
        self.assertFalse(r2_configured({}))
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_R2_CONFIGURATION"):
            r2_configured({"CRASH_REPORT_R2_BUCKET": "only-bucket"})
        self.assertTrue(r2_configured(self.env))

    def test_full_private_zip_uploaded_verified_and_signed(self):
        fake = types.ModuleType("boto3")
        fake.client = Mock()
        s3 = types.ModuleType("boto3.s3")
        transfer = types.ModuleType("boto3.s3.transfer")
        transfer.TransferConfig = lambda **kwargs: kwargs
        client = Mock()
        client.head_object.return_value = {"ContentLength":self.path.stat().st_size}
        client.generate_presigned_url.return_value = "https://private.example/signed"
        fake.client.return_value = client
        with patch.dict(sys.modules, {"boto3": fake, "boto3.s3": s3,
                                      "boto3.s3.transfer": transfer}):
            result = upload_private_report(str(self.path), 1_800_000_000,
                                           env=self.env)
        self.assertIn("/crash_data_123.zip", result["key"])
        self.assertEqual(result["url"], "https://private.example/signed")
        self.assertEqual(result["link_days"], 7)
        self.assertEqual(result["size_bytes"], self.path.stat().st_size)
        client.upload_file.assert_called_once()
        args, kw = client.upload_file.call_args
        self.assertEqual(args[0], str(self.path))
        self.assertEqual(args[1], "crash-reports")
        self.assertEqual(kw["ExtraArgs"], {"ContentType": "application/zip"})
        self.assertEqual(client.generate_presigned_url.call_args.kwargs[
            "ExpiresIn"], LINK_EXPIRY_SECONDS)
        self.assertEqual(client.head_object.call_count, 1)

    def test_r2_size_mismatch_does_not_produce_link(self):
        fake = types.ModuleType("boto3")
        fake.client = Mock()
        s3 = types.ModuleType("boto3.s3")
        transfer = types.ModuleType("boto3.s3.transfer")
        transfer.TransferConfig = lambda **kw: kw
        client = Mock()
        client.head_object.return_value = {"ContentLength":0}
        fake.client.return_value = client
        with patch.dict(sys.modules, {"boto3": fake, "boto3.s3": s3,
                                      "boto3.s3.transfer": transfer}):
            with self.assertRaisesRegex(ValueError, "R2_UPLOAD_SIZE_MISMATCH"):
                upload_private_report(str(self.path), 1_800_000_000,
                                      env=self.env)
        client.generate_presigned_url.assert_not_called()


if __name__ == "__main__":
    unittest.main()
