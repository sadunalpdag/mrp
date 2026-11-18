"""Test Gemini model probe without real Google requests or secret disclosure."""
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import btc_gemini_access_probe as probe


def reply(code, data):
    out = Mock()
    out.status_code = code
    out.json.return_value = data
    return out


class GeminiAccessProbeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "DATA_DIR": self.folder.name, "GEMINI_API_KEY": "TEST_SECRET_NOT_TO_PRINT"
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.folder.cleanup()

    def test_supported_new_model_smoke_and_cache(self):
        listed = reply(200, {"models": [
            {"name": "models/gemini-2.5-flash-lite",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.5-flash-lite",
             "supportedGenerationMethods": ["generateContent"]}
        ]})
        completed = reply(200, {
            "candidates": [{"content": {"parts": [{"text": "OK SECRET"}]}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2}
        })
        with patch.object(probe.requests, "get", return_value=listed) as get, \
             patch.object(probe.requests, "post", return_value=completed) as post, \
             patch("builtins.print") as printed:
            out = probe.probe_once()
            self.assertEqual(out["probe_status"], "SUCCESS")
            self.assertEqual(out["candidate"], "gemini-3.5-flash-lite")
            self.assertEqual(out["candidate_status"], 200)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(get.call_count, 1)
            self.assertTrue(post.call_args.args[0].endswith(
                "models/gemini-3.5-flash-lite:generateContent"))
            self.assertNotIn("TEST_SECRET_NOT_TO_PRINT", str(printed.call_args_list))
            self.assertNotIn("OK SECRET", str(printed.call_args_list))
            again = probe.probe_once()
            self.assertEqual(again["probe_status"], "SUCCESS")
            self.assertEqual(get.call_count, 1)
            self.assertEqual(post.call_count, 1)

    def test_list_failure_does_not_send_prompt(self):
        listed = reply(403, {"error": {"status": "PERMISSION_DENIED",
                                      "message": "secret text"}})
        with patch.object(probe.requests, "get", return_value=listed), \
             patch.object(probe.requests, "post") as post, \
             patch("builtins.print") as printed:
            result = probe.probe_once()
        self.assertEqual(result["probe_status"], "LIST_FAILED")
        self.assertEqual(result["models_error"], "PERMISSION_DENIED")
        post.assert_not_called()
        self.assertNotIn("secret text", str(printed.call_args_list))

    def test_unsupported_candidate_does_not_send_prompt(self):
        listed = reply(200, {"models": [
            {"name": "models/gemini-2.5-flash-lite",
             "supportedGenerationMethods": ["generateContent"]}
        ]})
        with patch.object(probe.requests, "get", return_value=listed), \
             patch.object(probe.requests, "post") as post:
            result = probe.probe_once()
        self.assertEqual(result["probe_status"], "NO_SUPPORTED_CANDIDATE")
        post.assert_not_called()

    def test_missing_key_fails_without_http(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), \
             patch.object(probe.requests, "get") as get, \
             patch.object(probe.requests, "post") as post:
            result = probe.probe_once()
        self.assertEqual(result["probe_status"], "KEY_MISSING")
        get.assert_not_called()
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
