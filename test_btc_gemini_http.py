"""Offline Gemini HTTP error diagnostics: no keys, orders or network calls."""
import os
import time
import unittest
from unittest.mock import Mock, patch

import btc_range_paper as paper


class GeminiDiagnosticTests(unittest.TestCase):
    def _response(self, code, status):
        response = Mock()
        response.status_code = code
        response.headers = {"Retry-After": "120"}
        response.json.return_value = {"error": {
            "code": code, "status": status, "message": "DO NOT LOG SECRET DATA"
        }}
        return response

    def test_http_429_logs_status_without_response_body_or_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "DONT_LOG_THIS_API_KEY"}), \
             patch.object(paper.requests, "post",
                          return_value=self._response(429, "RESOURCE_EXHAUSTED")):
            with self.assertLogs("btc_range_paper", level="WARNING") as logs:
                with self.assertRaises(paper.GeminiHTTPFailure) as captured:
                    paper._gemini({"symbol": "BTCUSDT"}, "hourly")
        self.assertEqual(captured.exception.code, 429)
        self.assertIn("code=429", logs.output[0])
        self.assertIn("RESOURCE_EXHAUSTED", logs.output[0])
        self.assertNotIn("SECRET", str(logs.output))
        self.assertNotIn("DO NOT LOG", str(logs.output))

    def test_http_403_is_diagnosable_and_delayed(self):
        r = paper._gemini_error_record(paper.GeminiHTTPFailure(403, "PERMISSION_DENIED"))
        self.assertEqual(r["http_code"], 403)
        self.assertEqual(r["http_reason"], "PERMISSION_DENIED")
        self.assertGreaterEqual(r["retry_after_ms"], int(time.time()*1000) + 5*3600*1000)

    def test_legacy_failed_reference_can_retry(self):
        hour_ms = int(time.time() * 1000)
        existing = {"schema": 1, "created_ms": hour_ms - 2*3600000,
                    "gemini": {"status": "ERROR", "error": "HTTPError"},
                    "anchor_price": 100, "start_utc": "2026-11-01T00:00:00Z",
                    "compact_candles": [[0, 0, 0, 0, 0, 1]],
                    "historical_patterns": {}}
        with patch.dict(os.environ, {"BTC_RANGE_AI_ENABLED": "1"}), \
             patch.object(paper, "_gemini", return_value={
                 "status": "COMPLETE", "analysis": {"patterns": []}}) as send, \
             patch.object(paper, "_save_reference") as save:
            result = paper.ensure_reference(hour_ms, lambda: False, existing)
        self.assertEqual(result["gemini"]["status"], "COMPLETE")
        self.assertEqual(send.call_count, 1)
        self.assertEqual(save.call_count, 1)

    def test_retry_backoff_avoids_immediate_repeat(self):
        hour_ms = int(time.time() * 1000)
        existing = {"schema": 1, "created_ms": hour_ms - 3600000,
                    "gemini": {"status": "ERROR", "retry_after_ms": hour_ms + 3600000}}
        with patch.dict(os.environ, {"BTC_RANGE_AI_ENABLED": "1"}), \
             patch.object(paper, "_gemini") as send:
            paper.ensure_reference(hour_ms, lambda: False, existing)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
