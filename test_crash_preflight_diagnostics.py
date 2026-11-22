"""Fail-closed preflight diagnostics: no Binance network or real orders."""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from crash_night import BinanceBroker, BrokerError


class PreflightDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.broker = BinanceBroker("dummy-key", "dummy-secret")
        self.broker.max_open_positions = 10
        self.broker.max_symbols = 10
        self.broker.max_pending_entries = 15

    def test_preflight_order_unchanged_on_success(self):
        calls = []
        answers = [
            {"dualSidePosition": False},
            [],
            [],
            [],
            {"leverage": 10},
        ]

        def fake_request(method, path, params=None):
            calls.append((method, path))
            return answers.pop(0)

        self.broker.request = fake_request
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.broker.preflight("BREVUSDT"), "BOTH")
        self.assertEqual(calls, [
            ("GET", "/fapi/v1/positionSide/dual"),
            ("GET", "/fapi/v3/positionRisk"),
            ("GET", "/fapi/v1/openOrders"),
            ("GET", "/fapi/v1/openAlgoOrders"),
            ("POST", "/fapi/v1/leverage"),
        ])

    def test_preflight_failure_logs_exact_stage_without_buy(self):
        calls = []

        def fail_positions(method, path, params=None):
            calls.append((method, path))
            if path == "/fapi/v1/positionSide/dual":
                return {"dualSidePosition": False}
            raise BrokerError("UNKNOWN", uncertain=True)

        self.broker.request = fail_positions
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(BrokerError):
            self.broker.preflight("BREVUSDT")
        self.assertIn("stage=02_positions", output.getvalue())
        self.assertIn("endpoint=/fapi/v3/positionRisk", output.getvalue())
        self.assertIn("code=UNKNOWN", output.getvalue())
        self.assertNotIn("dummy-secret", output.getvalue())
        self.assertEqual(len(calls), 2)

    def test_request_timeout_logs_only_safe_metadata(self):
        import requests
        output = io.StringIO()
        with patch("crash_rest_budget.BUDGET.reserve"), \
             patch("requests.request", side_effect=requests.Timeout("secret in signed URL")):
            with redirect_stdout(output), self.assertRaises(BrokerError) as exc:
                self.broker.request("GET", "/fapi/v3/positionRisk")
        self.assertEqual(exc.exception.code, "UNKNOWN")
        self.assertIn("kind=transport", output.getvalue())
        self.assertIn("exception=Timeout", output.getvalue())
        self.assertIn("endpoint=/fapi/v3/positionRisk", output.getvalue())
        self.assertNotIn("secret in signed URL", output.getvalue())
        self.assertNotIn("dummy-key", output.getvalue())
        self.assertNotIn("dummy-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
