"""Regression tests for strict Crash > paper Binance REST priority.

No network, no requests package and no real exchange operations.
"""
import time
import unittest
from unittest.mock import Mock, patch

from crash_rest_budget import IpBudget, RateDeferred, PAPER_CEILING, RESEARCH_CEILING
import crash_paper_rest_priority as gate


class BudgetPriorityTests(unittest.TestCase):
    def test_paper_ceiling_below_crash_and_trade_carpets(self):
        self.assertLess(PAPER_CEILING, RESEARCH_CEILING)
        self.assertLess(RESEARCH_CEILING, 1400)
        moment = [120.0]
        budget = IpBudget(clock=lambda: moment[0], sleep=lambda sec: moment.__setitem__(0, moment[0]+sec))
        budget.observe({"X-MBX-USED-WEIGHT-1M": str(PAPER_CEILING+1)}, 200)
        with self.assertRaises(RateDeferred) as ctx:
            budget.reserve("paper", 1)
        self.assertEqual(ctx.exception.kind, "paper")
        # The same minute still allows the higher-priority Crash history.
        budget.reserve("research", 2)
        self.assertGreater(budget.stats()["reserved_weight"], 0)

    def test_real_exchange_cooldown_denies_all_tiers(self):
        now = [120.0]
        budget = IpBudget(clock=lambda: now[0], sleep=lambda secs: now.__setitem__(0, now[0]+secs))
        budget.observe({"X-MBX-USED-WEIGHT-1M": "4"}, 429, retry_after=100)
        for kind in ("paper", "research", "account", "protect"):
            with self.assertRaises(RateDeferred):
                budget.reserve(kind, 1)

    def test_paper_call_never_reaches_http_when_crash_urgent(self):
        http = Mock()
        gate.set_crash_priority_checker(lambda: True)
        self.addCleanup(gate.set_crash_priority_checker, None)
        with self.assertRaises(gate.PaperRestDeferred) as err:
            gate.paper_get(http, "https://fapi.binance.com/fapi/v1/klines",
                           {"symbol":"BTCUSDT", "interval":"15m"})
        self.assertEqual(err.exception.reason, "FRESH_CRASH_SIGNAL")
        http.get.assert_not_called()

    def test_public_endpoint_allowlist(self):
        http = Mock()
        gate.set_crash_priority_checker(None)
        for url in ("https://fapi.binance.com/fapi/v1/order",
                    "https://evil.example/fapi/v1/klines",
                    "http://fapi.binance.com/fapi/v1/klines",
                    "https://fapi.binance.com/fapi/v3/account",
                    "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT"):
            with self.assertRaises(ValueError):
                gate.paper_get(http, url)
        http.get.assert_not_called()

    def test_paper_call_accounts_actual_binance_weight_and_429(self):
        http = Mock()
        http.get.return_value = Mock(headers={"X-MBX-USED-WEIGHT-1M":"99", "Retry-After":"120"},
                                     status_code=429)
        budget = Mock()
        budget.stats.return_value={"blocked_until":0, "server_weight":0, "reserved_weight":0}
        gate.set_crash_priority_checker(lambda: False)
        self.addCleanup(gate.set_crash_priority_checker, None)
        with patch.object(gate, "BUDGET", budget):
            result = gate.paper_get(http, "https://fapi.binance.com/fapi/v1/klines",
                                    {"symbol":"BTCUSDT", "interval":"5m", "limit":200})
        self.assertEqual(result.status_code, 429)
        budget.reserve.assert_called_once_with("paper", 2)
        budget.observe.assert_called_once_with(http.get.return_value.headers, 429, "120")

    def test_paper_precheck_avoids_paid_scans_when_weight_full(self):
        budget = Mock()
        budget.stats.return_value={"blocked_until":0,
                                   "server_weight":PAPER_CEILING,
                                   "reserved_weight":0}
        gate.set_crash_priority_checker(lambda:False)
        self.addCleanup(gate.set_crash_priority_checker, None)
        with patch.object(gate, "BUDGET", budget):
            self.assertFalse(gate.paper_rest_available())
            budget.stats.return_value={"blocked_until":0, "server_weight":5, "reserved_weight":0}
            self.assertTrue(gate.paper_rest_available())

    def test_deferred_budget_has_no_http_side_effects(self):
        http=Mock()
        budget=Mock()
        budget.reserve.side_effect=RateDeferred("paper", 31, "IP_WEIGHT_SOFT_CEILING")
        gate.set_crash_priority_checker(None)
        with patch.object(gate, "BUDGET", budget):
            with self.assertRaises(gate.PaperRestDeferred) as err:
                gate.paper_get(http,"https://fapi.binance.com/fapi/v1/ticker/price",
                               {"symbol":"BTCUSDT"})
        self.assertEqual(err.exception.reason, "IP_WEIGHT_SOFT_CEILING")
        http.get.assert_not_called()


if __name__=="__main__":
    unittest.main()
