"""Fake-exchange tests: no live Binance network or real orders."""
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

from profit_lock_live import LiveProfitLock


class FakeBroker:
    def __init__(self):
        self.balance = Decimal("800")
        self.positions_data = []
        self.regular = []
        self.algo = []
        self.writes = []
        self.income = []

    def request(self, method, path, params=None, budget_priority=None):
        if method == "GET":
            if path == "/fapi/v3/account":
                return {"multiAssetsMargin": False,
                        "totalMarginBalance": str(self.balance),
                        "totalWalletBalance": str(self.balance),
                        "totalUnrealizedProfit": "0",
                        "assets": [{"asset": "USDT", "marginBalance": str(self.balance)}]}
            if path == "/fapi/v1/income":
                return list(self.income)
            if path == "/fapi/v1/openOrders":
                return list(self.regular)
            if path == "/fapi/v1/openAlgoOrders":
                return list(self.algo)
        if method == "POST" and path == "/fapi/v1/order":
            self.writes.append((method, path, dict(params)))
            self.regular.append({"symbol": params["symbol"], "side": params["side"],
                                 "clientOrderId": params["newClientOrderId"],
                                 "reduceOnly": params.get("reduceOnly") == "true"})
            return {"orderId": 101}
        raise AssertionError((method, path))

    def positions(self):
        return list(self.positions_data)

    def info(self, symbol, budget_priority=None):
        return {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.001"}]}

    def cancel_entry(self, symbol, client_id):
        self.writes.append(("DELETE", symbol, client_id))
        self.regular = [x for x in self.regular if x.get("clientOrderId") != client_id]

    def cancel_owned_tp(self, symbol, client_id):
        self.writes.append(("DELETE_TP", symbol, client_id))
        self.regular = [x for x in self.regular if x.get("clientOrderId") != client_id]


class FakeExecutor:
    def __init__(self, clock):
        self.clock = lambda: clock[0]
        self.state = {"orders": {}, "halt": None, "paused": False, "outbox": []}
        self.broker = FakeBroker()
        self.enabled = threading.Event()
        self.enabled.set()
        self.preflight_retries = {}
        self.messages = []
        self.saved = []

    def save(self):
        self.saved.append(json.loads(json.dumps(self.state)))

    def alert(self, msg):
        self.messages.append(msg)


class LiveLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"DATA_DIR": self.temp.name,
                            "CRASH_FIXED_PROFIT_LOCK_ENABLED": "1",
                            "DAILY_PROFIT_PAPER_TARGET_PCT": "10"})
        self.env.start()
        self.addCleanup(self.env.stop)
        with open(os.path.join(self.temp.name, "daily_profit_lock_paper.json"), "w") as f:
            json.dump({"mode": "PAPER_ONLY", "target": "880"}, f)
        self.clock = [datetime(2026, 11, 20, 15, tzinfo=timezone.utc).timestamp()]
        self.ex = FakeExecutor(self.clock)
        self.ctrl = LiveProfitLock(self.ex)

    def advance(self):
        self.clock[0] += 130

    def test_target_persists_missed_days(self):
        self.assertTrue(self.ctrl.entries_blocked())
        self.ctrl.step()
        self.assertFalse(self.ctrl.entries_blocked())
        self.assertEqual(self.ctrl.state["target"], "880")
        for day in range(1, 4):
            self.clock[0] += 86400
            self.ex.broker.balance = Decimal("870")
            self.ctrl.step()
            self.assertEqual(self.ctrl.state["target"], "880")
            self.assertEqual(self.ctrl.state["phase"], "WAIT_TARGET")
        self.assertEqual(self.ex.broker.writes, [])

    def test_trigger_lock_and_limit_only(self):
        self.ex.broker.positions_data = [{
            "symbol": "ETHUSDT", "positionSide": "BOTH",
            "positionAmt": "1.123", "markPrice": "100.01"
        }]
        self.ctrl.step()
        self.advance()
        self.ex.broker.balance = Decimal("880")
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "CLOSING")
        self.assertFalse(self.ex.enabled.is_set())
        self.assertEqual(len(self.ex.broker.writes), 0)
        self.advance()
        self.ctrl.step()
        self.assertEqual(len(self.ex.broker.writes), 1)
        method, endpoint, p = self.ex.broker.writes[0]
        self.assertEqual((method, endpoint, p["type"]), ("POST", "/fapi/v1/order", "LIMIT"))
        self.assertEqual(p["reduceOnly"], "true")
        self.assertEqual(p["side"], "SELL")
        self.advance()
        self.ctrl.step()
        self.assertEqual(len(self.ex.broker.writes), 1)  # never duplicate
        self.assertTrue(self.ctrl.entries_blocked())

    def test_untracked_manual_buy_blocks_close(self):
        self.ex.broker.regular = [{
            "symbol": "ETHUSDT", "side": "BUY", "type": "LIMIT", "clientOrderId": "manual"
        }]
        self.ex.broker.positions_data = [{
            "symbol": "ETHUSDT", "positionSide": "BOTH",
            "positionAmt": "1", "markPrice": "100"
        }]
        self.ctrl.step()
        self.advance()
        self.ex.broker.balance = Decimal("880")
        self.ctrl.step()
        self.advance()
        self.ctrl.step()
        self.assertEqual(self.ex.broker.writes, [])
        self.assertTrue(self.ctrl.entries_blocked())

    def test_cashflow_does_not_trigger(self):
        self.ctrl.step()
        self.advance()
        self.ex.broker.balance = Decimal("900")
        self.ex.broker.income = [{
            "incomeType": "TRANSFER", "asset": "USDT", "time": int(self.clock[0]*1000),
            "income": "100", "tranId": 123
        }]
        self.ctrl.step()
        self.assertEqual(self.ctrl.state["phase"], "WAIT_TARGET")
        self.assertEqual(self.ctrl.state["cashflow_total"], "100")

    def test_no_live_emission_without_explicit_opt_in(self):
        os.environ["CRASH_FIXED_PROFIT_LOCK_ENABLED"] = "0"
        self.assertFalse(LiveProfitLock(self.ex).active())


if __name__ == "__main__":
    unittest.main()
