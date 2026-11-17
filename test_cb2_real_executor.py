"""CB2 real executor offline tests. Fake broker: NO real Binance orders."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from crash_night import BrokerError
from cb2_real_executor import Executor


def exchange_info(symbol):
    return {"symbol": symbol, "status": "TRADING",
            "contractType": "PERPETUAL",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01",
                 "minPrice": "0.01", "maxPrice": "999999"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001",
                 "minQty": "0.001", "maxQty": "100000"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"}
            ]}


class FakeBroker:
    def __init__(self):
        self.asks = "123.456"
        self.active_symbols = set()
        self.pending_symbols = set()
        self.entry = None
        self.tp_order = None
        self.buy_calls = []
        self.tp_calls = []
        self.cancel_calls = []
        self.fail_buy = False
    def info(self, symbol):
        return exchange_info(symbol)
    def preflight(self, symbol):
        if len(self.active_symbols) >= self.max_open_positions:
            raise ValueError("MAX_OPEN_POSITIONS_REACHED")
        if len(self.pending_symbols) >= self.max_pending_entries:
            raise ValueError("MAX_PENDING_ENTRY_ORDERS_REACHED")
        if symbol in self.active_symbols or symbol in self.pending_symbols:
            raise ValueError("SYMBOL_ALREADY_ACTIVE")
        return "BOTH"
    def request(self, method, path, params=None, signed=True):
        if method == "GET" and path == "/fapi/v1/ticker/bookTicker":
            return {"symbol": params["symbol"], "askPrice": self.asks}
        raise AssertionError("unexpected endpoint: " + method + " " + path)
    def buy(self, symbol, p, client_id):
        self.buy_calls.append((symbol, p, client_id))
        if self.fail_buy:
            raise BrokerError("UNKNOWN", uncertain=True)
        self.entry = {
            "symbol": symbol, "clientOrderId": client_id,
            "side": "BUY", "type": "LIMIT", "status": "FILLED",
            "executedQty": p["quantity"], "avgPrice": "123.45",
            "cumQuote": "0"
        }
        return {"status": "FILLED"}
    def order(self, symbol, client_id):
        if self.entry is not None and client_id == self.entry["clientOrderId"]:
            return dict(self.entry)
        if self.tp_order is not None and client_id == self.tp_order["clientOrderId"]:
            return dict(self.tp_order)
        raise BrokerError("-2013")
    def tp(self, symbol, qty, price, client_id, position_side="BOTH"):
        self.tp_calls.append((symbol, qty, price, client_id))
        self.tp_order = {
            "symbol": symbol, "clientOrderId": client_id, "side": "SELL",
            "positionSide": position_side, "origQty": qty, "price": price,
            "status": "NEW"
        }
        return dict(self.tp_order)
    def cancel_entry(self, symbol, client_id):
        self.cancel_calls.append((symbol, client_id))
        if self.entry and self.entry["clientOrderId"] == client_id:
            self.entry["status"] = "CANCELED"
        return {}


class ExecutorTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "CB2_REAL_TRADING_ENABLED": "1", "BOT_TOKEN": "fake",
            "CHAT_ID": "fake"
        })
        self.env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.crash_dir = self.root / "crash"
        self.crash_dir.mkdir()
        self.now = [datetime(2026, 11, 17, 4, 30, tzinfo=timezone.utc).timestamp()]
        self.snapshot = {"halt": None, "paused": False, "orders": {},
            "trade_settings": {
                "max_daily": 50, "max_symbols": 28, "max_open_positions": 11,
                "max_pending_entries": 14, "position_usdt": "314"}}
        self._save_crash()
        self.broker = FakeBroker()
        self.messages = []
        def deliver(msg):
            self.messages.append(msg)
            return {"ok": True}
        self.executor = Executor(self.root / "cb2", self.crash_dir,
                                  self.broker, deliver, clock=lambda: self.now[0])
        self.candidate = {
            "id": "signal-1", "symbol": "TESTUSDT",
            "archetype": "pattern_breakout", "sqs": 80,
            "observed_at": datetime.fromtimestamp(
                self.now[0] - 10, timezone.utc).isoformat(),
            "filters_passed": True,
            "indicators": {"adx14": 27, "change24_pct": 12,
                           "macd_histogram": 0.03}}
    def tearDown(self):
        self.executor.guard.close()
        self.tmp.cleanup()
        self.env.stop()
    def _save_crash(self):
        (self.crash_dir / "crash_night_state.json").write_text(json.dumps(self.snapshot))
    def test_real_limit_buy_and_fill_verified_tp(self):
        self.assertTrue(self.executor.enqueue(self.candidate))
        self.executor.step()
        rec = self.executor.state["orders"]["signal-1"]
        self.assertEqual(rec["phase"], "PROTECTED")
        self.assertEqual(len(self.broker.buy_calls), 1)
        self.assertEqual(self.broker.buy_calls[0][1]["price"], "123.46")
        self.assertEqual(self.broker.tp_calls[0][2], "124.69")
        self.assertEqual(self.broker.buy_calls[0][1]["quantity"], "2.543")
        self.assertTrue(rec["entry_id"].startswith("cbm"))
        self.assertFalse(self.executor.enqueue(self.candidate))
        self.broker.tp_order["status"] = "FILLED"
        self.now[0] += 46
        self.executor.step()
        self.assertEqual(rec["phase"], "DONE")
        self.assertEqual(len(self.broker.buy_calls), 1)
    def test_crash_paused_prevents_trade(self):
        self.snapshot["paused"] = True
        self._save_crash()
        self.assertTrue(self.executor.enqueue(self.candidate))
        self.executor.step()
        self.assertEqual(self.executor.state["orders"]["signal-1"]["phase"], "BLOCKED")
        self.assertEqual(self.broker.buy_calls, [])
    def test_reserve_crash_slot_at_ten_positions(self):
        self.broker.active_symbols = set("COIN"+str(i) for i in range(10))
        self.assertTrue(self.executor.enqueue(self.candidate))
        self.executor.step()
        self.assertEqual(self.executor.state["orders"]["signal-1"]["phase"], "BLOCKED")
        self.assertEqual(self.broker.buy_calls, [])
    def test_uncertain_buy_never_resubmitted(self):
        self.broker.fail_buy = True
        self.assertTrue(self.executor.enqueue(self.candidate))
        self.executor.step()
        rec = self.executor.state["orders"]["signal-1"]
        self.assertEqual(rec["phase"], "ENTRY_UNKNOWN")
        self.assertEqual(len(self.broker.buy_calls), 1)
        self.now[0] += 40
        self.executor.step()
        self.assertEqual(len(self.broker.buy_calls), 1)
        self.assertEqual(rec["phase"], "ENTRY_UNKNOWN")
    def test_unfilled_gtc_expires_and_cancels_owned_order(self):
        self.assertTrue(self.executor.enqueue(self.candidate))
        self.broker.buy = self._unfilled_buy
        self.executor.step()
        rec = self.executor.state["orders"]["signal-1"]
        self.assertEqual(rec["phase"], "ENTRY_UNKNOWN")
        self.now[0] += 8 * 3600 + 50
        self.executor.step()
        self.assertEqual(len(self.broker.cancel_calls), 1)
        self.now[0] += 20
        self.executor.step()
        self.assertEqual(rec["phase"], "NO_FILL")
        self.assertEqual(self.broker.tp_calls, [])
    def _unfilled_buy(self, symbol, p, client_id):
        self.broker.buy_calls.append((symbol, p, client_id))
        self.broker.entry = {
            "symbol": symbol, "clientOrderId": client_id,
            "side": "BUY", "type": "LIMIT", "status": "NEW",
            "executedQty": "0", "avgPrice": "0"
        }
        return {"status": "NEW"}
    def test_persistent_ledger_survives_executor_restart(self):
        self.assertTrue(self.executor.enqueue(self.candidate))
        self.executor.step()
        self.assertEqual(len(self.broker.buy_calls), 1)
        self.executor.guard.close()
        recovered = Executor(self.root / "cb2", self.crash_dir,
                              self.broker, lambda msg: {"ok": True},
                              clock=lambda: self.now[0])
        self.executor = recovered
        self.assertFalse(recovered.enqueue(self.candidate))
        recovered.step()
        self.assertEqual(len(self.broker.buy_calls), 1)


if __name__ == "__main__":
    unittest.main()
