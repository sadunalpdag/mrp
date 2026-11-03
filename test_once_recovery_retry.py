"""Exercise one-shot startup recovery without Binance credentials or network."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

def recovery_fn():
    tree = ast.parse(Path(__file__).with_name("ema.py").read_text(encoding="utf-8"))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                and x.name == "_crash_one_time_fresh_start")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "ema.py", "exec"), scope)
    return scope[node.name]

class OfflineBroker:
    def __init__(self):
        self.orders = [{"symbol":"PONSUSDT", "side":"BUY", "type":"LIMIT", "status":"NEW"}]
        self.positions_data = []
        self.calls = []
    def positions(self):
        self.calls.append("positions")
        return self.positions_data
    def open_entry_orders(self):
        self.calls.append("orders")
        return self.orders
    def request(self, method, path):
        self.calls.append((method, path))
        if (method, path) != ("GET", "/fapi/v1/openAlgoOrders"):
            raise AssertionError("No exchange writes allowed")
        return []

class OneTimeRecoveryTest(unittest.TestCase):
    def test_denial_then_success_then_never_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "night_execution"
            directory.mkdir()
            state_path = directory / "crash_night_state.json"
            original = {"halt":"STATE_HISTORY_UNVERIFIED", "paused":True,
                        "orders":{"old":{"phase":"DONE","day":"2026-11-03"}},
                        "outbox":[],
                        "trade_settings":{"position_usdt":"500","max_daily":20,"max_symbols":20}}
            state_path.write_text(json.dumps(original))
            broker = OfflineBroker()
            recover = recovery_fn()
            args = dict(expected_mount=root, require_mount=False)
            self.assertFalse(recover(str(directory), broker, Mock(), **args))
            self.assertEqual(json.loads(state_path.read_text()), original)
            self.assertTrue((directory / ".crash_once_reset_attempt_20261103").is_file())
            broker.orders = []
            self.assertTrue(recover(str(directory), broker, Mock(), **args))
            state = json.loads(state_path.read_text())
            self.assertIsNone(state["halt"])
            self.assertFalse(state["paused"])
            self.assertEqual(state["orders"], {})
            self.assertEqual(state["trade_settings"], original["trade_settings"])
            self.assertTrue(state["one_time_fresh_start_20261103"])
            backups = list(directory.glob("crash_night_state.json.pre_once_*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), original)
            calls_before = len(broker.calls)
            self.assertFalse(recover(str(directory), broker, Mock(), **args))
            self.assertEqual(len(broker.calls), calls_before)
            self.assertEqual(len(list(directory.glob("crash_night_state.json.pre_once_*"))), 1)

    def test_open_position_without_tp_cannot_rearm(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "night_execution"
            directory.mkdir()
            p = directory / "crash_night_state.json"
            p.write_text(json.dumps({"halt":"STATE_HISTORY_UNVERIFIED",
                                    "orders":{}, "outbox":[], "paused":True}))
            broker = OfflineBroker()
            broker.orders = []
            broker.positions_data = [{
                "symbol":"PONSUSDT", "positionSide":"BOTH", "positionAmt":"5",
                "entryPrice":"0.1"}]
            self.assertFalse(recovery_fn()(str(directory), broker, Mock(),
                             expected_mount=root, require_mount=False))
            self.assertEqual(json.loads(p.read_text())["halt"],"STATE_HISTORY_UNVERIFIED")

if __name__ == "__main__":
    unittest.main()
