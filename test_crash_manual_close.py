"""Offline account-wide manual Close recovery and durable archive tests.

No network, API credentials, live orders or deployment.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import Mock

from crash_night import NightExecutor, BrokerError, BinanceBroker, day
from crash_audit_archive import append_trade_event
from crash_close_recovery import confirmed_flat, account_safe_to_resume
from test_crash_services import FakeBroker


class ReadOnlyBroker(FakeBroker):
    is_closing_order = staticmethod(BinanceBroker.is_closing_order)

    def __init__(self):
        super().__init__()
        self.position_rows = []
        self.open_rows = []
        self.algo_rows = []
        self.close_rows = {}
        self.cooldown_until = 0

    def positions(self):
        return list(self.position_rows)

    def open_entry_orders(self):
        return list(self.open_rows)

    def open_algo_orders(self):
        return list(self.algo_rows)

    def closing_orders(self, symbol):
        return list(self.close_rows.get(symbol, []))


def closing_tp(symbol, qty="5"):
    return dict(symbol=symbol, side="SELL", positionSide="BOTH", type="LIMIT",
                reduceOnly=True, status="NEW", origQty=qty, executedQty="0", price="10.1")


def open_position(symbol, qty="5"):
    return dict(symbol=symbol, positionSide="BOTH", positionAmt=qty, entryPrice="10")


class ManualCloseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1793669400.0
        self.b = ReadOnlyBroker()
        self.send = Mock(return_value={"ok": True})
        self.e = NightExecutor(self.tmp.name, self.b, self.send, Mock(),
                               clock=lambda: self.now, enabled=True)
        self.addCleanup(self.e.guard.close)

    def setup_order(self, symbol="AKEUSDT", status="CANCELED"):
        signal = symbol + "_1"
        order = dict(symbol=symbol, signal_id=signal, day=day(self.now), phase="PROTECTED",
                     position_side="BOTH", entry_id=signal+"e", tp_id=signal+"t",
                     quantity="5", filled_qty="5", price="10", tp="10.1")
        self.e.state["orders"][signal] = order
        self.b.orders[order["tp_id"]] = dict(symbol=symbol, side="SELL", status=status,
                                            positionSide="BOTH", origQty="5", price="10.1",
                                            reduceOnly=True)
        self.e.save()
        return order

    def test_manual_close_any_symbol_is_recorded_not_tp(self):
        order = self.setup_order("ANYCOINUSDT")
        self.assertTrue(self.e.reconcile(order))
        self.assertEqual(order["phase"], "CLOSED_EXTERNALLY")
        self.assertNotEqual(order["phase"], "DONE")
        history = os.path.join(self.e.audit_directory, "2026-11-03.jsonl")
        with open(history, encoding="utf-8") as f:
            events = [json.loads(x) for x in f]
        self.assertTrue(any(e["kind"] == "CLOSED_EXTERNALLY" and
                            e["symbol"] == "ANYCOINUSDT" for e in events))
        self.assertFalse(any(e["kind"] == "TP_FILLED" for e in events))

    def test_old_global_halt_can_recover_only_on_explicit_on(self):
        a = self.setup_order("AKEUSDT")
        g = self.setup_order("GUSDT")
        self.e.state["halt"] = "TP DOĞRULANAMADI\nKod: TP_NOT_ACTIVE"
        self.e.save()
        self.e.reconcile(a)
        self.e.reconcile(g)
        self.assertTrue(self.e.state["halt"])
        message = self.e.apply_control("on")
        self.assertIn("AÇIK", message)
        self.assertIsNone(self.e.state["halt"])
        self.assertEqual(a["phase"], "CLOSED_EXTERNALLY")
        self.assertEqual(g["phase"], "CLOSED_EXTERNALLY")
        self.assertFalse(self.b.buys)
        self.assertFalse(self.b.tps)

    def test_partial_manual_close_keeps_halt(self):
        order = self.setup_order()
        self.b.position_rows = [open_position(order["symbol"], "2")]
        self.assertFalse(self.e.reconcile(order))
        self.assertTrue(self.e.state["halt"])
        self.assertEqual(order["phase"], "PROTECTED")
        self.assertIn("Açılmadı", self.e.apply_control("on"))
        self.assertTrue(self.e.state["halt"])

    def test_pending_buy_same_symbol_prevents_false_flat(self):
        order = self.setup_order()
        self.b.open_rows = [dict(symbol=order["symbol"], side="BUY", status="NEW")]
        self.assertFalse(confirmed_flat(self.b, order["symbol"]))
        self.assertFalse(self.e.reconcile(order))
        self.assertEqual(order["phase"], "PROTECTED")

    def test_orphan_algo_tp_prevents_false_flat(self):
        order = self.setup_order()
        self.b.close_rows[order["symbol"]] = [dict(symbol=order["symbol"],
                 side="SELL", status="NEW", type="TAKE_PROFIT_MARKET")]
        self.assertFalse(self.e.reconcile(order))
        self.assertEqual(order["phase"], "PROTECTED")

    def test_rate_limit_during_flat_verification_keeps_halt(self):
        order = self.setup_order()
        self.b.positions = Mock(side_effect=BrokerError("-1003"))
        self.assertFalse(self.e.reconcile(order))
        self.assertEqual(order["phase"], "PROTECTED")
        self.e.state["halt"] = "TP DOĞRULANAMADI\nKod: TP_NOT_ACTIVE"
        self.assertIn("Açılmadı", self.e.apply_control("on"))

    def test_no_tp_query_record_then_confirmed_flat(self):
        order = self.setup_order("OTHERSYMBOL")
        self.b.order = Mock(side_effect=BrokerError("-2013"))
        self.assertTrue(self.e.reconcile(order))
        self.assertEqual(order["phase"], "CLOSED_EXTERNALLY")

    def test_unrelated_pending_order_blocks_global_resume(self):
        self.setup_order("AUSDT")
        self.e.state["halt"] = "TP DOĞRULANAMADI\nKod: TP_NOT_ACTIVE"
        self.b.open_rows = [dict(symbol="BUSDT", side="BUY", status="NEW")]
        self.assertIn("Açılmadı", self.e.apply_control("on"))
        self.assertTrue(self.e.state["halt"])

    def test_other_coin_unprotected_keeps_global_lock(self):
        self.setup_order("AUSDT")
        self.e.state["halt"] = "TP DOĞRULANAMADI\nKod: TP_NOT_ACTIVE"
        self.b.position_rows = [open_position("BUSDT")]
        self.assertIn("Açılmadı", self.e.apply_control("on"))
        self.assertTrue(self.e.state["halt"])

    def test_protected_other_coin_with_valid_tp_can_resume(self):
        self.setup_order("AUSDT")
        p = self.setup_order("BUSDT", status="NEW")
        self.b.position_rows = [open_position("BUSDT")]
        self.b.open_rows = [closing_tp("BUSDT")]
        self.b.close_rows["BUSDT"] = [closing_tp("BUSDT")]
        self.e.state["halt"] = "TP DOĞRULANAMADI\nKod: TP_NOT_ACTIVE"
        message = self.e.apply_control("on")
        self.assertIn("AÇIK", message)
        self.assertEqual(p["phase"], "PROTECTED")
        self.assertIsNone(self.e.state["halt"])

    def test_unrelated_short_blocks_global_resume(self):
        self.e.state["halt"] = "TP DOĞRULANAMADI\nKod: TP_NOT_ACTIVE"
        self.b.position_rows = [dict(symbol="SHORTUSDT", positionSide="SHORT",
                                     positionAmt="-2", entryPrice="10")]
        self.assertIn("Açılmadı", self.e.apply_control("on"))
        self.assertTrue(self.e.state["halt"])

    def test_error_halt_other_than_tp_cannot_be_unlocked(self):
        self.e.state["halt"] = "EXECUTOR_ERROR"
        self.assertIn("Açılmadı", self.e.apply_control("on"))
        self.assertEqual(self.e.state["halt"], "EXECUTOR_ERROR")

    def test_ledger_appends_without_overwriting_past_runs(self):
        out = os.path.join(self.tmp.name, "daily")
        path = append_trade_event(out, {"kind": "SIGNAL", "symbol": "AUSDT"}, self.now)
        append_trade_event(out, {"kind": "TP_FILLED", "symbol": "AUSDT"}, self.now)
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual([r["kind"] for r in rows], ["SIGNAL", "TP_FILLED"])
        self.assertTrue(all(r["schema_version"] == 1 for r in rows))

    def test_incomplete_account_data_fails_closed(self):
        self.b.positions = Mock(return_value=None)
        with self.assertRaises(ValueError):
            confirmed_flat(self.b, "AUSDT")
        self.b.positions = Mock(return_value=[])
        self.b.open_algo_orders = Mock(return_value=None)
        with self.assertRaises(ValueError):
            account_safe_to_resume(self.b, [])


if __name__ == "__main__":
    unittest.main()
