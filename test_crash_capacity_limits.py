"""Regressions for independent real-entry caps; no exchange network calls."""
import tempfile
import unittest
from unittest.mock import Mock
from crash_night import BinanceBroker, NightExecutor
from crash_telegram_control import TelegramControl


class CapacityTests(unittest.TestCase):
    def broker(self, active_count=0, pending_count=0, closing_count=0, fail_on=None):
        broker=object.__new__(BinanceBroker)
        broker.max_symbols=50
        broker.max_open_positions=10
        broker.max_pending_entries=15
        calls=[]
        positions=[{"symbol":f"POS{i}USDT","positionAmt":"1"}
                   for i in range(active_count)]
        entries=[{"symbol":f"ENTRY{i}USDT","side":"BUY",
                  "type":"LIMIT","reduceOnly":False}
                 for i in range(pending_count)]
        closing=[{"symbol":f"POS{i}USDT","side":"SELL",
                  "reduceOnly":True} for i in range(closing_count)]
        def request(method,path,params=None,**kwargs):
            calls.append((method,path))
            if path=="/fapi/v1/positionSide/dual":
                return {"dualSidePosition":False}
            if path=="/fapi/v3/positionRisk":return positions
            if path=="/fapi/v1/openOrders":return entries+closing
            if path=="/fapi/v1/openAlgoOrders":return []
            if path=="/fapi/v1/leverage":return {"leverage":10}
            raise AssertionError("UNEXPECTED_REQUEST "+path)
        broker.request=request
        return broker,calls

    def test_ten_positions_block_before_open_order_requests(self):
        broker,calls=self.broker(active_count=10)
        with self.assertRaisesRegex(ValueError,"MAX_OPEN_POSITIONS_REACHED"):
            broker.preflight("NEWUSDT")
        self.assertNotIn(("GET","/fapi/v1/openOrders"),calls)
        self.assertNotIn(("POST","/fapi/v1/leverage"),calls)

    def test_fifteen_pending_entries_block_without_changing_orders(self):
        broker,calls=self.broker(active_count=3,pending_count=15)
        with self.assertRaisesRegex(ValueError,"MAX_PENDING_ENTRY_ORDERS_REACHED"):
            broker.preflight("NEWUSDT")
        self.assertNotIn(("POST","/fapi/v1/leverage"),calls)
        self.assertEqual(broker.last_capacity_snapshot["pending_entries"],15)

    def test_closing_tp_orders_do_not_occupy_pending_entry_limit(self):
        broker,calls=self.broker(active_count=9,pending_count=14,closing_count=16)
        self.assertEqual(broker.preflight("NEWUSDT"),"BOTH")
        self.assertIn(("POST","/fapi/v1/leverage"),calls)
        self.assertEqual(broker.last_capacity_snapshot["pending_entries"],14)

    def test_controls_and_upgrade_persist_without_resetting_used_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            b=Mock()
            e=NightExecutor(directory,b,Mock(),Mock(),max_daily=50)
            self.assertEqual((e.max_daily,e.max_open_positions,e.max_pending_entries),
                             (50,10,15))
            e.apply_control("counter:7")
            e.apply_control("open_limit:12")
            e.apply_control("pending_limit:18")
            self.assertEqual(e.daily_count(),7)
            self.assertEqual((b.max_open_positions,b.max_pending_entries),(12,18))
            self.assertIn("açık pozisyon limiti 12",e.summary())
            e.guard.close()
            restarted=NightExecutor(directory,b,Mock(),Mock(),max_daily=50)
            self.assertEqual((restarted.max_daily,restarted.max_open_positions,
                              restarted.max_pending_entries),(50,12,18))
            self.assertEqual(restarted.daily_count(),7)
            self.assertEqual(restarted.state["orders"],{})
            self.assertIn("⚠️",restarted.apply_control("open_limit:51"))
            self.assertIn("⚠️",restarted.apply_control("pending_limit:0"))
            restarted.guard.close()

    def test_telegram_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            tg=TelegramControl(directory,"fake","123",None,Mock(),Mock(),Mock(),
                               lambda:1000,Mock(),counter_only=True,require_owner=False)
            def msg(value):
                return {"message":{"chat":{"id":123},"from":{"id":123,"is_bot":False},
                                   "date":1000,"text":value}}
            self.assertEqual(tg.command(msg("/acik 10")),"open_limit:10")
            self.assertEqual(tg.command(msg("/pozisyon 11")),"open_limit:11")
            self.assertEqual(tg.command(msg("/bekleyen 15")),"pending_limit:15")
            self.assertEqual(tg.command(msg("/limit 50")),"limit:50")
            self.assertIsNone(tg.command(msg("/acik 10 extra")))


if __name__=="__main__":
    unittest.main()
