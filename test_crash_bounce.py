"""Offline Crash Bounce V2 integration checks (no network or real orders)."""
import ast
import math
import json
import threading
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class CrashBounceTests(unittest.TestCase):
    def setUp(self):
        names={"_crash_signal_score","crash_bounce_plan","run_crash_bounce_alerts",
               "_crash_send_pending","_crash_initialize_tracking","tg_send"}
        tree=ast.parse(Path(__file__).with_name("ema.py").read_text())
        subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                                and n.name in names],type_ignores=[])
        self.now=2_000_000_000
        self.g=dict(Decimal=Decimal,math=math,time=SimpleNamespace(time=lambda:self.now),
                    CRASH_BOUNCE_ENTRY_BUFFER=Decimal(".003"),
                    CRASH_BOUNCE_TP_PCT=Decimal(".01"),CRASH_BOUNCE_TOP_LOSERS=20,
                    CRASH_BOUNCE_DROP_PCT=12.,
                    CRASH_BOUNCE_FILTER_VERSION="five_day_entry_v2_only",
                    CRASH_V2_SIGNAL_TTL_SECONDS=90,
                    CRASH_BOUNCE_ONLY_TELEGRAM=True,_TG_COMMAND_CONTEXT=threading.local(),
                    BOT_TOKEN="test",CHAT_ID="test",requests=Mock(),log=Mock(),
                    _crash_save_state=Mock(),_crash_append_data=Mock(),
                    _crash_queue_history=Mock(),_crash_track_results=Mock(),
                    _crash_review_pending_v2=Mock(),_crash_apply_filter_migration=Mock(),
                    _crash_features=Mock(return_value={"rsi14_wilder":24,"ema25_slope_pct":-.1}),
                    BinanceRateLimiter=SimpleNamespace(is_banned=lambda:False))
        exec(compile(subset,"ema.py","exec"),self.g)
        self.rows=[[int((self.now-(16-i)*900)*1000),"1","1.1",".99","1","1",
                    int((self.now-(15-i)*900)*1000)-1] for i in range(16)]
        self.tickers=[]
        self.g["_CRASH_WS_CACHE"]=SimpleNamespace(snapshot=lambda now:self.tickers,
                                                   reconcile=Mock())
        self.g["_crash_get_candles"]=lambda sym:self.rows
        self.receipt=Mock(return_value={"ok":True,"attempt_at":self.now,
                                        "accepted_at":self.now,"message_id":31})
        self.g["_crash_telegram_receipt"]=self.receipt

        def qualify(state,record,now,allow_rest=False):
            record.update(v2_status="STRONG_V2",v2_features={
                "recovery_4h_pct":1.5,"rsi14":27.0,"candles":480,
                "return_5d_pct":-17.0})
            state["symbols"][record["symbol"]].update(record)
        self.g["_crash_review_candidate"]=qualify

    def test_tick_rounding_and_closed_low(self):
        plan=self.g["crash_bounce_plan"](self.rows,"1",".001",self.now*1000)
        self.assertEqual(plan["entry"],".993")
        self.assertEqual(plan["tp"],"1.003")
        self.assertGreaterEqual(Decimal(plan["tp"])/Decimal(plan["entry"]),Decimal("1.01"))

    def test_qualified_v2_alerts_immediately_before_entry(self):
        self.tickers=[dict(symbol="AUSDT",priceChangePercent="-15",lastPrice="1")]
        state={"history":[],"symbols":{}}
        self.g["run_crash_bounce_alerts"](state,{"AUSDT":".001"},self.now)
        self.assertEqual(len(state["history"]),1)
        record=state["history"][0]
        self.assertEqual(record["tracking_model"],"ws_aggtrade_upcross_v3")
        self.assertEqual(record["tracking_status"],"WAIT_ENTRY")
        self.assertTrue(record["v2_sent"])
        self.assertEqual(record["v2_telegram_message_id"],31)
        self.assertNotIn("entered_at",record)
        self.receipt.assert_called_once()

    def test_unqualified_candidate_is_silent(self):
        self.tickers=[dict(symbol="AUSDT",priceChangePercent="-15",lastPrice="1")]
        def reject(state,record,now,allow_rest=False):
            record.update(v2_status="FILTERED_V2",tracking_status="V2_FILTERED")
        self.g["_crash_review_candidate"]=reject
        state={"history":[],"symbols":{}}
        self.g["run_crash_bounce_alerts"](state,{"AUSDT":".001"},self.now)
        self.assertEqual(len(state["history"]),1)
        self.assertFalse(state["history"][0]["v2_sent"])
        self.receipt.assert_not_called()

    def test_no_new_signal_without_two_percent_lower_entry(self):
        self.tickers=[dict(symbol="AUSDT",priceChangePercent="-15",lastPrice="1")]
        state={"history":[],"symbols":{}}
        self.g["run_crash_bounce_alerts"](state,{"AUSDT":".001"},self.now)
        self.g["run_crash_bounce_alerts"](state,{"AUSDT":".001"},self.now+180)
        self.assertEqual(len(state["history"]),1)
        self.assertEqual(self.receipt.call_count,1)

    def test_only_crash_automatic_telegram_source(self):
        response=self.g["requests"].post.return_value
        response.json.return_value={"ok":True}
        self.assertFalse(self.g["tg_send"]("fibonacci",source="FIBONACCI"))
        self.assertFalse(self.g["tg_send"]("old signal"))
        self.g["requests"].post.assert_not_called()
        self.assertTrue(self.g["tg_send"]("crash",source="CRASH_BOUNCE"))
        self.g["_TG_COMMAND_CONTEXT"].active=True
        self.assertTrue(self.g["tg_send"]("explicit command"))

    def test_top_20_ranking_applied_first(self):
        self.tickers=[dict(symbol=f"C{i:02}USDT",priceChangePercent=str(-40+i),
                           lastPrice="1") for i in range(25)]
        market={t["symbol"]:".001" for t in self.tickers}
        state={"history":[],"symbols":{}}
        self.g["run_crash_bounce_alerts"](state,market,self.now)
        self.assertEqual(len(state["history"]),20)
        self.assertEqual(len([r for r in state["history"] if r["v2_sent"]]),20)


if __name__=="__main__":
    unittest.main()
