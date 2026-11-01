"""Offline checks for independent AI market research; never call Binance/OpenAI."""
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo
from ai_top_gainers import TopGainers, features, SCHEMA

class Cache:
    def __init__(self, price=101):
        self.price=price
    def snapshot(self, now):
        # Exact normalized field names returned by ema.py CrashBounceWebSocketCache.
        return [{"symbol":f"COIN{i}USDT","lastPrice":str(self.price+i),
                 "priceChangePercent":"1","closeTime":int(now*1000)}
                for i in range(60)] + [{"symbol":"BTCUSDT","lastPrice":str(self.price)}]

def rows():
    end=int(time.time()//3600)*3600
    return [[(end-i*3600)*1000-3600000,0,103,98,100+i*.01,0,
             (end-i*3600)*1000-1,1000,0,0,0,0] for i in reversed(range(60))]

class Tests(unittest.TestCase):
    def test_schema_has_entry_and_no_trading_tools(self):
        item=SCHEMA["properties"]["analyses"]["items"]
        self.assertIn("entry_price",item["required"])
        self.assertEqual(item["properties"]["entry_price"]["type"],["number","null"])

    def test_collect_real_ws_snapshot_and_legacy_fallback(self):
        now=time.time()
        with tempfile.TemporaryDirectory() as d:
            bot=TopGainers(d,Cache(),lambda *a:rows(),lambda *a:{"ok":True},
                           env={"TOP_GAINERS_ENABLED":"1"},clock=lambda:now)
            prices=bot.collect(now)
            self.assertEqual(len(prices),61)
            self.assertEqual(prices["BTCUSDT"],101)
            class Legacy:
                def snapshot(self, moment):
                    return [{"s":"ETHUSDT","c":"2500.5"},
                            {"s":"BTCUSDT","c":"0"},
                            {"symbol":"NOTUSDT","lastPrice":"nan"}]
            bot.cache=Legacy()
            self.assertEqual(bot.collect(now),{"ETHUSDT":2500.5})

    def test_real_ws_snapshot_allows_top_gainers_report(self):
        now=time.time()
        delivered=[]
        with tempfile.TemporaryDirectory() as d:
            bot=TopGainers(d,Cache(),lambda *a:rows(),
                           lambda message:delivered.append(message) or {"ok":True},
                           env={"TOP_GAINERS_ENABLED":"1"},clock=lambda:now)
            slot=int(now//(4*3600))
            old={s:p*0.99 for s,p in bot.collect(now).items()}
            bot.state["snapshots"].append({"at":now-4*3600,
                                            "slot":slot-1,"prices":old})
            self.assertTrue(bot.step())
            self.assertEqual(bot.state["reports"][0]["status"],"NO_AI_KEY")
            self.assertEqual(len(bot.state["reports"][0]["ranking"]),10)
            self.assertIn("AI TOP GAINERS",delivered[0])

    def test_btc_real_ws_snapshot_is_available(self):
        now=time.time()
        with tempfile.TemporaryDirectory() as d:
            bot=TopGainers(d,Cache(),lambda *a:rows(),lambda *a:{"ok":True},
                           env={"TOP_GAINERS_ENABLED":"1"},clock=lambda:now)
            bot.btc_step()
            self.assertEqual(len(bot.state["btc_slots"]),1)
            self.assertEqual(bot.state["btc"][0]["status"],"NO_AI_KEY")
            self.assertEqual(bot.state["btc"][0]["reference_price"],101)

    def test_btc_input_has_real_changes(self):
        f=features("BTCUSDT",rows(),0,None)
        self.assertIn("closed_4h_return_pct",f)
        self.assertIn("return_1h_pct",f)

    def test_entry_requires_later_snapshot_then_tp(self):
        now=[time.time()]
        with tempfile.TemporaryDirectory() as d:
            cache=Cache(price=99)
            bot=TopGainers(d,cache,lambda *x:rows(),lambda x:{"ok":True},
                           env={"TOP_GAINERS_ENABLED":"1"},clock=lambda:now[0])
            bot.state["trades"]=[{"symbol":"BTCUSDT","created_at":now[0]-60,
                "entry":100.0,"tp":101.0,"status":"WAIT_ENTRY",
                "entry_at":None,"tp_at":None,"observations":0,
                "first_observed_at":None,"last_observed_at":None}]
            bot.follow_step()
            self.assertEqual(bot.state["trades"][0]["status"],"OPEN")
            self.assertIsNone(bot.state["trades"][0]["tp_at"])
            now[0]+=60
            cache.price=102
            bot.follow_step()
            self.assertEqual(bot.state["trades"][0]["status"],"TP_1PCT")

    def test_btc_missing_ws_does_not_consume_hour(self):
        now=[time.time()]
        with tempfile.TemporaryDirectory() as d:
            class Unready:
                def snapshot(self, moment):
                    return []
            bot=TopGainers(d,Unready(),lambda *args:rows(),
                           lambda message:{"ok":True},
                           env={"TOP_GAINERS_ENABLED":"1"},clock=lambda:now[0])
            bot.btc_step()
            self.assertEqual(bot.state["btc_slots"],[])
            self.assertEqual(bot.state["btc"],[])

    def test_daily_does_not_claim_unresolved_success(self):
        now=datetime(2026,10,31,21,10,tzinfo=ZoneInfo("Europe/Paris")).timestamp()
        posts=[]
        with tempfile.TemporaryDirectory() as d:
            bot=TopGainers(d,Cache(),Mock(),lambda s:posts.append(s) or {"ok":True},
                           env={"TOP_GAINERS_ENABLED":"1"},clock=lambda:now)
            bot.daily_step();bot.daily_step()
            self.assertEqual(len(posts),1)
            self.assertIn("Sonuçlananlarda TP: henüz yok",posts[0])
            self.assertEqual(len(bot.state["daily"]),1)

if __name__=="__main__":
    unittest.main()
