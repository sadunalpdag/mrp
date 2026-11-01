"""Offline checks for independent AI market research; never call Binance/OpenAI."""
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo
from ai_top_gainers import TopGainers, features, SCHEMA, normalize_recommendation, format_top_gainers_report

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
        self.assertIn("decision",item["required"])
        self.assertIn("entry_reason",item["required"])
        self.assertEqual(item["properties"]["decision"]["enum"],["BUY","NEUTRAL","AVOID"])


    def test_normalization_does_not_force_unsafe_entry(self):
        base={"symbol":"COINUSDT","decision":"BUY","entry_price":99,
              "entry_reason":"Support test","score":78,"bias":"CONTINUE",
              "reason":"momentum","risk":"volatility"}
        self.assertEqual(normalize_recommendation(base,100)["entry_price"],99)
        missing=normalize_recommendation({**base,"entry_price":None},100)
        self.assertEqual(missing["decision"],"NEUTRAL")
        self.assertIn("Geçerli LONG",missing["entry_reason"])
        invalid=normalize_recommendation({**base,"entry_price":80},100)
        self.assertEqual(invalid["decision"],"NEUTRAL")
        avoid=normalize_recommendation({**base,"decision":"AVOID"},100)
        self.assertIsNone(avoid["entry_price"])
        self.assertEqual(avoid["decision"],"AVOID")
        neutral=normalize_recommendation({**base,"decision":"NEUTRAL"},100)
        self.assertIsNone(neutral["entry_price"])

    def test_ten_buy_decisions_fit_telegram_character_limit(self):
        ranked=[(f"COIN{i}USDT",12.34) for i in range(10)]
        assessments=[{"symbol":symbol,"score":80,"bias":"CONTINUE",
                      "reason":"momentum" * 45,"risk":"volatility" * 45,
                      "decision":"BUY","entry_price":100,
                      "entry_reason":"test support " * 60}
                     for symbol,ret in ranked]
        report={"status":"COMPLETE","analyses":assessments,
                "agents":{"news":{"status":"SOURCED_DATE_REPORTED"}}}
        message=format_top_gainers_report(report,ranked)
        self.assertLessEqual(len(message.encode("utf-16-le"))//2,4096)
        self.assertIn("Kararlar: AL 10 | NÖTR 0 | SAT/KAÇIN 0",message)
        self.assertEqual(message.count("LONG sanal giriş:"),10)

    def test_ai_report_shows_buy_neutral_avoid_without_false_paper_trades(self):
        now=time.time()
        sent=[]
        class FakeAgents:
            def __init__(self,*args,**kwargs):
                pass
            def evaluate(self,entries,*,market_prices,**kwargs):
                analyses=[]
                for i,e in enumerate(entries):
                    symbol=e["symbol"]
                    price=market_prices[symbol]
                    decision="BUY" if i in (0,1) else ("AVOID" if i==2 else "NEUTRAL")
                    entry=(price*.98 if i==0 else
                           None if i==1 else
                           price*.96 if i==2 else None)
                    analyses.append({"symbol":symbol,"score":80,
                                     "bias":"CONTINUE" if i<2 else "UNCERTAIN",
                                     "reason":"RSI EMA destek sinyali",
                                     "risk":"ATR oynaklık riski",
                                     "decision":decision,"entry_price":entry,
                                     "entry_reason":"Teknik seviyeye geri çekilme bekleniyor"})
                return {"status":"COMPLETE","analyses":analyses,
                        "agents":{"news":{"status":"NOT_SEARCHED"}},
                        "derivative_metrics":[],"usage":[],
                        "estimated_model_usd":0.0,"web_tool_calls":0,
                        "estimated_search_usd":0.0,"estimated_total_usd":0.0}
        with tempfile.TemporaryDirectory() as d:
            bot=TopGainers(d,Cache(),lambda *a:rows(),
                           lambda msg:sent.append(msg) or {"ok":True},
                           env={"TOP_GAINERS_ENABLED":"1","OPENAI_API_KEY":"offline"},
                           clock=lambda:now)
            slot=int(now//(4*3600))
            old={sym:p*.99 for sym,p in bot.collect(now).items()}
            bot.state["snapshots"].append({"slot":slot-1,"at":now-4*3600,
                                           "prices":old})
            with patch("ai_market_agents.AgentsResearch",FakeAgents):
                self.assertTrue(bot.step())
            report=bot.state["reports"][0]
            self.assertEqual(report["status"],"COMPLETE")
            self.assertEqual([a["decision"] for a in report["analyses"][:3]],
                             ["BUY","NEUTRAL","AVOID"])
            self.assertEqual(len(bot.state["trades"]),1)
            self.assertEqual(bot.state["trades"][0]["symbol"],
                             report["analyses"][0]["symbol"])
            self.assertIn("Kararlar: AL 1 | NÖTR 8 | SAT/KAÇIN 1",sent[0])
            self.assertIn("Geçerli LONG giriş fiyatı yok",sent[0])
            self.assertIn("SAT / KAÇIN",sent[0])
            self.assertLessEqual(len(sent[0].encode("utf-16-le"))//2,4096)

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
