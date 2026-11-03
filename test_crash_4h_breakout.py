"""Offline 4H research tests: no Binance/network calls."""
import tempfile
import unittest
from unittest.mock import Mock

from crash_4h_breakout import BreakoutResearch, candidates, export_csv, confirm_15m

PERIOD=4*3600


class Cache:
    def snapshot(self, now):
        return [{"symbol":f"COIN{i}USDT","lastPrice":str(100+i),
                 "priceChangePercent":"-5","quoteVolume":"15000000"}
                for i in range(40)]
    def candles(self, symbol):
        return []


def bars(now):
    # 65 completed 4H bars including the last completed candle.
    last=int(now//PERIOD)*PERIOD
    return [[(last-(65-i)*PERIOD)*1000,"90","100","80","90","100",
             (last-(64-i)*PERIOD)*1000-1,"15000000",0,0,0,0]
            for i in range(65)]


class Tests(unittest.TestCase):
    def test_volume_and_bucket_filter(self):
        tickers=Cache().snapshot(100)
        selected=candidates(tickers,{x["symbol"] for x in tickers})
        self.assertEqual(len(selected),30)
        self.assertEqual(selected[0][0],"COIN0USDT")
        tickers[0]["quoteVolume"]="1"
        self.assertNotIn("COIN0USDT",[r[0] for r in candidates(tickers,{x["symbol"] for x in tickers})])

    def test_120_second_pacing_and_reuse_until_new_4h_close(self):
        now=[200*PERIOD+300]
        fetch=Mock(side_effect=lambda route,params: bars(now[0]))
        sent=Mock(return_value={"ok":True})
        with tempfile.TemporaryDirectory() as tmp:
            b=BreakoutResearch(tmp,Cache(),fetch,sent,
                market_symbols=lambda:{f"COIN{i}USDT" for i in range(40)},
                env={"CRASH_4H_BREAKOUT_ENABLED":"1"},clock=lambda:now[0])
            b.step()
            self.assertEqual(fetch.call_count,1)
            self.assertEqual(fetch.call_args.args[1]["limit"],65)
            now[0]+=60;b.step()
            self.assertEqual(fetch.call_count,1)
            now[0]+=60;b.step()
            self.assertEqual(fetch.call_count,2)
            self.assertLessEqual(len(b.state["cache"]),30)

    def test_ws_closed_4h_replaces_followup_rest(self):
        now=[200*PERIOD+600]
        fetch=Mock(side_effect=lambda route,params: bars(now[0]))
        class WithWS(Cache):
            def __init__(self):
                self.streamed=[]
                self.subscriptions=[]
            def set_4h_research_symbols(self, symbols):
                self.subscriptions.append(list(symbols))
            def closed_4h_candles(self, symbol):
                return self.streamed if symbol=="COIN0USDT" else []
        ws=WithWS()
        with tempfile.TemporaryDirectory() as tmp:
            b=BreakoutResearch(tmp,ws,fetch,Mock(),
                market_symbols=lambda:{f"COIN{i}USDT" for i in range(40)},
                env={"CRASH_4H_BREAKOUT_ENABLED":"1"},clock=lambda:now[0])
            b.step()
            self.assertEqual(fetch.call_count,1)
            self.assertEqual(b.state["cache"]["COIN0USDT"]["source"],"REST")
            original=b.state["cache"]["COIN0USDT"]["close_ms"]
            now[0]+=PERIOD-510   # New 4H close + 90s, inside WS grace.
            ws.streamed=[bars(now[0])[-1]]
            b.step()
            self.assertEqual(fetch.call_count,1)
            self.assertGreater(b.state["cache"]["COIN0USDT"]["close_ms"],original)
            self.assertEqual(b.state["cache"]["COIN0USDT"]["source"],"WS")
            self.assertEqual(len(b.state["cache"]["COIN0USDT"]["rows"]),65)

    def test_gap_requires_paced_rest_repair(self):
        now=[200*PERIOD+600]
        fetch=Mock(side_effect=lambda route,params: bars(now[0]))
        class WithGap(Cache):
            def __init__(self):
                self.streamed=[]
            def set_4h_research_symbols(self, symbols):
                pass
            def closed_4h_candles(self, symbol):
                return self.streamed if symbol=="COIN0USDT" else []
        ws=WithGap()
        with tempfile.TemporaryDirectory() as tmp:
            b=BreakoutResearch(tmp,ws,fetch,Mock(),
                market_symbols=lambda:{f"COIN{i}USDT" for i in range(40)},
                env={"CRASH_4H_BREAKOUT_ENABLED":"1"},clock=lambda:now[0])
            b.step()
            original=b.state["cache"]["COIN0USDT"]["close_ms"]
            # Two full 4H intervals later: missing one middle candle.
            now[0]+=2*PERIOD
            ws.streamed=[bars(now[0])[-1]]
            # A different symbol might be chosen first, so other 29 are fresh.
            latest=int(now[0]//PERIOD)*PERIOD*1000-1
            for i in range(1,30):
                b.state["cache"][f"COIN{i}USDT"]={"close_ms":latest,"rows":bars(now[0])}
            b.step()
            self.assertEqual(b.state["cache"]["COIN0USDT"]["source"],"REST")
            self.assertEqual(fetch.call_count,2)
            self.assertGreater(b.state["cache"]["COIN0USDT"]["close_ms"],original)

    def test_no_15m_rest_when_ws_missing(self):
        self.assertEqual(confirm_15m(Cache(),"COIN0USDT",100000),"NO_WS_CANDLES")

    def test_final_24h_observation_and_zero_peak(self):
        now=[200*PERIOD+3600]
        with tempfile.TemporaryDirectory() as tmp:
            b=BreakoutResearch(tmp,Cache(),Mock(),Mock(),
                market_symbols=lambda:{f"COIN{i}USDT" for i in range(40)},
                env={"CRASH_4H_BREAKOUT_ENABLED":"1"},clock=lambda:now[0])
            cutoff=now[0]-24*3600-60
            b.state["events"]=[{
                "event_id":"old","symbol":"COIN0USDT","status":"BREAKOUT",
                "close_at":cutoff,"close_ms":1,"close":200.0,
                "partial_max":{"1":0,"4":0.1,"12":0.2,"24":0.25},
                "max_observed_gain_pct":0.0,"last_observed_at":now[0]-120,
                "observations":1
            }]
            latest=int(now[0]//PERIOD)*PERIOD*1000-1
            b.state["cache"]={f"COIN{i}USDT":{"close_ms":latest}
                              for i in range(30)}
            b.step()
            item=b.state["events"][0]
            self.assertEqual(item["completed_horizons"]["24"],0.25)
            self.assertEqual(item["max_observed_gain_pct"],0.0)

    def test_export_missing_horizons_not_falsely_zero(self):
        s=export_csv([{"event_id":"s","symbol":"AAAUSDT","status":"BREAKOUT",
            "close_at":1,"close":2,"observations":3,"max_observed_gain_pct":1.2,
            "completed_horizons":{"1":0.8}}])
        self.assertIn("0.8",s)
        self.assertIn("gain_24h",s.splitlines()[0])


if __name__=="__main__":
    unittest.main()
