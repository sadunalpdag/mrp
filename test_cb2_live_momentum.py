"""Regression tests for read-only CB2 live momentum observer."""
import json
import sqlite3
import unittest
import time
from datetime import datetime, timezone
from unittest.mock import patch

import cb2_live_momentum as m


BASE = ((int(time.time()*1000)//m.FIFTEEN_MS)-501)*m.FIFTEEN_MS


def candles(n=501, growth=.0018):
    rows=[]
    close=0.1
    for i in range(n):
        opening=close
        close=opening*(1+growth)
        timestamp=BASE+i*m.FIFTEEN_MS
        rows.append({"openTime":timestamp, "closeTime":timestamp+m.FIFTEEN_MS-1,
                     "open":opening,"high":close*1.0001,"low":opening*.9999,
                     "close":close,"volume":2000,"quoteVolume":200})
    return rows


def response(bars):
    return {"exchange":"binance-futures","symbol":"TESTUSDT",
            "timeframe":"15m","count":len(bars),"bars":bars}


def source(now, sqs=75):
    return {"setups":[{
        "symbol":"TESTUSDT","tf":"15m","bias":"bullish","sqs":sqs,
        "computedAt":now,"price":1.2,
        "archetype":{"key":"pattern_breakout","barsSinceTrigger":0},
    }]}


class Market:
    def __init__(self, rows, now):
        self.rows=rows
        self.now=now
        self.calls=[]
    def cb(self,path,cost,params=None):
        self.calls.append((path,cost,params))
        if "ohlcv/" in path:
            return response(self.rows)
        if "quotes/batch" in path:
            return {"asOf":self.now,"quotes":[{
                "m":"binance-futures:TESTUSDT","venueId":"binance-futures",
                "symbol":"TESTUSDT","price":self.rows[-1]["close"]}]}
        raise AssertionError("Unexpected API call: "+path)


class MomentumTest(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        m.initialize(self.db)
        self.rows=candles()
        self.now=int(time.time()*1000)
    def tearDown(self):
        self.db.close()
    def test_sheet_rows_clear_old_trade_cells(self):
        # A rejected FHE row must overwrite EVA's prior entry/TP/drawdown.
        old = ["2026-11-17", "EVAUSDT", "breakout_retest",
               "TP_1_PERCENT"] + [None] * 6 + [4.0344, 4.074744,
               "2026-11-17T08:59:59+00:00", -9.7759]
        rejected = ["2026-11-17", "FHEUSDT", "breakout_retest",
                    "REJECTED_FILTER", "2026-11-17T09:30:00+00:00",
                    81.4, 32.3379, 2.6277, 0.000000193314, False]
        full = m._rectangular_sheet_values([old, rejected], 20)
        self.assertTrue(all(len(row) == 20 for row in full))
        self.assertEqual(full[1][10:14], ["", "", "", ""])
        self.assertFalse(full[1][9])
        self.assertEqual(full[1][1], "FHEUSDT")

    def test_sheet_rejects_width_mismatch(self):
        with self.assertRaises(ValueError):
            m._rectangular_sheet_values([[1, 2, 3]], 2)

    def test_indicators_use_closed_history(self):
        parsed=m._closed_bars(response(self.rows),self.now)
        result,reason=m.indicators(parsed,self.now)
        self.assertEqual(reason,"OK")
        self.assertGreater(result["adx14"],20)
        self.assertGreater(result["change24_pct"],10)
        self.assertGreater(result["macd_histogram"],0)
    def test_open_bar_is_excluded(self):
        rows=self.rows+[dict(self.rows[-1],openTime=self.now,
                   closeTime=self.now+m.FIFTEEN_MS-1)]
        self.assertEqual(len(m._closed_bars(response(rows),self.now)),501)
    def test_signal_creates_paper_at_live_quote_not_next_open(self):
        market=Market(self.rows,self.now)
        with patch.object(m,"_notify",return_value=True):
            result=m.cycle(self.db,market,source(self.now),m._iso(self.now))
        signal=m.records(self.db)[0]
        self.assertEqual(result["passed"],1)
        self.assertEqual(signal["status"],"PAPER_OPEN")
        self.assertEqual(signal["entry_price"],self.rows[-1]["close"])
        self.assertEqual(signal["entered_at"],m._iso(self.now))
        self.assertEqual(signal["entry_bar_ms"],((self.now//m.FIFTEEN_MS)+1)*m.FIFTEEN_MS)
        self.assertAlmostEqual(signal["tp_price"],self.rows[-1]["close"]*1.01,places=10)
        self.assertTrue(signal["notified"])
        self.assertEqual(len([x for x in market.calls if "/public/setups" in x[0]]),0)
        with patch.object(m,"_notify",return_value=True):
            m.cycle(self.db,market,source(self.now),m._iso(self.now))
        self.assertEqual(len(m.records(self.db)),1)
    def test_live_tp_after_entry(self):
        market=Market(self.rows,self.now)
        with patch.object(m,"_notify",return_value=True):
            m.cycle(self.db,market,source(self.now),m._iso(self.now))
        first_full_open = ((self.now//m.FIFTEEN_MS)+1)*m.FIFTEEN_MS
        first_full=dict(candles(502)[-1])
        first_full["openTime"]=first_full_open
        first_full["closeTime"]=first_full_open+m.FIFTEEN_MS-1
        first_full["high"]=self.rows[-1]["close"]*1.015
        first_full["low"]=self.rows[-1]["close"]*.998
        market.rows=self.rows+[first_full]
        market.now=first_full["closeTime"]+1
        m.cycle(self.db,market,None,m._iso(market.now))
        self.assertEqual(m.records(self.db)[0]["status"],"TP_1_PERCENT")
    def test_snapshot_is_not_used_for_price_if_quote_fails(self):
        market=Market(self.rows,self.now)
        original=market.cb
        def failed(path,cost,params=None):
            if "quotes/batch" in path:
                return None
            return original(path,cost,params)
        market.cb=failed
        with patch.object(m,"_notify",return_value=True) as notify:
            m.cycle(self.db,market,source(self.now),m._iso(self.now))
            notify.assert_not_called()
        record=m.records(self.db)[0]
        self.assertTrue(record["filters_passed"])
        self.assertIsNone(record["entry_price"])
        self.assertEqual(record["status"],"PENDING_PRICE")
    def test_below_minimum_sqs_not_enrolled(self):
        market=Market(self.rows,self.now)
        m.cycle(self.db,market,source(self.now,sqs=65),m._iso(self.now))
        self.assertEqual(m.records(self.db),[])
        self.assertEqual(market.calls,[])
    def test_no_order_calls_or_exchange_endpoints(self):
        market=Market(self.rows,self.now)
        with patch.object(m,"_notify",return_value=True):
            m.cycle(self.db,market,source(self.now),m._iso(self.now))
        self.assertTrue(all("/public/" in x[0] for x in market.calls))
        self.assertTrue(all(x[0] != "/public/setups" for x in market.calls))
        self.assertEqual(m.export(self.db)["mode"],
                         "LIVE_SIGNAL_PAPER_ONLY_NO_BINANCE_ORDERS")


if __name__=="__main__":
    unittest.main()
