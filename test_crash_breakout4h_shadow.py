"""Offline guards for 4h CoinBeacon paper longs; no exchange or R2 calls."""
import datetime as dt
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import Mock, patch

import breakout4h_shadow as sh


def setup(now):
    return {"setups": [{
        "symbol": "TESTUSDT", "tf": "4h", "bias": "bullish",
        "computedAt": sh.last_four_hour_close(now), "sqs": 82.3,
        "archetype": {"direction": "long", "eventKind": "trendline_break",
                       "barsSinceTrigger": 0, "label": "Trendline break"},
        "participation": {"rvol": 2.2}
    }]}


class FourHourBreakShadowTests(unittest.TestCase):
    now = 1794050000.0

    def test_scan_only_recent_confirmed_long_breakout(self):
        with tempfile.TemporaryDirectory() as directory:
            data = sh.parse_setups(setup(self.now), sh.last_four_hour_close(self.now),
                                   {"TESTUSDT": 100}, self.now, directory,
                                   lambda _: [], {"TESTUSDT":"0.01"})
        self.assertEqual(len(data), 1)
        s = data[0]
        self.assertEqual((s["entry"], s["tp"], s["status"]), (100, 101, "OPEN"))
        self.assertEqual(s["coinbeacon_rvol"], 2.2)
        self.assertEqual(s["metric_status"], "PARTIAL")
        self.assertIsNone(s["rsi14_4h"])
        self.assertNotIn("api_key", str(s))

    def test_stale_trigger_bearish_spot_and_not_listed_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            original = setup(self.now)
            for mutate in (
                lambda x: x["setups"][0].update(bias="bearish"),
                lambda x: x["setups"][0]["archetype"].update(barsSinceTrigger=3),
                lambda x: x["setups"][0].update(tf="1h"),
                lambda x: x["setups"][0].update(computedAt=1),
                lambda x: x["setups"][0]["archetype"].update(direction="short"),
            ):
                doc = setup(self.now)
                mutate(doc)
                self.assertEqual(sh.parse_setups(
                    doc, sh.last_four_hour_close(self.now),
                    {"TESTUSDT": 100}, self.now, directory, lambda _: [],
                    {"TESTUSDT":"0.01"}), [])
            self.assertEqual(sh.parse_setups(
                original, sh.last_four_hour_close(self.now),
                {"TESTUSDT": 100}, self.now, directory, lambda _: [], {}), [])
            with self.assertRaisesRegex(ValueError, "COINBEACON_INVALID_RESPONSE"):
                sh.parse_setups({"junk": 1},0,{},self.now,directory,lambda _: [],{})

    def test_fully_closed_contiguous_four_hour_only(self):
        four = sh.FOUR_H_MS
        step = sh.FIFTEEN_M_MS
        end = int(self.now*1000)//four*four
        base = end - 23*four
        rows = []
        for j in range(23):
            for i in range(16):
                t = base + j*four + i*step
                px = 100 + .1*j + .001*i
                rows.append([t, px, px+.2, px-.2, px,
                             100+j, t+step-1])
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            folder = Path(d)/"crash_research"/"history_cache"
            folder.mkdir(parents=True)
            (folder/"TESTUSDT.json").write_text(json.dumps(rows),encoding="utf-8")
            aggregated = sh._four_hour_bars(d, "TESTUSDT", [], self.now)
            self.assertEqual(len(aggregated),23)
            values = sh.indicators(aggregated)
            self.assertEqual(values["bars_4h"],23)
            self.assertTrue(0 <= values["rsi14_4h"] <= 100)
            self.assertGreater(values["atr14_pct_4h"],0)
            self.assertGreater(values["volume_ratio_4h"],0)
            broken = [r for r in rows if r[0] != base+step]
            (folder/"TESTUSDT.json").write_text(json.dumps(broken),encoding="utf-8")
            self.assertEqual(len(sh._four_hour_bars(d,"TESTUSDT",[],self.now)),22)

    def test_observed_tp_and_72h_price_path(self):
        t = self.now
        sig = sh.parse_setups(setup(t), sh.last_four_hour_close(t),
                              {"TESTUSDT":100},t,"/tmp",lambda _:[],
                              {"TESTUSDT":"0.01"})[0]
        state={"signals":[sig]}
        self.assertFalse(sh.update_prices(state, {}, t+10))
        self.assertTrue(sh.update_prices(state, {"TESTUSDT":96},t+60))
        self.assertEqual(sig["status"],"OPEN")
        self.assertEqual(sig["max_adverse_pct"],-4)
        sh.update_prices(state, {"TESTUSDT":101.1},t+120)
        self.assertEqual(sig["status"],"TP")
        self.assertEqual(sig["max_favorable_pct"],1.1)
        sh.update_prices(state, {"TESTUSDT":97},t+24*3600+20)
        self.assertEqual(sig["returns"]["24"],-3)
        sh.update_prices(state, {"TESTUSDT":111},t+48*3600+60)
        self.assertEqual(sig["returns"]["48"],11)
        sh.update_prices(state, {"TESTUSDT":99},t+72*3600+90)
        self.assertIsNotNone(sig["finished_at"])
        self.assertEqual(sig["status"],"TP")
        self.assertFalse(sh.update_prices(state,{"TESTUSDT":400},t+73*3600))
        self.assertEqual(sig["highest"],111)
        payload=sh.export_payload(state,t+72*3600+90)
        self.assertEqual(payload["summary"]["tp"],1)
        self.assertEqual(payload["tp_pct"],1)

    def test_empty_monitor_has_zero_signals_and_no_real_broker(self):
        out=sh.export_payload({"signals":[]},self.now)
        self.assertEqual(out["summary"]["n"],0)
        self.assertEqual(out["strategy"],"PAPER_ONLY_4H_BULLISH_COINBEACON_BREAK")
        self.assertNotIn("broker",out)

    def test_key_hbreak_is_sent_only_as_header(self):
        fake = Mock()
        fake.json.return_value={"setups":[]}
        with tempfile.TemporaryDirectory() as d:
            monitor=sh.Breakout4HShadow(d,lambda _:None,http_get=Mock(return_value=fake))
            with patch.dict(os.environ,{"hbreak":"a-private-key","COINBEACON_API_KEY":"old"},clear=False):
                self.assertEqual(monitor._fetch(),{"setups":[]})
        args=monitor.http_get.call_args
        self.assertEqual(args.kwargs["headers"],{"X-API-Key":"a-private-key"})
        self.assertNotIn("a-private-key",str(args.kwargs["params"]))
        self.assertEqual(args.kwargs["params"]["timeframe"],"4h")
        self.assertEqual(args.kwargs["params"]["exchange"],"binance-futures")

    def test_every_new_coin_breakout_creates_one_paper_long_even_without_ws_ticker(self):
        from pathlib import Path
        now = self.now
        board = setup(now)
        board["setups"][0]["price"] = 22.0
        duplicate = dict(board["setups"][0])
        duplicate["sqs"] = 60
        board["setups"].append(duplicate)
        other = dict(board["setups"][0])
        other["symbol"] = "OTHERUSDT"
        other["price"] = 40.0
        board["setups"].append(other)
        class Reply:
            def __init__(self, value):
                self.value = value
            def raise_for_status(self):
                pass
            def json(self):
                return self.value
        candles = []
        close = sh.last_four_hour_close(now)
        for j in range(32):
            t = close + 1 - (32-j) * sh.FOUR_H_MS
            price = 20 + j * .25
            candles.append({"openTime":t, "closeTime":t+sh.FOUR_H_MS-1,
                            "open": price, "high": price + .5,
                            "low": price - .5, "close": price + .1,
                            "volume": 100+j})
        # A still-forming candle must never enter RSI or ATR.
        candles.append({"openTime":close+1, "closeTime":close+sh.FOUR_H_MS,
                        "open": 100, "high": 200, "low": 1, "close": 199,
                        "volume": 100000})
        calls=[]
        def get(url, **kwargs):
            calls.append(url)
            if url.endswith("/public/setups"):
                return Reply(board)
            sym = url.rsplit("/", 2)[-2]
            return Reply({"symbol":sym, "timeframe":"4h", "bars":candles})
        with tempfile.TemporaryDirectory() as d:
            monitor=sh.Breakout4HShadow(
                d, lambda _:None, http_get=get, sheet_writer=lambda *_:True,
                r2_upload=lambda *_:{"json":"x","zip":"y"})
            with patch.dict(os.environ, {"hbreak":"secret","COINBEACON_API_KEY":""}):
                monitor.lock.acquire()
                monitor._work({}, {"TESTUSDT":".1", "OTHERUSDT":".1"}, lambda _:[], now)
                self.assertEqual(len(monitor.state["signals"]),2)
                for paper in monitor.state["signals"]:
                    self.assertEqual(paper["status"],"OPEN")
                    self.assertEqual(paper["metric_status"],"COMPLETE")
                    self.assertEqual(paper["metric_source"],"coinbeacon_closed_4h_ohlcv")
                    self.assertGreater(paper["atr14_pct_4h"],0)
                    self.assertGreater(paper["volume_ratio_4h"],0)
                    self.assertEqual(paper["entry_source"],"COINBEACON_BINANCE_FUTURES_PRICE")
                self.assertEqual({p["entry"] for p in monitor.state["signals"]},{22.0,40.0})
                monitor.lock.acquire()
                monitor._work({}, {"TESTUSDT":".1", "OTHERUSDT":".1"}, lambda _:[], now+60)
                self.assertEqual(len(monitor.state["signals"]),2)
                self.assertEqual(len(calls),3)  # 1 board + 1 OHLCV for each coin

    def test_stale_setup_board_is_retried_not_silently_marked_processed(self):
        now = self.now
        doc = setup(now)
        doc["setups"][0]["computedAt"] -= sh.FOUR_H_MS
        class Reply:
            def raise_for_status(self): pass
            def json(self): return doc
        with tempfile.TemporaryDirectory() as d:
            monitor=sh.Breakout4HShadow(
                d, lambda _:None, http_get=lambda *_a,**_k:Reply(),
                sheet_writer=lambda *_: True,
                r2_upload=lambda *_:{"json":"ok","zip":"ok"})
            with patch.dict(os.environ, {"hbreak":"secret"}):
                monitor.lock.acquire()
                monitor._work({}, {"TESTUSDT":".1"},lambda _:[],now)
            self.assertEqual(monitor.state["last_scan_slot"],0)
            self.assertEqual(len(monitor.state["signals"]),0)

    def test_separate_r2_json_and_zip_match_exactly(self):
        stored={}
        class FakeS3:
            def upload_file(self,path,bucket,key,ExtraArgs=None):
                stored[key]=(open(path,"rb").read(),ExtraArgs["ContentType"])
            def head_object(self,Bucket,Key):
                return {"ContentLength":len(stored[Key][0])}
        module=types.SimpleNamespace(client=lambda *args,**kwargs:FakeS3())
        env={"CRASH_REPORT_R2_ACCOUNT_ID":"a"*32,
             "CRASH_REPORT_R2_BUCKET":"test", "CRASH_REPORT_R2_ACCESS_KEY_ID":"x",
             "CRASH_REPORT_R2_SECRET_ACCESS_KEY":"y"}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,{"boto3":module}):
            keys=sh.upload_pair(d,{"signals":[]},self.now,env)
            self.assertEqual(set(keys),{"json","zip"})
            self.assertIn("/4h-break/",keys["json"])
            self.assertEqual(json.loads(stored[keys["json"]][0]),{"signals":[]})
            with zipfile.ZipFile(io.BytesIO(stored[keys["zip"]][0])) as z:
                content=z.read(z.namelist()[0])
                self.assertEqual(content,stored[keys["json"]][0])
            self.assertEqual(list((__import__("pathlib").Path(d)/"4h_break_exports").iterdir()),[])

    def test_existing_old_tabs_removed_only_by_exact_name(self):
        old1=types.SimpleNamespace(title="Structure Agent Test",id=10)
        old2=types.SimpleNamespace(title="Break Bounce Test",id=11)
        other=types.SimpleNamespace(title="Sinyal",id=12)
        created=types.SimpleNamespace(title=sh.TAB,id=13,col_count=21)
        book=Mock(title="EMA Filter")
        book.worksheets.return_value=[old1,old2,other]
        book.add_worksheet.return_value=created
        client=Mock()
        client.open_by_key.return_value=book
        with patch("crash_sheets_dashboard._credentials",return_value=(client,"id")), \
             patch("crash_sheets_dashboard._name",return_value="emafilter"):
            self.assertTrue(sh.publish_sheet({"signals":[]},self.now))
        self.assertEqual([c.args[0].title for c in book.del_worksheet.call_args_list],
                         ["Structure Agent Test","Break Bounce Test"])
        body=book.values_batch_update.call_args.args[0]
        self.assertEqual(len(body["data"]),1)
        self.assertEqual(body["data"][0]["range"],"'4H Break Long'!A1:U82")


if __name__ == "__main__":
    unittest.main()
