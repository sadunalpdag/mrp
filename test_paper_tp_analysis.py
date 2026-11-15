"""Offline tests for three-day TP-only analysis; no HTTP or real orders."""
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import paper_tp_analysis as subject


def bars(start_ms, count):
    out=[]
    for i in range(count):
        open_ms=start_ms+i*900000
        price=100+0.01*i
        out.append([str(open_ms),str(price),str(price+1),str(price-1),
                    str(price+0.5),"100",str(open_ms+899999),str(2000+i),
                    "20","10","10","0"])
    return out


class TPAnalysisTests(unittest.TestCase):
    def test_only_pre_signal_closed_candles_in_indicators(self):
        b=bars(1760000000000,482)
        # Last two candles would introduce future information if accidentally used.
        b[-2][4]="1000000"
        b[-1][4]="1000000"
        event=int(b[480][0])
        calculated=subject.indicators(b,event)
        self.assertEqual(calculated["status"],"READY")
        self.assertEqual(calculated["coverage"],480)
        self.assertLess(calculated["ema20"],110)
        self.assertIsNotNone(calculated["adx14"])
        self.assertIsNotNone(calculated["rvol20"])

    def test_stale_prior_day_bar_is_not_reported_as_signal_time(self):
        b=bars(1760000000000,480)
        event=int(b[-1][6])+12*3600000
        output=subject.indicators(b,event)
        self.assertEqual(output["status"],"LATEST_SIGNAL_CANDLE_NOT_PUBLISHED")

    def test_archive_gap_never_returns_ready(self):
        b=bars(1760000000000,480)
        b[200][0]=str(int(b[200][0])+120000)
        output=subject.indicators(b,int(b[-1][6])+1000)
        self.assertEqual(output["status"],"ARCHIVE_GAP")

    def test_tp_only_filters_open_watch_and_old_dates(self):
        date=datetime(2026,11,15,14,0,tzinfo=timezone.utc)
        same=date.isoformat()
        tp={"id":"a","symbol":"FOOUSDT","strategy":"trend",
            "source":"coinbeacon-api2","observed_at":same,
            "entry_scenarios":{"next_15m_open":{
                "status":"TP_1_PERCENT","entry_price":2,
                "tp_at_candle_close":same}}}
        open_trade=dict(tp,id="b",entry_scenarios={"next_15m_open":{
            "status":"OPEN_IN_OBSERVED_WINDOW","entry_price":2}})
        watch=dict(tp,id="c",family="trend_watch",strategy="trend_watch")
        old=dict(tp,id="d",observed_at="2026-11-01T00:00:00+00:00")
        output=subject.candidates("cb2",[tp,open_trade,watch,old],date)
        self.assertEqual([x["id"] for x in output],["a","b","c"])
        self.assertEqual(output[-1]["outcome_status"],"WATCH_ONLY")
        full=subject.candidates("cb2",[tp,open_trade,watch,old],date,days=0)
        self.assertEqual(len(full),4)

    def test_missing_archive_does_not_create_false_indicators(self):
        with tempfile.TemporaryDirectory() as directory:
            today=datetime(2026,11,15,14,0,tzinfo=timezone.utc)
            signal={"id":"a","symbol":"FOOUSDT","strategy":"trend",
                "source":"coinbeacon-api2",
                "observed_at":"2026-11-14T12:00:00+00:00",
                "entry_scenarios":{"next_15m_open":{
                    "status":"TP_1_PERCENT","entry_price":2}}}
            with patch.object(subject,"ROOT",Path(directory)), \
                 patch.object(subject,"klines",return_value=[]), \
                 patch.object(subject,"_upload",return_value=None):
                result=subject.run("cb2",[signal],None,None,now=today)
            self.assertEqual(result["totals"]["tp_records"],1)
            self.assertEqual(result["totals"]["comparison_records"],0)
            self.assertEqual(result["totals"]["indicator_complete"],0)
            self.assertEqual(result["records"][0]["indicators"]["status"],
                             "PENDING_480_CLOSED_BARS")
            path=Path(directory)/"paper_tp_analysis"/"cb2_3day_latest.json"
            self.assertTrue(path.exists())

    def test_all_25_signals_enriched_in_one_pass_with_single_archive_window(self):
        from unittest.mock import Mock
        import json
        with tempfile.TemporaryDirectory() as directory:
            current=datetime(2026,11,15,14,0,tzinfo=timezone.utc)
            date=datetime(2026,11,13,18,0,tzinfo=timezone.utc)
            base=int(datetime(2026,11,7,0,0,tzinfo=timezone.utc).timestamp()*1000)
            records=[]
            for i in range(25):
                when=(date+timedelta(minutes=i)).isoformat()
                kind="TP_1_PERCENT" if i<13 else "OPEN_IN_OBSERVED_WINDOW"
                records.append({
                    "id":"signal-"+str(i),"source":"coinbeacon-api2",
                    "symbol":"FOOUSDT","strategy":"rvol","observed_at":when,
                    "status":"COMPLETE_48H",
                    "entry_scenarios":{"next_15m_open":{
                        "status":kind,"entry_price":2.0}}
                })
            with patch.object(subject,"ROOT",Path(directory)), \\
                 patch.object(subject,"klines",return_value=bars(base,680)) as loader, \\
                 patch.object(subject,"_upload",return_value=None), \\
                 patch.dict(subject.os.environ,{"GOOGLE_SERVICE_ACCOUNT_JSON":""}):
                report=subject.run("cb2",records,None,None,now=current)
            self.assertEqual(report["totals"]["signal_records"],25)
            self.assertEqual(report["totals"]["tp_records"],13)
            self.assertEqual(report["totals"]["comparison_records"],12)
            self.assertEqual(report["totals"]["indicator_complete"],25)
            self.assertEqual(loader.call_count,1)
            cumulative=json.loads((Path(directory)/"paper_tp_analysis"/
                                  "cb2_all_signals_latest.json").read_text())
            self.assertEqual(cumulative["outcome_counts"]["TP_1_PERCENT"],13)
            self.assertEqual(cumulative["outcome_counts"]["OPEN_IN_OBSERVED_WINDOW"],12)
            self.assertEqual(len(cumulative["records"]),25)

    def test_missing_price_and_watch_are_included_not_counted_as_tp(self):
        now=datetime(2026,11,15,15,0,tzinfo=timezone.utc)
        record={"id":"watch","source":"coinbeacon-api2","symbol":"BARUSDT",
                "strategy":"volume_watch","family":"volume_watch",
                "status":"WATCH","reference_price":None,
                "observed_at":"2026-11-14T13:30:00+00:00",
                "entry_scenarios":{}}
        selected=subject.candidates("cb2",[record],now,days=0)
        self.assertEqual(len(selected),1)
        self.assertIsNone(selected[0]["entry_price"])
        self.assertEqual(selected[0]["outcome_status"],"WATCH_ONLY")
        report=subject._report("cb2",[{"signal":selected[0],
                             "indicators":{"status":"PENDING_480_CLOSED_BARS"}}],now,None)
        self.assertEqual(report["totals"]["tp_records"],0)
        self.assertEqual(report["outcome_counts"]["WATCH_ONLY"],1)

    def test_cumulative_json_retains_old_days_after_three_day_rollover(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            date=datetime(2026,11,15,14,0,tzinfo=timezone.utc)
            record={"id":"a","source":"coinbeacon-api2","symbol":"FOOUSDT",
                    "strategy":"retest","observed_at":"2026-11-13T13:00:00+00:00",
                    "entry_scenarios":{"next_15m_open":{
                        "status":"TP_1_PERCENT","entry_price":2}}}
            with patch.object(subject,"ROOT",Path(directory)), \\
                 patch.object(subject,"klines",return_value=[]), \\
                 patch.object(subject,"_upload",return_value=None), \\
                 patch.dict(subject.os.environ,{"GOOGLE_SERVICE_ACCOUNT_JSON":""}):
                before=subject.run("cb2",[record],None,None,now=date)
                after=subject.run("cb2",[],None,None,now=date+timedelta(days=5))
            self.assertEqual(before["totals"]["signal_records"],1)
            self.assertEqual(after["totals"]["signal_records"],0)
            cumulative=json.loads((Path(directory)/"paper_tp_analysis"/
                                  "cb2_all_signals_latest.json").read_text())
            self.assertEqual(cumulative["totals"]["signal_records"],1)



if __name__=="__main__":
    unittest.main()
