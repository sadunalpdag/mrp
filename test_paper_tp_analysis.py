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
        self.assertEqual([x["id"] for x in output],["a","b"])

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
            self.assertEqual(result["totals"]["tp_records"],1)\n            self.assertEqual(result["totals"]["comparison_records"],0)
            self.assertEqual(result["totals"]["indicator_complete"],0)
            self.assertEqual(result["records"][0]["indicators"]["status"],
                             "PENDING_480_CLOSED_BARS")
            path=Path(directory)/"paper_tp_analysis"/"cb2_3day_latest.json"
            self.assertTrue(path.exists())

if __name__=="__main__":
    unittest.main()
