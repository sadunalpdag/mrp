"""Offline safety tests for the read-only daily-profit watcher."""
import tempfile
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from crash_daily_profit_paper_live import DailyProfitPaperMonitor

PARIS = ZoneInfo('Europe/Paris')
def clock(day, hour):
    return datetime(2026, 11, day, hour, tzinfo=PARIS).timestamp()

class FakeBroker:
    def __init__(self):
        self.equity = '1000'
        self.positions = []
        self.income = []
        self.calls = []
    def request(self, method, path, params=None):
        self.calls.append((method, path))
        if method != 'GET':
            raise AssertionError('live order attempted')
        if path == '/fapi/v3/account':
            return {'multiAssetsMargin': False, 'totalWalletBalance': self.equity,
                    'totalUnrealizedProfit': '0', 'totalMarginBalance': self.equity,
                    'assets': [{'asset':'USDT','marginBalance':self.equity}]}
        if path == '/fapi/v1/income':
            return [r for r in self.income if params['startTime'] <= r['time'] <= params['endTime']]
        if path == '/fapi/v3/positionRisk':
            return self.positions
        raise AssertionError('not allowlisted')

class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = clock(12,12)
        self.b = FakeBroker()
        self.alerts = []
        self.m = DailyProfitPaperMonitor(
            self.tmp.name, self.b, lambda msg: None,
            lambda msg: self.alerts.append(msg) or {'ok':True}, clock=lambda:self.t)
    def test_peak_not_decreased_and_new_day_frozen_goal(self):
        self.m.step()
        self.t=clock(12,16);self.b.equity='1040';self.m.step()
        self.t=clock(13,12);self.b.equity='950';self.m.step()
        self.assertEqual(self.m.state['peak'], '1040')
        self.assertEqual(self.m.state['target'], '1092.00')
    def test_real_account_snapshot_only_no_order(self):
        self.m.step()
        self.t=clock(12,14);self.b.equity='1050'
        self.b.positions=[{'symbol':'BTCUSDT','positionAmt':'0.01',
                           'positionSide':'BOTH','markPrice':'100000'}]
        self.m.step()
        self.assertEqual(self.m.state['trigger']['positions'][0]['observed_price'], '100000')
        self.assertEqual(len(self.alerts),1)
        self.b.positions[0]['markPrice']='99000'; self.t=clock(12,15);self.m.step()
        self.assertEqual(self.m.state['trigger']['indicative_touches'],0)
        self.assertTrue(all(method=='GET' for method,_ in self.b.calls))
    def test_transfer_is_not_paper_gain(self):
        self.m.step()
        self.t=clock(12,14);self.b.equity='1060'
        self.b.income=[{'time':int(clock(12,13)*1000),'tranId':1,
                        'incomeType':'TRANSFER','income':'60','asset':'USDT'}]
        self.m.step()
        self.assertIsNone(self.m.state['trigger'])
        self.assertEqual(self.m.state['last']['adjusted_equity'],'1000')
    def test_paper_settings_persist_without_live_trading(self):
        self.m.step()
        self.m.control('paper_target:3')
        self.m.control('paper_off')
        self.assertFalse(self.m.state['enabled'])
        self.assertEqual(self.m.state['target'],'1050.00')
        restored=DailyProfitPaperMonitor(self.tmp.name,self.b,lambda msg:None,clock=lambda:self.t)
        self.assertFalse(restored.state['enabled'])
        self.t=clock(13,12)
        restored.step()
        self.assertEqual(restored.state['target'],'1030.00')
        self.assertTrue(all(method=='GET' for method,_ in self.b.calls))

if __name__ == '__main__':
    unittest.main()
