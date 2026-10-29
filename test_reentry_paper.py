"""Offline closed-candle/as-of/SL-free paper strategy tests; no network."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from reentry_paper import ReentryPaper, setup, advance, closed_rows, FIVE, FOUR, MAX_AGE


def bar(t,o,h,l,c,interval=FIVE):
    return [t,o,h,l,c,10,t+interval-1]


def fixture():
    four=[bar(i*FOUR,99+i,102+i,98+i,100+i,FOUR) for i in range(30)]
    five=[bar(30*FOUR+i*FIVE,128,128.2,127.8,128) for i in range(10)]
    five[-3]=bar(five[-3][0],127.5,127.6,125.9,126)
    five[-2]=bar(five[-2][0],126,126.2,125.8,126)
    five[-1]=bar(five[-1][0],126.1,128.1,126,128)
    return five,four


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.rows5,self.rows4=fixture()
        self.now=(self.rows5[-1][6]+1)/1000+1
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.fetch=Mock(side_effect=lambda path,p:copy.deepcopy(self.rows5 if p['interval']=='5m' else self.rows4))
        self.deliver=Mock(return_value={'ok':True})
        self.universe=Mock(return_value=[{'symbol':'AUSDT','quoteVolume':'100'}])
        self.e=self.make()

    def make(self):
        return ReentryPaper(self.tmp.name,self.fetch,self.universe,self.deliver,Mock(),clock=lambda:self.now)

    def test_long_and_short_closed_candle_setup(self):
        sig=setup(self.rows5,self.rows4)
        self.assertEqual(sig['direction'],'LONG');self.assertNotIn('sl',sig)
        def mirror(rows):
            return [[r[0],260-r[1],260-r[3],260-r[2],260-r[4],r[5],r[6]] for r in rows]
        self.assertEqual(setup(mirror(self.rows5),mirror(self.rows4))['direction'],'SHORT')

    def test_future_4h_candle_does_not_change_frozen_zone(self):
        before=setup(self.rows5,self.rows4)
        future=self.rows4+[bar(30*FOUR,10,20,1,5,FOUR)]
        self.assertEqual(setup(self.rows5,future),before)

    def test_reentry_more_than_three_candles_late_rejected(self):
        self.rows5.insert(-1,bar(self.rows5[-1][0],126,126.2,125.8,126))
        self.rows5[-1][0]+=FIVE;self.rows5[-1][6]+=FIVE
        # Existing break now three before confirmation remains valid.
        self.assertIsNotNone(setup(self.rows5,self.rows4))
        self.rows5.insert(-1,bar(self.rows5[-1][0],126,126.2,125.8,126))
        self.rows5[-1][0]+=FIVE;self.rows5[-1][6]+=FIVE
        self.assertIsNone(setup(self.rows5,self.rows4))

    def test_first_confirmation_only(self):
        self.rows5[-2]=bar(self.rows5[-2][0],126,128.1,125.9,128)
        self.assertIsNone(setup(self.rows5,self.rows4))

    def test_unclosed_and_gapped_data(self):
        current=bar(self.rows5[-1][0]+FIVE,128,129,127,128)
        self.assertEqual(len(closed_rows(self.rows5+[current],FIVE,self.now)),10)
        with self.assertRaises(ValueError):closed_rows(self.rows5[:4]+self.rows5[5:],FIVE,self.now)

    def test_signal_is_persisted_no_duplicate_after_restart(self):
        self.e.step();self.assertEqual(len(self.e.state['trades']),1)
        self.assertEqual(len(self.e.state['outbox']),1)
        self.e=self.make();self.e.step()
        self.assertEqual(len(self.e.state['trades']),1)
        self.assertEqual(self.deliver.call_count,1)
        self.assertIn('SANAL LONG',self.deliver.call_args.args[0])
        self.assertIn('SL yok',self.e.summary())

    def test_no_sl_even_after_large_adverse_move_and_no_entry_bar_tp(self):
        self.e.step();trade=next(iter(self.e.state['trades'].values()))
        # Confirmation candle cannot count as TP.
        advance(trade,self.rows5,self.now);self.assertEqual(trade['status'],'OPEN')
        after=bar(trade['last_bar']+FIVE,128,128,64,80)
        advance(trade,[after],self.now+300)
        self.assertEqual(trade['status'],'OPEN');self.assertEqual(trade['max_adverse_pct'],50)
        hit=bar(after[0]+FIVE,80,130,79,129.5)
        advance(trade,[hit],self.now+600);self.assertEqual(trade['status'],'TP')

    def test_gap_is_not_a_tp_win(self):
        self.e.step();t=next(iter(self.e.state['trades'].values()))
        advance(t,[bar(t['last_bar']+2*FIVE,128,140,127,130)],self.now+600)
        self.assertEqual(t['status'],'DATA_GAP')

    def test_timeout_precedes_later_tp(self):
        self.e.step();t=next(iter(self.e.state['trades'].values()))
        target=int((t['entered_at']+MAX_AGE)*1000)
        t['last_bar']=target-FIVE
        advance(t,[bar(target,128,140,127,130)],self.now+MAX_AGE)
        self.assertEqual(t['status'],'TIMEOUT')

    def test_cooldown_prevents_all_rest_and_reports_wait(self):
        self.e.cooldown=lambda:self.now+1000
        self.e.step();self.fetch.assert_not_called()
        self.assertIn('API beklemesi',self.e.summary())

    def test_4h_cached_and_5m_not_refetched_within_interval(self):
        self.rows5[-1]=bar(self.rows5[-1][0],128,128.1,127.9,128)
        self.e.step();self.e.step();self.assertEqual(self.fetch.call_count,2)
        self.now+=300;self.rows5.append(bar(self.rows5[-1][0]+FIVE,128,128.1,127.9,128))
        self.e.step()
        self.assertEqual(sum(c.args[1]['interval']=='4h' for c in self.fetch.call_args_list),1)

    def test_heartbeat_contains_paper_summary(self):
        from crash_heartbeat import Heartbeat
        h=Heartbeat(self.tmp.name,self.deliver,lambda:{'reentry_paper':self.e.summary()},Mock(),clock=lambda:self.now)
        self.assertIn('RE-ENTRY 4H/5m — SANAL',h.message(self.now))

    def test_module_cannot_send_real_exchange_orders(self):
        source=Path(__file__).with_name('reentry_paper.py').read_text()
        self.assertNotIn('BinanceBroker',source)
        self.assertNotIn('/fapi/v1/order',source)
        self.assertNotIn('execute_real_trade',source)

if __name__=='__main__':unittest.main()
