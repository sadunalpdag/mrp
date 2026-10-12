import tempfile
import unittest
from unittest.mock import Mock
from fibonacci_signals import select,setup,FibonacciEngine

class FibonacciTests(unittest.TestCase):
    def test_dynamic_ranking_boundaries(self):
        ticks=[dict(symbol=str(i),priceChangePercent='7',quoteVolume=str(i+1)) for i in range(12)]
        market={str(i):'.01' for i in range(12)}
        self.assertEqual([t['symbol'] for t in select(ticks,market)],[str(i) for i in range(11,1,-1)])
        ticks[0]['quoteVolume']='1000';ticks[11]['priceChangePercent']='10.01'
        selected=select(ticks,market)
        self.assertEqual(selected[0]['symbol'],'0')
        self.assertNotIn('11',[t['symbol'] for t in selected])
        self.assertEqual(select([dict(symbol='0',priceChangePercent='5',quoteVolume='1')],market)[0]['symbol'],'0')
        self.assertEqual(select([dict(symbol='0',priceChangePercent='10',quoteVolume='1')],market)[0]['symbol'],'0')

    def fixture(self):
        now=300*3600
        hourly=[]
        for i in range(260):
            t=(40+i)*3600000;c=80+i*.1
            hourly.append([t,c,c+.1,c-.1,c,1,t+3599999])
        rows=[]
        for i in range(64):
            t=int(now*1000)-(64-i)*900000;c=101+i*.03
            rows.append([t,c,c+.2,c-.2,c,1,t+899999])
        # Confirmed A=100, B=110, then retracement/confirmation near 104.7.
        for i,lo,hi,cl in [(55,100,102,101),(56,101,104,103),(57,103,107,106),
                            (58,105,110,109),(59,106,109,107),(60,105,108,106),
                            (61,104.4,106,105),(62,103.9,104.5,104.1),(63,104,104.9,104.7)]:
            rows[i][1:5]=[lo+.1,hi,lo,cl]
        return rows,hourly,now

    def test_confirmed_setup_and_risk(self):
        rows,hourly,now=self.fixture()
        p=setup(rows,hourly,104.7,'.01',now)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p['zone_low'],103.82)
        self.assertEqual(p['zone_high'],105)
        self.assertGreaterEqual(p['rr'],2)
        self.assertIsNone(setup(rows,hourly,109.8,'.01',now))
        rows[-1][4]=104.2
        self.assertIsNone(setup(rows,hourly,104.2,'.01',now))

    def test_stale_and_broken_low_rejected(self):
        rows,hourly,now=self.fixture()
        self.assertIsNone(setup(rows,hourly,104.7,'.01',now+1800))
        rows[-2][3]=99
        self.assertIsNone(setup(rows,hourly,104.7,'.01',now))

    def test_tracking_outside_universe_and_restart_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            now=1000
            tick=dict(symbol='TESTUSDT',priceChangePercent='-2',quoteVolume='100',lastPrice='111')
            fetch=Mock();send=Mock(return_value=True)
            engine=FibonacciEngine(directory,fetch,lambda n:[tick],lambda:{'TESTUSDT':'.01'},send,Mock(),lambda:False,Mock())
            engine.history.start=Mock()
            engine.state['records']=[dict(symbol='TESTUSDT',signal_id='FIB-test',created_at=900,status='OPEN',entry=105,sl=103,tp=110,sent=True,history_queued=True)]
            engine.scan(now)
            self.assertEqual(engine.state['records'][0]['status'],'TP')
            fetch.assert_not_called()
            other=FibonacciEngine(directory,fetch,lambda n:[],lambda:{},send,Mock(),lambda:False,Mock())
            self.assertEqual(other.state['records'][0]['signal_id'],'FIB-test')

    def test_signal_delivery_history_and_duplicate(self):
        rows,hourly,now=self.fixture()
        tick=dict(symbol='TESTUSDT',priceChangePercent='7',quoteVolume='1000',lastPrice='104.7')
        with tempfile.TemporaryDirectory() as directory:
            fetch=Mock(side_effect=lambda path,p: hourly if p['interval']=='1h' else rows)
            send=Mock(return_value=True)
            engine=FibonacciEngine(directory,fetch,lambda n:[tick],lambda:{'TESTUSDT':'.01'},send,Mock(),lambda:False,Mock())
            engine.history.start=Mock()
            engine.scan(now)
            self.assertEqual(send.call_count,1)
            self.assertIn('FIBONACCI',send.call_args.args[0])
            self.assertEqual(len(engine.history.jobs),1)
            engine.scan(now+180)
            self.assertEqual(send.call_count,1)
            self.assertEqual(len(engine.state['records']),1)

if __name__=='__main__':unittest.main()
