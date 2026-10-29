import tempfile
import unittest
from unittest.mock import Mock
from crash_history import HistoryCollector, STEP


def rows(start, count):
    return [[t,'1','1.1','.9','1','10',t+STEP-1] for t in range(start,start+count*STEP,STEP)]

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fetch=Mock(side_effect=lambda path,p: rows(p['startTime'],(p['endTime']+1-p['startTime'])//STEP))
        self.emit=Mock()
        self.worker=HistoryCollector(self.tmp.name,self.fetch,self.emit,lambda:False,Mock())
        self.job=dict(symbol='TESTUSDT',signal_id='test',phase='signal',as_of=1000*STEP/1000+20)

    def test_five_days_closed_only_and_incremental(self):
        event=self.worker.collect(self.job)
        self.assertTrue(event['complete'])
        self.assertEqual(len(event['closed_15m_ohlcv']),480)
        self.assertLess(event['closed_15m_ohlcv'][-1][6],self.job['as_of']*1000)
        self.worker.collect(self.job)
        self.assertEqual(self.fetch.call_count,1)
        self.job['as_of']+=900
        self.worker.collect(self.job)
        self.assertEqual(self.fetch.call_args.args[1]['startTime'],1000*STEP)

    def test_delayed_snapshot_excludes_future_cache(self):
        self.worker.collect(dict(self.job,as_of=self.job['as_of']+3600))
        event=self.worker.collect(self.job)
        self.assertEqual(event['closed_15m_ohlcv'][-1][0],999*STEP)
        self.assertEqual(event['candle_count'],480)

    def test_partial_history_marked(self):
        self.fetch.side_effect=lambda path,p:rows(990*STEP,10)
        event=self.worker.collect(self.job)
        self.assertFalse(event['complete'])
        self.assertEqual(event['missing_candles'],470)

    def test_persistent_retry_and_nonblocking_submit(self):
        record=dict(symbol='TESTUSDT',signal_id='test')
        self.worker.submit(record,'entry',self.job['as_of'])
        self.fetch.assert_not_called()
        self.fetch.side_effect=RuntimeError('offline')
        self.worker.step(now=100)
        other=HistoryCollector(self.tmp.name,self.fetch,self.emit,lambda:False,Mock())
        self.assertEqual(other.jobs['test:entry']['retry_after'],1900)
        other.step(now=200)
        self.assertEqual(self.fetch.call_count,1)

    def test_block_and_hour_budget(self):
        self.worker.submit(dict(symbol='TESTUSDT',signal_id='test'),'signal',self.job['as_of'])
        self.worker.blocked=lambda:True
        self.worker.step(now=100)
        self.fetch.assert_not_called()
        self.worker.blocked=lambda:False
        self.worker.attempts=[100]*24
        self.worker.step(now=101)
        self.fetch.assert_not_called()

    def test_emit_and_remove_job(self):
        self.worker.submit(dict(symbol='TESTUSDT',signal_id='test'),'signal',self.job['as_of'])
        self.worker.step(now=100)
        self.assertEqual(self.emit.call_args.args[0]['phase'],'signal')
        self.assertEqual(self.worker.jobs,{})

class ScheduledHistoryTests(HistoryTests):
    def test_download_spacing_and_restart(self):
        self.worker.submit(dict(symbol='AAAUSDT',signal_id='a'),'signal',self.job['as_of'])
        self.worker.submit(dict(symbol='BBBUSDT',signal_id='b'),'signal',self.job['as_of'])
        self.worker.step(now=100)
        self.worker.step(now=115)
        self.assertEqual(self.fetch.call_count,1)
        other=HistoryCollector(self.tmp.name,self.fetch,self.emit,lambda:False,Mock())
        other.step(now=159)
        self.assertEqual(self.fetch.call_count,1)
        other.step(now=160)
        self.assertEqual(self.fetch.call_count,2)

    def test_gap_request_does_not_redownload_cached_tail(self):
        self.worker.collect(self.job)
        path=self.tmp.name+'/TESTUSDT.json'
        cached=self.worker._read(path,[])
        gap=cached[20][0]
        self.worker._write(path,cached[:20]+cached[21:])
        self.worker.collect(self.job)
        params=self.fetch.call_args.args[1]
        self.assertEqual(params['startTime'],gap)
        self.assertEqual(params['endTime'],gap+STEP-1)

    def test_partial_job_remains_queued(self):
        self.fetch.side_effect=lambda path,p:rows(990*STEP,10)
        self.worker.submit(dict(symbol='TESTUSDT',signal_id='test'),'signal',self.job['as_of'])
        self.worker.step(now=100)
        self.assertIn('test:signal',self.worker.jobs)

if __name__=='__main__':unittest.main()

class EntrySnapshotTests(unittest.TestCase):
    def test_tracking_captures_entry_once(self):
        import ast, math
        from pathlib import Path
        from types import SimpleNamespace
        source=ast.parse(Path(__file__).with_name('ema.py').read_text())
        names={'_crash_track_results','_crash_initialize_tracking','_crash_entry_features'}
        queue=Mock();features=Mock(return_value={'rsi14_wilder':31})
        g=dict(math=math,CRASH_BOUNCE_FILTER_VERSION='five_day_entry_v2_only',
               _crash_queue_history=queue,_crash_features=features,
               _CRASH_WS_CACHE=SimpleNamespace(candles=lambda s:[]),
               _crash_cached_candles_valid=lambda r,n:False,log=Mock(),
               _crash_append_data=Mock(),_crash_save_state=Mock())
        exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'ema.py','exec'),g)
        record=dict(symbol='TESTUSDT',created_at=100,entry='1',tp='1.01',reference_low='.99',
                    last_price='.995',previous_watch_price=.995,tracking_model='sampled_upward_cross_v2')
        state={'history':[record],'symbols':{}}
        g['_crash_track_results'](state,[dict(symbol='TESTUSDT',lastPrice='1',closeTime=200000)],200)
        self.assertEqual(record['entry_features_at'],200)
        self.assertEqual(record['entry_features_status'],'missing_or_stale')
        self.assertEqual(queue.call_args.args[1: ],('entry',200))
        g['_crash_track_results'](state,[dict(symbol='TESTUSDT',lastPrice='1.02',closeTime=300000)],300)
        self.assertEqual(record['tracking_status'],'TP')
        self.assertEqual(queue.call_count,1)

