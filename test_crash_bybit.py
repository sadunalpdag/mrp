"""Offline tests for gap-only Bybit fallback and Binance source priority."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from crash_bybit import BybitReference, STEP
from crash_data_sources import DataSources, source_counts, tagged
from crash_history import HistoryCollector
from crash_v2 import assess_entry

COUNT = 480


def candles(start, count):
    return [[t, '1', '1.2', '.8', '1.1', '10', t+STEP-1, '11', 0, 0, 0, 0]
            for t in range(start, start + count*STEP, STEP)]


def payload(rows, symbol='BTCUSDT'):
    response=Mock(status_code=200)
    response.json.return_value={
        'retCode':0,
        'result':{'symbol':symbol, 'category':'linear', 'list':[
            [str(r[0]),str(r[1]),str(r[2]),str(r[3]),str(r[4]),
             str(r[5]),str(r[7])]
            for r in reversed(rows)]}}
    return response


class BybitGapTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.end=20000*86400000
        self.start=self.end-COUNT*STEP
        self.now=self.end/1000+60
        self.get=Mock()
        self.log=Mock()
        self.b=BybitReference(self.tmp.name, self.log,
                              get=self.get, clock=lambda:self.now)
        self.archive=[tagged(r,'BINANCE_ARCHIVE') for r in candles(self.start,309)]
        self.missing=[self.start+t*STEP for t in range(309,COUNT)]

    def test_only_171_missing_candles_requested_and_source_tagged(self):
        answer=candles(self.start+308*STEP,172)
        self.get.return_value=payload(answer)
        result=self.b.fill_missing('BTCUSDT',self.start,self.end,
                                   self.missing,self.archive)
        self.assertEqual(len(result),171)
        self.assertTrue(all(r[12]=='BYBIT_LINEAR' for r in result))
        self.assertEqual(result[0][0],self.missing[0])
        self.assertEqual(result[-1][6],self.end-1)
        params=self.get.call_args.kwargs['params']
        self.assertEqual(params['limit'],172)  # Only gap + 1 Binance anchor.
        self.assertEqual(params['start'],self.start+308*STEP)
        self.assertEqual(params['end'],self.end-1)
        self.assertEqual(params['interval'],'15')
        self.assertEqual(params['category'],'linear')
        self.assertFalse(self.get.call_args.kwargs['allow_redirects'])
        # Successful fetch is paced, not immediately repeated.
        self.assertEqual(self.b.fill_missing('BTCUSDT',self.start,self.end,
                           self.missing,self.archive),[])
        self.get.assert_called_once()

    def test_no_binance_data_never_fetches_five_days_bybit(self):
        self.assertEqual(self.b.fill_missing('BTCUSDT',self.start,self.end,
                         list(range(self.start,self.end,STEP)),[]),[])
        self.get.assert_not_called()

    def test_wrong_bybit_market_price_fails_closed(self):
        rows=candles(self.start+308*STEP,172)
        rows[0][4]='1.5'  # Same-time Binance anchor close = 1.1
        self.get.return_value=payload(rows)
        self.assertEqual(self.b.fill_missing('BTCUSDT',self.start,self.end,
                         self.missing,self.archive),[])
        self.assertTrue(any('CROSS_EXCHANGE_PRICE_MISMATCH' in str(x)
                            for x in self.log.call_args_list))

    def test_unfinished_or_wrong_timestamp_is_ignored(self):
        rows=candles(self.start+308*STEP,172)
        rows[-1][0]=self.end  # Already open, must not be used.
        self.get.return_value=payload(rows)
        result=self.b.fill_missing('BTCUSDT',self.start,self.end,
                                   self.missing,self.archive)
        self.assertEqual(len(result),170)
        self.assertTrue(all(r[6] < self.end for r in result))

    def test_transport_failure_and_unsupported_symbol(self):
        response=Mock(status_code=429)
        self.get.return_value=response
        self.assertEqual(self.b.fill_missing('BTCUSDT',self.start,self.end,
                        self.missing,self.archive),[])
        self.assertEqual(self.b.fill_missing('../BTC',self.start,self.end,
                        self.missing,self.archive),[])
        self.get.assert_called_once()

    def test_mixed_480_candles_qualify_for_v2_and_binance_replaces_bybit(self):
        self.get.return_value=payload(candles(self.start+308*STEP,172))
        is_blocked=True
        rest=Mock(return_value=candles(self.start+309*STEP,171))
        resolver=DataSources(self.tmp.name+'/archive',rest,lambda:is_blocked,
            self.log,clock=lambda:self.now,bybit=self.b)
        resolver.archive=Mock(return_value=[])
        mixed=resolver.resolve('BTCUSDT',self.start,self.end,self.archive)
        self.assertEqual(len(mixed),480)
        self.assertEqual(source_counts(mixed)['BYBIT_LINEAR'],171)
        self.assertEqual(source_counts(mixed)['BINANCE_ARCHIVE'],309)
        self.assertEqual(assess_entry(mixed,[],self.end/1000)['status'],'STRONG_V2')
        rest.assert_not_called()
        # Once Binance REST recovers, it must replace exact timestamps.
        is_blocked=False
        preferred=resolver.resolve('BTCUSDT',self.start,self.end,mixed)
        self.assertEqual(len(preferred),480)
        self.assertEqual(source_counts(preferred).get('BYBIT_LINEAR',0),0)
        self.assertEqual(source_counts(preferred)['BINANCE_REST'],171)
        rest.assert_called_once()
        self.get.assert_called_once()  # Bybit is no longer needed.

    def test_collector_writes_mixed_series_and_retries_for_binance(self):
        self.get.return_value=payload(candles(self.start+308*STEP,172))
        fetch=Mock()
        resolver=DataSources(self.tmp.name+'/archive',fetch,lambda:True,
            self.log,clock=lambda:self.now,bybit=self.b)
        resolver.archive=Mock(return_value=[])
        on_rows=Mock()
        collector=HistoryCollector(self.tmp.name+'/history',fetch,Mock(),
            lambda:True,self.log,resolver=resolver,
            ws_rows=lambda sym:self.archive,on_rows=on_rows)
        collector.submit(dict(symbol='BTCUSDT',signal_id='warmup:BTCUSDT'),
                         'warmup',self.end/1000)
        collector.step(now=self.now)
        self.assertTrue(collector.jobs)  # Keep upgrading to Binance later.
        rows=on_rows.call_args.args[1]
        self.assertEqual(len(rows),480)
        self.assertEqual(source_counts(rows).get('BYBIT_LINEAR'),171)
        fetch.assert_not_called()
        self.assertEqual(collector.jobs['warmup:BTCUSDT:warmup']['retry_after'],
                         self.now+1800)


if __name__=='__main__':
    unittest.main()
