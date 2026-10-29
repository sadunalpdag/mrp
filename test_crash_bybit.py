import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from crash_bybit import BybitReference, COUNT, STEP
from crash_history import HistoryCollector


def sample(end, count=COUNT):
    start = end - COUNT * STEP
    # Bybit returns newest candles first.
    return [[str(t), '1', '1.2', '.8', '1.1', '10', '11']
            for t in reversed(range(start, start + count * STEP, STEP))]


class BybitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.end = 20000 * 86400000
        self.clock = lambda: self.end / 1000 + 60
        self.log = Mock()
        self.get = Mock()

    def payload(self, candles, symbol='BTCUSDT'):
        r = Mock(status_code=200)
        r.json.return_value = {
            'retCode':0,
            'result':{'symbol':symbol,'category':'linear','list':candles},
        }
        return r

    def test_480_closed_candles_are_saved_only_as_bybit(self):
        self.get.return_value = self.payload(sample(self.end))
        src = BybitReference(self.temp.name, self.log, get=self.get, clock=self.clock)
        result = src.inspect('BTCUSDT', self.end)
        self.assertEqual(result['source'], 'BYBIT_LINEAR')
        self.assertTrue(result['reference_only'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['candle_count'], 480)
        self.assertIn(result['v2_reference_status'], ('STRONG_V2', 'FILTERED_V2'))
        kwargs = self.get.call_args.kwargs
        self.assertEqual(kwargs['params']['limit'], 480)
        self.assertEqual(kwargs['params']['interval'], '15')
        self.assertEqual(kwargs['params']['end'], self.end - 1)
        self.assertEqual(kwargs['params']['category'], 'linear')
        self.assertFalse(kwargs['allow_redirects'])
        saved = json.loads(Path(self.temp.name, 'BTCUSDT.json').read_text())
        self.assertEqual(saved['source'], 'BYBIT_LINEAR')
        self.assertEqual(len(saved['closed_15m_ohlcv']), 480)
        self.assertTrue(all(row[-1] == 'BYBIT_LINEAR' for row in saved['closed_15m_ohlcv']))
        self.assertEqual(saved['closed_15m_ohlcv'][-1][6], self.end - 1)
        self.assertIsNone(src.inspect('BTCUSDT', self.end + STEP))
        # The exact complete window is reused from cache.
        self.assertEqual(src.inspect('BTCUSDT', self.end), result)
        self.get.assert_called_once()

    def test_missing_data_is_never_marked_complete(self):
        self.get.return_value = self.payload(sample(self.end, 479))
        src = BybitReference(self.temp.name, self.log, get=self.get, clock=self.clock)
        result = src.inspect('BTCUSDT', self.end)
        self.assertEqual(result['candle_count'], 479)
        self.assertFalse(result['complete'])
        self.assertEqual(result['v2_reference_status'], 'MISSING_5D_HISTORY')
        self.assertIsNone(src.inspect('BTCUSDT', self.end))
        self.get.assert_called_once()

    def test_current_unclosed_candle_is_excluded(self):
        candles = sample(self.end, 479)
        candles.insert(0, [str(self.end),'1','1.2','.8','1.1','10','11'])
        self.get.return_value = self.payload(candles)
        src = BybitReference(self.temp.name, self.log, get=self.get, clock=self.clock)
        result = src.inspect('BTCUSDT', self.end)
        self.assertEqual(result['candle_count'], 479)
        self.assertFalse(result['complete'])

    def test_http_errors_and_invalid_symbol_do_not_modify_cache(self):
        r = Mock(status_code=429)
        self.get.return_value = r
        src = BybitReference(self.temp.name, self.log, get=self.get, clock=self.clock)
        self.assertIsNone(src.inspect('BTCUSDT', self.end))
        self.assertFalse(Path(self.temp.name, 'BTCUSDT.json').exists())
        self.assertIsNone(src.inspect('../BAD', self.end))
        self.get.assert_called_once()

    def test_collector_does_not_mix_bybit_into_binance_snapshot(self):
        fake = Mock()
        fake.inspect.return_value = {'source':'BYBIT_LINEAR','reference_only':True,
                                     'candle_count':480,'complete':True}
        fetch = Mock()
        resolver = Mock()
        resolver.resolve.return_value = []
        c = HistoryCollector(self.temp.name + '/binance', fetch, Mock(), lambda:True,
                             self.log, resolver=resolver, bybit=fake)
        event = c.collect({'symbol':'BTCUSDT','signal_id':'s1',
                           'phase':'signal','as_of':self.end/1000})
        self.assertEqual(event['candle_count'], 0)
        self.assertFalse(event['complete'])
        self.assertEqual(event['bybit_reference']['source'], 'BYBIT_LINEAR')
        fake.inspect.assert_called_once_with('BTCUSDT', self.end)
        fetch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
