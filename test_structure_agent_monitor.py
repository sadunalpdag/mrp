import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import structure_agent_monitor as mon


def row(ts, op=100, high=101, low=99, close=100, volume=10):
    return [ts, op, high, low, close, volume, ts+mon.STEP-1]


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.end = 2000*mon.FOUR_H
        self.now = self.end/1000+30
        self.rows = [row(t) for t in range(self.end-480*mon.STEP, self.end, mon.STEP)]

    def test_universe_quotes_age_and_exact_top50(self):
        tickers = [dict(symbol=f'C{i}USDT', quoteVolume=100-i,
                        lastPrice=1, priceChangePercent=-2) for i in range(51)]
        market = {t['symbol']:'.01' for t in tickers}
        onboard = {s:1 for s in market}
        onboard.pop('C0USDT')
        top = mon.universe(tickers, market, onboard, self.now)
        self.assertEqual(len(top), 50)
        self.assertFalse(top[0]['age_ready'])
        self.assertEqual(top[-1]['symbol'], 'C49USDT')
        self.assertNotIn('C50USDT', [r['symbol'] for r in top])

    def test_missing_latest_and_interior_never_generate_setup(self):
        for removed in (0, 250, 479):
            rows = self.rows[:removed]+self.rows[removed+1:]
            result = mon.analyse(rows, self.now, -5)
            self.assertEqual(result['decision'], 'VERİ BEKLENİYOR')
            self.assertNotIn('entry', result)

    def test_four_hour_blocks_never_bridge_gaps_or_partial_blocks(self):
        self.assertEqual(len(mon.aggregate(self.rows)), 30)
        self.assertEqual(len(mon.aggregate(self.rows[1:])), 29)
        rows = self.rows[:200]+self.rows[201:]
        self.assertEqual(len(mon.aggregate(rows)), 29)

    def test_pivot_requires_two_closed_right_bars(self):
        bars = [row(i*mon.FOUR_H, high=h) for i,h in enumerate([101,102,110,103,102])]
        self.assertEqual(mon.pivots(bars[:4], 2, True), [])
        self.assertEqual(mon.pivots(bars, 2, True), [(2,110)])

    def test_cache_prefers_binance_and_rejects_unclosed_or_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'crash_research/history_cache/AAAUSDT.json'
            path.parent.mkdir(parents=True)
            tagged = list(self.rows[-1])+[None]*5+['BINANCE_ARCHIVE']
            path.write_text(json.dumps([tagged]))
            alternate = list(tagged)
            alternate[12] = 'OKX_SWAP'
            alternate[4] = 100.5
            invalid = row(self.end-mon.STEP, high=90)
            result = mon.cached_rows(directory, 'AAAUSDT', [alternate, invalid, row(self.end)], self.now)
            self.assertEqual(result, [tagged])

    def test_first_retest_reversal_and_dedup_are_prospective(self):
        # Downward confirmed swings followed by a fresh 4h body break,
        # then a single completed 15m first contact/reclaim.
        start = self.end-30*mon.FOUR_H
        rows = []
        for i in range(30):
            price = 200-i*2
            high = price+(10 if i%5 == 2 else 1)
            low = price-(10 if i%5 == 4 else 1)
            close = price
            for j in range(16):
                if i == 29 and j == 15:
                    rows.append(row(start+i*mon.FOUR_H+j*mon.STEP, price, 170, 141, 170))
                else:
                    rows.append(row(start+i*mon.FOUR_H+j*mon.STEP, price, high, low, close))
        rows.append(row(self.end, 166.2, 168, 165.9, 167))
        rows = rows[-480:]
        now = (self.end+mon.STEP)/1000+30
        analysis = mon.analyse(rows, now, -5)
        self.assertEqual(analysis['method'], 'DÖNÜŞ')
        self.assertEqual(analysis['decision'], 'TEYİT VAR')
        ticker = dict(symbol='AAAUSDT', quoteVolume=1e8, lastPrice=167,
                      priceChangePercent=-5)
        with tempfile.TemporaryDirectory() as directory:
            a = mon.scan(directory, [ticker], {'AAAUSDT':'.01'}, {'AAAUSDT':1}, lambda s:rows, now, {})
            b = mon.scan(directory, [ticker], {'AAAUSDT':'.01'}, {'AAAUSDT':1}, lambda s:rows, now, a)
            self.assertEqual(len(a['events']), 1)
            self.assertEqual(len(b['events']), 1)
            self.assertFalse(b['events'][0]['execution_eligible'])
            # A newly started monitor one candle later must not backdate entry.
            later = rows[1:]+[row(self.end+mon.STEP, 167, 169, 166, 168)]
            c = mon.scan(directory, [ticker], {'AAAUSDT':'.01'}, {'AAAUSDT':1}, lambda s:later,
                         now+900, {})
            self.assertEqual(c['events'], [])

    def test_no_upstream_state_mutation_and_sheet_ranges(self):
        ticker = dict(symbol='AAAUSDT', quoteVolume=1e8, lastPrice=100,
                      priceChangePercent=2)
        original = copy.deepcopy(ticker)
        with tempfile.TemporaryDirectory() as directory:
            obj = mon.scan(directory, [ticker], {'AAAUSDT':'.01'}, {'AAAUSDT':1},
                           lambda s:self.rows, self.now, {})
            values = mon.sheet_values(obj)
            self.assertEqual(len(values), 162)
            self.assertEqual(len(values[5]), 18)
            self.assertEqual(len(values[6]), 18)
            self.assertEqual(ticker, original)
            self.assertEqual(obj['mode'], 'MONITOR_ONLY')

    def test_publish_only_own_tab_and_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = mon.scan(directory, [], {}, {}, lambda s:[], self.now, {})
        book = Mock(title='EMA Filter')
        other = Mock(title='Sinyal')
        ws = Mock(title=mon.TAB, id=88)
        ws.get.return_value = [[mon.TAB, 'YALNIZ İZLEME — GERÇEK EMİR YOK']]
        book.worksheets.return_value = [other]
        book.add_worksheet.return_value = ws
        client = Mock()
        client.open_by_key.return_value = book
        with patch('crash_sheets_dashboard._credentials', return_value=(client,'book')):
            self.assertTrue(mon.publish(obj))
        self.assertEqual(book.values_batch_update.call_args.args[0]['data'][0]['range'],
                         "'Structure Agent Test'!A1:R162")
        other.update.assert_not_called()
        book.del_worksheet.assert_not_called()

    def test_monitor_has_no_execution_or_live_state_dependencies(self):
        tree = ast.parse(Path(mon.__file__).read_text())
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any('night' in n or 'broker' in n or 'live_v2' in n for n in imports))


if __name__ == '__main__':
    unittest.main()
