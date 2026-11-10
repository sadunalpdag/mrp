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

    def test_only_active_formations_are_izle(self):
        # Bare 4h trend / market structure is not a working setup.
        self.assertEqual(mon.analyse(self.rows, self.now, 2)["decision"], "YAPI YOK")
        with patch.object(mon, "structure", return_value=("YÜKSELİŞ", [], [])):
            result = mon.analyse(self.rows, self.now, 2)
            self.assertEqual(result["decision"], "YAPI YOK")
            self.assertEqual(result["method"], "—")

            forming = copy.deepcopy(self.rows)
            impulse_ts, pull_ts, last_ts = [r[0] for r in forming[-3:]]
            forming[-3] = row(impulse_ts, 100, 106, 99, 105, 100)
            forming[-2] = row(pull_ts, 104, 105, 102, 103, 10)
            forming[-1] = row(last_ts, 103, 104.5, 102.5, 104, 12)
            pending = mon.analyse(forming, self.now, 2)
            self.assertEqual(pending["method"], "DEVAM")
            self.assertEqual(pending["decision"], "İZLE")
            self.assertEqual(pending["level"], 105)

            forming[-1] = row(last_ts, 104, 107, 103, 106, 12)
            confirmed = mon.analyse(forming, self.now, 2)
            self.assertEqual(confirmed["decision"], "TEYİT VAR")

            forming[-1] = row(last_ts, 101, 103, 98, 99, 12)
            invalid = mon.analyse(forming, self.now, 2)
            self.assertEqual(invalid["decision"], "YAPI YOK")

    def test_top_five_per_method_replaced_within_same_fifty_coins(self):
        # Scan all 50 locally; show only 5+5 valid formations. If one leaves
        # the top five, the next valid already-scanned coin takes its place.
        tickers = []
        symbols = [f"D{i}USDT" for i in range(7)] + [
            f"C{i}USDT" for i in range(7)] + [f"OTHER{i}USDT" for i in range(36)]
        for rank, symbol in enumerate(symbols):
            tickers.append(dict(symbol=symbol, quoteVolume=1000-rank,
                                lastPrice=100,
                                priceChangePercent=-5 if symbol.startswith("D") else 5))
        market = {s: ".01" for s in symbols}
        onboard = {s: 1 for s in symbols}
        decisions = {"D0USDT": "TEYİT VAR", "C0USDT": "TEYİT VAR"}

        def analyse_symbol(rows, now, change):
            symbol = rows[0]
            method = ("DÖNÜŞ" if symbol.startswith("D") else
                      "DEVAM" if symbol.startswith("C") else "—")
            decision = decisions.get(symbol, "İZLE" if method != "—" else "YAPI YOK")
            return dict(count=480, four_count=29, sources={}, reason="test",
                        trend="YÜKSELİŞ", method=method, decision=decision,
                        stop=98, confirmed=1234, resistances=[])

        with tempfile.TemporaryDirectory() as directory:
            candles = Mock(return_value=[])
            with patch.object(mon, "cached_rows",
                              side_effect=lambda d, sym, ws, now: [sym]), \
                 patch.object(mon, "analyse", side_effect=analyse_symbol):
                first = mon.scan(directory, tickers, market, onboard,
                                 candles, self.now, {})
                self.assertEqual(len(first["rows"]), 50)
                self.assertEqual(len(first["selected"]), 10)
                self.assertEqual(first["selected"][:5],
                                 [f"D{i}USDT" for i in range(5)])
                self.assertEqual(first["selected"][5:],
                                 [f"C{i}USDT" for i in range(5)])
                sheet = mon.sheet_values(first)
                self.assertEqual([r[0] for r in sheet[6:16]], first["selected"])
                self.assertTrue(all(not any(s in r for s in symbols[14:])
                                    for r in sheet[6:16]))
                self.assertTrue(all(not any(r) for r in sheet[16:]))
                self.assertEqual(candles.call_count, 50)

                # Invalidate a watched reversal and continuation, preserve
                # other eight, promote ranks 6 from the same 50-coin pool.
                decisions.update(D1USDT="SÜRESİ DOLDU", C1USDT="REDDET")
                candles.reset_mock()
                second = mon.scan(directory, tickers, market, onboard,
                                  candles, self.now+300, first)
                self.assertEqual(len(second["selected"]), 10)
                self.assertNotIn("D1USDT", second["selected"])
                self.assertNotIn("C1USDT", second["selected"])
                self.assertIn("D5USDT", second["selected"])
                self.assertIn("C5USDT", second["selected"])
                self.assertEqual([r[0] for r in mon.sheet_values(second)[6:16]],
                                 second["selected"])
                self.assertEqual(candles.call_count, 50)

    def test_no_forced_fill_if_less_than_five_per_method(self):
        tickers = [dict(symbol=f"D{i}USDT", quoteVolume=200-i,
                        lastPrice=100, priceChangePercent=-5) for i in range(50)]
        market = {t["symbol"]: ".01" for t in tickers}
        onboard = {t["symbol"]: 1 for t in tickers}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(mon, "cached_rows", side_effect=lambda d, sym, ws, now: [sym]), \
                 patch.object(mon, "analyse", side_effect=lambda rows, now, change: dict(
                     count=480, four_count=29, sources={}, reason="test",
                     trend="DÜŞÜŞ", method="DÖNÜŞ" if rows[0] in (
                         "D0USDT", "D1USDT") else "—",
                     decision="İZLE" if rows[0] in ("D0USDT", "D1USDT")
                     else "YAPI YOK")):
                obj = mon.scan(directory, tickers, market, onboard,
                               lambda sym: [], self.now, {})
                self.assertEqual(obj["selected"], ["D0USDT", "D1USDT"])
                view = mon.sheet_values(obj)
                self.assertEqual([r[0] for r in view[6:8]], obj["selected"])
                self.assertFalse(any(view[8]))
                self.assertIn("DÖNÜŞ 2/5 | DEVAM 0/5", view[3])

    def test_zero_watchlist_explains_candle_and_formation_blockers_in_sheet(self):
        records = [
            dict(symbol="AAAUSDT", age_ready=True, count=479,
                 four_count=0, method="—", decision="VERİ BEKLENİYOR",
                 reason="480 gerçek kapanmış 15m mum gerekli"),
            dict(symbol="BBBUSDT", age_ready=False, count=480,
                 four_count=30, method="—", decision="VERİ BEKLENİYOR",
                 reason="30 günlük sözleşme geçmişi bekleniyor"),
            dict(symbol="CCCUSDT", age_ready=True, count=480,
                 four_count=30, method="—", decision="YAPI YOK",
                 reason="Etkin dönüş veya devam formasyonu yok"),
            dict(symbol="DDDUSDT", age_ready=True, count=480,
                 four_count=30, method="DEVAM", decision="REDDET",
                 reason="Stop geçersiz veya yakın direnç 1,5R alan bırakmıyor"),
        ]
        counts = mon.diagnostic_counts(records)
        self.assertEqual(counts["universe"], 4)
        self.assertEqual(counts["exact_480"], 3)
        self.assertEqual(counts["age_ready"], 3)
        self.assertEqual(counts["ready_480_and_age"], 2)
        self.assertEqual(counts["four_ready"], 2)
        self.assertEqual(counts["active"], 0)
        self.assertEqual(counts["decisions"]["VERİ BEKLENİYOR"], 2)
        obj = dict(rows=records, selected=[], observed_at=self.now,
                   references=[], mode="MONITOR_ONLY")
        values = mon.sheet_values(obj)
        self.assertEqual(values[4][0], "Veri bekleyen (arka plan)")
        self.assertEqual(values[4][1], 2)
        self.assertEqual(values[4][4], "2/4")
        self.assertEqual(values[4][6], 2)
        self.assertIn("480 gerçek kapanmış", values[4][8])
        self.assertIn("Etkin dönüş", values[4][8])
        self.assertEqual(len(values), 162)
        self.assertEqual(len(values[5]), 18)
        self.assertFalse(any(values[6]))  # Diagnostic counts never fake a signal.

    def test_scan_attaches_structural_diagnostics_without_altering_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            tickers = [dict(symbol="AAAUSDT", quoteVolume=1e8,
                            lastPrice=100, priceChangePercent=2)]
            obj = mon.scan(directory, tickers, {"AAAUSDT": ".01"},
                           {"AAAUSDT": 1}, lambda s: [], self.now, {})
            summary = obj["diagnostic_summary"]
            self.assertEqual(summary["universe"], 1)
            self.assertEqual(summary["exact_480"], 0)
            self.assertEqual(summary["age_ready"], 1)
            self.assertEqual(summary["active"], 0)
            self.assertEqual(obj["selected"], [])
            self.assertEqual(obj["events"], [])

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
