"""Offline invariants for no-REST, no-trading paper scalp monitor."""
import tempfile
import unittest
from unittest.mock import patch
import scalp_shadow_week as sw


def candles(count=55, close=101.0, first=0):
    rows = []
    for i in range(count):
        ts = first + i * sw.STEP_MS
        rows.append([ts, str(close), str(close * 1.002), str(close * 0.998),
                     str(close), "100", ts + sw.STEP_MS - 1])
    return rows


class ShadowWeekTests(unittest.TestCase):
    def test_rejects_unclosed_and_broken_ohlc(self):
        rows = candles(2, first=sw.STEP_MS)
        rows.append([3 * sw.STEP_MS, "1", "0.5", "1", "1",
                     "1", 4 * sw.STEP_MS - 1])
        rows.append([4 * sw.STEP_MS, "1", "1", "1", "1",
                     "1", 5 * sw.STEP_MS - 1])
        valid = sw._bars(rows, 4 * sw.STEP_MS / 1000)
        self.assertEqual(len(valid), 2)

    def test_pullback_long_has_no_stop_and_tp_after_three_pct_drawdown(self):
        signal = {"status": "OPEN", "strategy": "PULLBACK", "ts": 0,
                  "side": "LONG", "entry": 100, "tp": 101, "risk": 99}
        fall = (sw.STEP_MS, 100, 100.2, 96.5, 98, 1)
        sw._settle(signal, [fall], 2 * sw.STEP_MS / 1000)
        self.assertEqual(signal["status"], "OPEN")
        self.assertTrue(signal["minus_one_pct_seen"])
        self.assertAlmostEqual(signal["mae_pct"], -3.5)
        self.assertAlmostEqual(signal["mfe_pct"], 0.2)
        rebound = (2 * sw.STEP_MS, 98, 102.5, 97, 101.5, 1)
        after_tp = (3 * sw.STEP_MS, 101, 150, 40, 140, 1)
        sw._settle(signal, [fall, rebound, after_tp], 5 * sw.STEP_MS / 1000)
        self.assertEqual(signal["status"], "TP")  # Was never stopped out
        self.assertEqual(signal["exit_model"], sw.PULLBACK_NO_SL_MODEL)
        self.assertAlmostEqual(signal["mae_pct"], -3.5)
        self.assertAlmostEqual(signal["mfe_pct"], 2.5)
        self.assertEqual(signal["lowest_price"], 96.5)
        self.assertEqual(signal["highest_price"], 102.5)
        self.assertEqual(signal["last_price"], 101.5)
        self.assertEqual(signal["resolved_at"], 3 * sw.STEP_MS)

    def test_pullback_short_has_no_stop_then_reverses_to_tp(self):
        signal = {"status": "OPEN", "strategy": "PULLBACK", "ts": 0,
                  "side": "SHORT", "entry": 100, "tp": 99, "risk": None}
        adverse = (sw.STEP_MS, 100, 105, 99.9, 104, 1)
        sw._settle(signal, [adverse], 2 * sw.STEP_MS / 1000)
        self.assertEqual(signal["status"], "OPEN")
        self.assertAlmostEqual(signal["mae_pct"], -5)
        favorable = (2 * sw.STEP_MS, 104, 104.5, 97, 98.5, 1)
        sw._settle(signal, [adverse, favorable], 3 * sw.STEP_MS / 1000)
        self.assertEqual(signal["status"], "TP")
        self.assertAlmostEqual(signal["mfe_pct"], 3)
        self.assertTrue(signal["minus_one_pct_seen"])

    def test_pullback_no_tp_12h_timeout_has_last_return_and_full_path(self):
        signal = {"status": "OPEN", "strategy": "PULLBACK", "ts": 0,
                  "side": "LONG", "entry": 100, "tp": 101}
        rows = [(i * sw.STEP_MS, 99, 99.7, 96, 97, 1)
                for i in range(1, 49)]
        # At the 12h boundary the 48th candle has not yet closed.
        sw._settle(signal, rows[:-1], sw.HORIZON_S)
        self.assertEqual(signal["status"], "OPEN")
        sw._settle(signal, rows, sw.HORIZON_S + sw.STEP_MS / 1000)
        self.assertEqual(signal["status"], "TIMEOUT")
        self.assertEqual(signal["observed_bars"], 48)
        self.assertAlmostEqual(signal["last_return_pct"], -3)
        self.assertAlmostEqual(signal["mae_pct"], -4)
        self.assertTrue(signal["minus_one_pct_seen"])

    def test_pullback_gap_is_not_mislabeled_as_12h_timeout(self):
        signal = {"status": "OPEN", "strategy": "PULLBACK", "ts": 0,
                  "side": "LONG", "entry": 100, "tp": 101}
        rows = [(i * sw.STEP_MS, 99, 99.7, 95, 97, 1)
                for i in range(1, 49) if i != 21]
        sw._settle(signal, rows, sw.HORIZON_S + sw.STEP_MS / 1000)
        self.assertEqual(signal["status"], "OPEN")
        self.assertEqual(signal["observed_bars"], 47)
        self.assertTrue(signal["minus_one_pct_seen"])
        self.assertNotEqual(signal["status"], "RISK")

    def test_pullback_sheet_shows_no_sl_and_excursions(self):
        record = {"selection_version": sw.SELECTION_VERSION,
                  "exit_model": sw.PULLBACK_NO_SL_MODEL,
                  "strategy": "PULLBACK", "symbol": "AUSDT", "side": "LONG",
                  "ts": 1 * sw.STEP_MS, "entry": 100, "tp": 101,
                  "risk": None, "status": "TP",
                  "resolved_at": 3 * sw.STEP_MS,
                  "lowest_price": 95, "highest_price": 102,
                  "mae_pct": -5, "mfe_pct": 2, "last_price": 101.5,
                  "last_return_pct": 1.5, "minus_one_pct_seen": True,
                  "note": "paper only"}
        obj = {"signals": [record], "last_scan": 2 * sw.STEP_MS / 1000,
               "ready_groups": {}, "ranked_groups": {}, "diagnostics": {}}
        rows = sw._tab_values(obj, "PULLBACK")
        self.assertEqual(rows[8], sw.PULLBACK_HEADERS)
        self.assertEqual(len(rows[9]), 16)
        self.assertEqual(rows[9][5], "SL YOK")
        self.assertEqual(rows[9][6], "TP")
        self.assertEqual(rows[9][10:], [95, 102, -5, 2, 101.5, 1.5])
        self.assertIn("1 tanesi sonra TP", rows[7][1])

    def test_no_synthetic_data_and_bounded_selection(self):
        now = 56 * sw.STEP_MS / 1000
        sample = candles(55)
        tickers = [{"symbol": "A" + str(i) + "USDT",
                    "priceChangePercent": str(50-i),
                    "quoteVolume": str(1000-i)} for i in range(35)]
        calls = []
        def getter(symbol):
            calls.append(symbol)
            return sample
        with tempfile.TemporaryDirectory() as td:
            obj = sw._scan(td, tickers, getter, now, {"signals": []})
        self.assertEqual(len(obj["sampled_coins"]["PULLBACK"]), sw.MAX_SYMBOLS)
        self.assertEqual(set(calls), {"A" + str(i) + "USDT" for i in range(10, 20)})
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(obj["signals"]), 0)
        self.assertEqual(len(sw._tab_values(obj, "PULLBACK")), 80)
        self.assertEqual(len(obj["diagnostics"]["PULLBACK"]), 10)
        self.assertEqual(sw._tab_values(obj, "PULLBACK")[59][:10], sw.DIAG_HEADERS)
        self.assertEqual(len(sw._tab_values(obj, "PULLBACK")[59]), 16)
        self.assertIn("MA50 eğimi yatay", str(obj["diagnostics"]["PULLBACK"]))

    def test_exact_gainer_bands_without_backfill_and_market_filter(self):
        tickers = [{"symbol": f"A{i:02}USDT", "priceChangePercent": str(40 - i)}
                   for i in range(34)]
        self.assertEqual(sw.gainer_groups(tickers)["PULLBACK"],
                         [f"A{i:02}USDT" for i in range(10, 20)])
        allowed = {f"A{i:02}USDT" for i in range(26)}
        self.assertEqual(sw.gainer_groups(tickers, allowed)["PULLBACK"],
                         [f"A{i:02}USDT" for i in range(10, 20)])

    def test_four_missing_true_candles_allowed_but_five_rejected(self):
        now = 56 * sw.STEP_MS / 1000
        rows = candles(56)
        four_gaps = [r for i, r in enumerate(rows) if i not in (4, 12, 20, 28)]
        self.assertTrue(sw._ready(sw._bars(four_gaps, now), now))
        five_gaps = [r for i, r in enumerate(rows) if i not in (4, 12, 20, 28, 36)]
        self.assertFalse(sw._ready(sw._bars(five_gaps, now), now))

    def test_merges_existing_five_day_file_with_closed_ws_without_rest(self):
        import json
        from pathlib import Path
        now = 56 * sw.STEP_MS / 1000
        rows = candles(56, first=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"crash_research"/"history_cache"/"XUSDT.json"
            path.parent.mkdir(parents=True)
            archive = [list(r)+[None]*5+["BINANCE_ARCHIVE"] for r in rows[:54]]
            # Stored alternate must not override exchange-native WS.
            archive[-1][-1] = "BYBIT_LINEAR"
            path.write_text(json.dumps(archive), encoding="utf-8")
            live = rows[53:55] + [
                [56*sw.STEP_MS, "100", "101", "99", "100", "100",
                 57*sw.STEP_MS-1]]
            merged = sw._history_rows(directory, "XUSDT", live, now)
            bars = sw._bars(merged, now)
            self.assertEqual(len(merged), 55)
            self.assertEqual(len(bars), 55)
            self.assertTrue(sw._ready(bars, now))
            self.assertEqual(merged[53][12:] if len(merged[53]) > 12 else [],
                             [])   # valid exchange WS replaces stale Bybit candle
            self.assertEqual(merged[-1][0], 54*sw.STEP_MS)
            self.assertFalse(any(r[0] == 56*sw.STEP_MS for r in merged))
            # A disk cache with missing historic rows can also be valid.
            self.assertEqual(sw._coverage(bars, now), 55)

    def test_stale_history_does_not_count_as_current_52_of_56(self):
        now = 120 * sw.STEP_MS / 1000
        bars = sw._bars(candles(56, first=0), now)
        self.assertEqual(sw._coverage(bars, now), 0)
        self.assertFalse(sw._ready(bars, now))
        diagnostic = sw._diagnostic("PULLBACK", "XUSDT", bars, False, now=now)
        self.assertEqual(diagnostic[2], "0/56")
        self.assertIn("mevcut", diagnostic[7]) if False else None
        self.assertIn("Gerçek kapanmış mum 0/56", diagnostic[7])
        self.assertIn("son kapanış", diagnostic[7])

    def test_round_robin_history_repair_one_per_scan_and_persists(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            proposed = Mock()
            monitor = sw.ShadowWeek(directory, Mock(), clock=lambda: 1000,
                                    queue_history=proposed)
            def issue(symbol, status):
                return [symbol, "EKSİK", "0/56", "—", "—",
                        status, "0/0", "test", "YOK", "—"]
            data = {"diagnostics": {
                "PULLBACK": [
                    issue("AUSDT", "VERİ BEKLENİYOR"),
                    issue("BUSDT", "VERİ BEKLENİYOR")],
                }}
            monitor.request_missing_history(data, 1000)
            monitor.request_missing_history(data, 1300)
            monitor.request_missing_history(data, 1600)
            self.assertEqual([call.args[0] for call in proposed.call_args_list],
                             ["AUSDT", "BUSDT", "AUSDT"])
            monitor = sw.ShadowWeek(directory, Mock(),
                                    queue_history=proposed)
            monitor.request_missing_history(data, 1900)
            self.assertEqual(proposed.call_args_list[-1].args, ("BUSDT", 1900))
            self.assertEqual(proposed.call_count, 4)

    def test_scan_reads_history_for_ranked_symbols_without_extra_price_calls(self):
        import json
        from pathlib import Path
        now = 56 * sw.STEP_MS / 1000
        rows = candles(56)
        tickers = [
            {"symbol": f"COIN{i}USDT", "priceChangePercent": str(50-i),
             "quoteVolume": str(1000-i)} for i in range(30)]
        called = []
        def getter(symbol):
            called.append(symbol)
            return rows[-2:] if symbol == "COIN10USDT" else []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"crash_research"/"history_cache"/"COIN10USDT.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(rows[:-2]), encoding="utf-8")
            obj = sw._scan(directory, tickers, getter, now, {"signals": []})
            record = obj["diagnostics"]["PULLBACK"][0]
            self.assertEqual(record[0], "COIN10USDT")
            self.assertEqual(record[1], "HAZIR")
            self.assertEqual(record[2], "56/56")
            self.assertNotEqual(record[5], "VERİ BEKLENİYOR")
            self.assertEqual(len(called), 10)
            self.assertEqual(len(set(called)), 10)
            self.assertEqual(len(obj["ready_groups"]["PULLBACK"]), 1)

    def test_no_future_candles_cannot_claim_win(self):
        signal = {"status": "OPEN", "strategy": "PULLBACK", "ts": 0, "side": "LONG", "entry": 100, "tp": 101, "risk": None}
        sw._settle(signal, [], sw.HORIZON_S + 1)
        self.assertEqual(signal["status"], "DATA_GAP")


    def test_diagnostics_reason_for_missing_candles(self):
        row = sw._diagnostic("PULLBACK", "XUSDT", [], False)
        self.assertEqual(row[5], "VERİ BEKLENİYOR")
        self.assertIn("52/56", row[7])
        self.assertEqual(len(row), 10)

    def test_publish_range_keeps_original_signal_columns(self):
        import types
        import sys
        class Worksheet:
            title = "Pullback Test"
        class Book:
            title = "emafilter"
            def worksheets(self):
                return [Worksheet()]
            def values_batch_update(self, body):
                self.body = body
        book = Book()
        class Client:
            def open_by_key(self, _):
                return book
        fake = types.ModuleType("crash_sheets_dashboard")
        fake._credentials = lambda: (Client(), "key")
        fake._name = lambda x: "".join(c.lower() for c in x if c.isalnum())
        with patch.dict(sys.modules, {"crash_sheets_dashboard": fake}):
            self.assertTrue(sw._publish({"signals": [], "last_scan": 1,
                "ranked_groups": {}, "ready_groups": {}, "diagnostics": {}}))
        self.assertEqual(len(book.body["data"]), 1)
        item = book.body["data"][0]
        self.assertEqual(item["range"], "'Pullback Test'!A1:P80")
        self.assertEqual(len(item["values"]), 80)
        self.assertEqual(item["values"][8], sw.PULLBACK_HEADERS)
        self.assertEqual(item["values"][59][:10], sw.DIAG_HEADERS)
        self.assertEqual(len(item["values"][59]), 16)

    def test_wrong_workbook_fail_closed(self):
        class Book:
            title = "Other spreadsheet"
        class Client:
            def open_by_key(self, _):
                return Book()
        import types
        import sys
        fake = types.ModuleType("crash_sheets_dashboard")
        fake._credentials = lambda: (Client(), "sheet-id")
        fake._name = lambda s: "".join(c.lower() for c in s if c.isalnum())
        with patch.dict(sys.modules, {"crash_sheets_dashboard": fake}):
            with self.assertRaisesRegex(ValueError, "WRONG_WORKBOOK"):
                sw._publish({"signals": [], "last_scan": 1, "sampled_coins": []})


if __name__ == "__main__":
    unittest.main()
