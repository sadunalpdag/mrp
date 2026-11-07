import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import crash_sheets_dashboard as dash


class SheetDashboardTests(unittest.TestCase):
    def setUp(self):
        self.now = 1794050000.0
        self.record = {
            "symbol": "TESTUSDT", "signal_id": "s-1", "created_at": self.now - 60,
            "v2_status": "STRONG_V2", "tracking_status": "WAIT_ENTRY",
            "entry": "1.00", "tp": "1.01",
            "v2_features": {
                "rsi14": 25.0, "recovery_4h_pct": 1.2,
                "strength_score": {"score": 55.0},
            },
            "failure_shadow": {"flags": {
                "weak_volume": True, "falling_rsi_ema": True,
                "repeat_lower_entry": False}},
        }
        self.tickers = [{"symbol": "TESTUSDT", "priceChangePercent": "-24",
                         "lastPrice": "1.002"}]

    def data(self, flags=None, score_previous=None):
        rec = dict(self.record)
        if flags is not None:
            rec["failure_shadow"] = {"flags": flags}
        return dash.build_rows(
            {"symbols": {"TESTUSDT": rec}, "history": [rec]}, self.tickers,
            {"auto": "AÇIK"}, {"connected": True}, self.now,
            score_previous or {}, None)

    def test_risk_two_is_half_position_and_yellow(self):
        status, table, colors, previous = self.data()
        row = table[1]
        self.assertEqual(len(row), len(dash.HEADERS))
        self.assertEqual(len(dash.HEADERS), 40)
        self.assertEqual(row[22], "⚪ VERİ YOK")
        self.assertEqual(row[26], "0/480")
        self.assertEqual(row[13], 2)
        self.assertIn("%50", row[17])
        self.assertEqual(colors[0], ("yellow", "yellow"))
        self.assertEqual(status[3][1], "BAĞLI")
        self.assertEqual(previous["TESTUSDT"]["score"], 55)

    def test_risk_zero_green_and_three_blocked(self):
        keys = dash.FLAGS
        _, table, colors, _ = self.data({k: False for k in keys})
        self.assertEqual(table[1][13], 0)
        self.assertEqual(colors[0], ("green", "green"))
        _, table, colors, _ = self.data({k: True for k in keys})
        self.assertEqual(table[1][13], 3)
        self.assertEqual(colors[0], ("red", "red"))
        self.assertIn("İŞLEM YOK", table[1][17])

    def test_missing_risk_is_unknown_not_zero(self):
        _, table, colors, _ = self.data({"weak_volume": True})
        self.assertEqual(table[1][13], "—")
        self.assertEqual(colors[0], ("red", "grey"))

    def test_previous_score_delta_and_current_rescore(self):
        prior = {"TESTUSDT": {"signal_id": "s-1", "score": 55.0}}
        state = {"symbols": {"TESTUSDT": self.record}, "history": [self.record]}
        _, table, _, cache = dash.build_rows(
            state, self.tickers, {}, {}, self.now, prior,
            lambda record, now: {"strength_score": {"score": 58.5}})
        self.assertEqual(table[1][8], 55)
        self.assertEqual(table[1][9], 3.5)
        self.assertEqual(cache["TESTUSDT"]["score"], 58.5)

    def test_expired_signal_is_not_shown_as_active_even_in_top20(self):
        old=dict(self.record,created_at=self.now - 24*3600 - 1)
        _, table, _, _=dash.build_rows(
            {"symbols":{"TESTUSDT":old},"history":[]},self.tickers,
            {},{},self.now)
        self.assertEqual(table[1][0],"TESTUSDT")
        self.assertEqual(table[1][4],"SİNYAL YOK")
        self.assertEqual(table[1][21],"—")

    def test_479_latest_missing_is_not_mislabeled_as_failed_v2(self):
        rec = dict(self.record, v2_status="MISSING_5D_HISTORY",
                   v2_reason="HISTORY_NOT_READY_IN_TIME",
                   v2_features={"candles": 479,
                                "missing_reason": "LAST_CLOSED_CANDLE_MISSING"})
        rescore = Mock(return_value={"status": "STRONG_V2", "qualified": True,
                                      "candles": 479})
        _, table, colors, _ = dash.build_rows(
            {"symbols": {"TESTUSDT": rec}, "history": [rec]}, self.tickers,
            {}, {}, self.now, rescore=rescore)
        row = table[1]
        self.assertIn("VERİ EKSİK", row[18])
        self.assertEqual(colors[0][0], "yellow")
        self.assertEqual(row[37], "479/480")
        self.assertIn("90 saniye", row[38])
        self.assertIn("son kapanmış", row[38])
        self.assertIn("SONRADAN V2 ONAYLI", row[39])
        self.assertIn("canlı emir yok", row[39])
        rescore.assert_called_once_with(rec, rec["created_at"])

    def test_479_current_history_is_not_confused_with_signal_time(self):
        rec = dict(self.record, v2_status="MISSING_5D_HISTORY",
                   v2_features={"candles": 455,
                                "missing_reason": "TOO_FEW_5D_CANDLES"})
        with patch("crash_levels_5d.load_cached", return_value=[]), \
                patch("crash_levels_5d.analyze", return_value={
                    "status": "VERİ YOK", "direction": "—", "coverage": "479/480",
                    "distance_pct": None, "confirmed_at": None, "support": None,
                    "resistance": None, "fib": {}, "fib_band": "—"}):
            _, table, _, _ = dash.build_rows(
                {"symbols": {"TESTUSDT": rec}, "history": [rec]}, self.tickers,
                {}, {}, self.now, candle_get=lambda _: [], directory="/tmp")
        self.assertEqual(table[1][26], "479/480")  # current trend window
        self.assertEqual(table[1][37], "455/480")  # original candidate

    def test_no_sheet_deletion_or_creation_on_missing_tab(self):
        book = Mock()
        book.title = "EMA Filter"
        book.worksheets.return_value = [SimpleNamespace(title="Bot Status", id=1)]
        with self.assertRaisesRegex(ValueError, "TAB_MISSING"):
            dash._tabs(book)
        self.assertFalse(book.add_worksheet.called)
        self.assertFalse(book.del_worksheet.called)

    def test_wrong_workbook_is_rejected(self):
        book = Mock()
        book.title = "Completely unrelated"
        with self.assertRaisesRegex(ValueError, "NOT_EMA_FILTER"):
            dash._tabs(book)
        self.assertFalse(book.worksheets.called)

    def test_update_uses_existing_tabs_and_does_not_reset(self):
        book = Mock()
        book.title = "EMA Filter"
        book.worksheets.return_value = [
            SimpleNamespace(title="Bot Status", id=123),
            SimpleNamespace(title="Sinyal", id=456)]
        client = Mock()
        client.open_by_key.return_value = book
        with patch.object(dash, "_credentials", return_value=(client, "testing-id")), \
                patch.object(dash, "_read_previous", return_value={}), \
                patch.object(dash, "_write_previous") as state_store:
            with patch.object(dash.os, "makedirs"):
                self.assertTrue(dash.publish(
                    {"symbols": {"TESTUSDT": self.record}, "history": []},
                    self.tickers, {}, {}, self.now, "/tmp/test-dashboard"))
        batch = book.values_batch_update.call_args.args[0]
        self.assertEqual(len(batch["data"]), 2)
        self.assertEqual(batch["data"][0]["range"], "'Bot Status'!A1:B26")
        self.assertEqual(batch["data"][1]["range"], "'Sinyal'!A1:AN61")
        self.assertEqual(len(batch["data"][1]["values"]), 61)
        self.assertTrue(state_store.called)
        self.assertFalse(book.add_worksheet.called)
        self.assertFalse(book.del_worksheet.called)


if __name__ == "__main__":
    unittest.main()
