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

    def test_partial_risk_visibility_is_independent(self):
        _, table, _, _ = self.data({"weak_volume": True,
                                    "falling_rsi_ema": None,
                                    "repeat_lower_entry": False})
        row = table[1]
        self.assertEqual(row[14], "🔴 VAR")
        self.assertEqual(row[15], "VERİ EKSİK")
        self.assertEqual(row[16], "🟢 YOK")
        self.assertIn("EKSİK", row[17])

    def test_missing_risk_never_claims_safe_or_displays_question_marks(self):
        rec = dict(self.record)
        rec.pop("failure_shadow")
        _, table, colors, _ = dash.build_rows(
            {"symbols":{"TESTUSDT":rec}, "history":[rec]},
            self.tickers, {}, {}, self.now)
        self.assertEqual(table[1][14:17], ["VERİ EKSİK"] * 3)
        self.assertIn("EKSİK", table[1][17])
        self.assertEqual(colors[0], ("red", "grey"))

    def test_top_loser_without_v2_candidate_uses_only_cached_live_score(self):
        score = Mock(return_value={"strength_score": {"score": 64.5}})
        _, table, _, _ = dash.build_rows(
            {"symbols":{}, "history":[]}, self.tickers, {}, {}, self.now,
            rescore=score)
        self.assertEqual(table[1][4], "ADAY KAYDI YOK — NEDEN KAYDEDİLMEMİŞ")
        self.assertEqual(table[1][7], 64.5)
        self.assertEqual(table[1][14:17], ["—"] * 3)
        self.assertEqual(score.call_args.args[0]["symbol"], "TESTUSDT")

    def test_non_signal_rsi_and_recovery_use_only_validated_cached_values(self):
        view = Mock(return_value={
            "status": "FILTERED_V2", "qualified": False,
            "rsi14": 31.25, "recovery_4h_pct": 0.75,
            "strength_score": {"score": 43.5}})
        _, table, _, cache = dash.build_rows(
            {"symbols": {}, "history": []}, self.tickers, {}, {},
            self.now, rescore=view)
        row = table[1]
        self.assertEqual(row[4], "ADAY KAYDI YOK — NEDEN KAYDEDİLMEMİŞ")
        self.assertEqual(row[5:8], [31.25, 0.75, 43.5])
        self.assertEqual(cache["TESTUSDT"]["score"], 43.5)
        self.assertNotIn("TESTUSDT", dash.build_rows(
            {"symbols": {}, "history": []}, [], {}, {}, self.now)[3])


    def test_no_candidate_below_drop_threshold_is_not_a_v2_verdict(self):
        tickers = [{"symbol": "TESTUSDT", "priceChangePercent": "-8",
                    "lastPrice": "1.002"}]
        _, table, colors, _ = dash.build_rows(
            {"symbols": {}, "history": []}, tickers, {}, {}, self.now)
        self.assertEqual(table[1][4], "EŞİK DIŞI — 24s düşüş %12 altında")
        self.assertEqual(table[1][18], "⚪ AKTİF ADAY KAYDI YOK")
        self.assertEqual(table[1][14:17], ["—"] * 3)
        self.assertEqual(colors[0][0], "grey")

    def test_no_candidate_with_missing_candles_is_not_called_no_signal(self):
        view = Mock(return_value={"status": "MISSING_5D_HISTORY",
                                  "candles": 479})
        _, table, _, _ = dash.build_rows(
            {"symbols": {}, "history": []}, self.tickers, {}, {},
            self.now, rescore=view)
        self.assertEqual(table[1][4], "ADAY KAYDI YOK — GÖSTERGE VERİSİ EKSİK")
        self.assertEqual(table[1][7], "HESAPLANAMADI")
        self.assertEqual(table[1][14:17], ["—"] * 3)

    def test_non_signal_previous_power_and_delta_on_next_tick(self):
        view = Mock(return_value={"status": "FILTERED_V2",
                                  "rsi14": 23, "recovery_4h_pct": 0.44,
                                  "strength_score": {"score": 47.0}})
        _, table, _, cache = dash.build_rows(
            {"symbols": {}, "history": []}, self.tickers, {}, {},
            self.now, rescore=view)
        view.return_value["strength_score"]["score"] = 49.0
        _, updated, _, _ = dash.build_rows(
            {"symbols": {}, "history": []}, self.tickers, {}, {},
            self.now + 300, score_cache=cache, rescore=view)
        self.assertEqual(updated[1][8:10], [47, 2])

    def test_unknown_current_data_not_invented(self):
        view = Mock(return_value={"status": "MISSING_5D_HISTORY",
                                  "candles": 477})
        _, table, _, cache = dash.build_rows(
            {"symbols": {}, "history": []}, self.tickers, {}, {},
            self.now, rescore=view)
        self.assertEqual(table[1][5:7], ["—", "—"])
        self.assertEqual(table[1][7], "HESAPLANAMADI")
        self.assertEqual(cache, {})

    def test_format_resets_old_indicator_fills_before_current_colors(self):
        book = Mock()
        status = SimpleNamespace(title="Bot Status", id=123)
        signal = SimpleNamespace(title="Sinyal", id=456)
        rows = [dash.HEADERS, [""] * len(dash.HEADERS)]
        rows[1][5:8] = [30, 1.2, 55]
        dash._format(book, status, signal, [("green", "grey")], rows)
        ops = book.batch_update.call_args.args[0]["requests"]
        clears = [op["repeatCell"] for op in ops
                  if (op.get("repeatCell", {}).get("cell") ==
                      {"userEnteredFormat": {}})]
        self.assertEqual({(op["range"]["startColumnIndex"],
                           op["range"]["endColumnIndex"])
                          for op in clears}, {(5, 8), (13, 17)})
        self.assertTrue(all(op["range"]["endRowIndex"] ==
                            dash.MAX_ROWS + 1 for op in clears))
        self.assertTrue(all(op["fields"] ==
                            "userEnteredFormat.backgroundColor" for op in clears))

    def test_retrospective_power_uses_original_closed_candles(self):
        rec = dict(self.record,
                   v2_status="MISSING_5D_HISTORY",
                   v2_features={"candles": 477},
                   retrospective_v2={"status":"STRONG_V2",
                                     "candles":480, "research_only":True,
                                     "strength_score":{"score":49.5}})
        _, table, _, _ = dash.build_rows(
            {"symbols":{"TESTUSDT":rec}, "history":[rec]},
            self.tickers, {}, {}, self.now)
        self.assertEqual(table[1][7], 49.5)
        self.assertEqual(table[1][4], "MISSING_5D_HISTORY")
        self.assertIn("canlı emir yok", table[1][39])

    def test_stored_late_research_never_shows_live_order(self):
        rec = dict(self.record, v2_status="MISSING_5D_HISTORY",
                   v2_reason="HISTORY_NOT_READY_IN_TIME",
                   retrospective_v2={"status":"STRONG_V2",
                                     "candles":479,"research_only":True})
        _, table, _, _ = dash.build_rows(
            {"symbols":{"TESTUSDT":rec}, "history":[rec]},
            self.tickers, {}, {}, self.now)
        self.assertIn("SONRADAN V2 ONAYLI (479/480)", table[1][39])
        self.assertIn("canlı emir yok", table[1][39])
        self.assertEqual(table[1][4], "MISSING_5D_HISTORY")

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
        self.assertEqual(table[1][4],"ÖNCEKİ ADAY SÜRESİ DOLDU")
        self.assertEqual(table[1][21],"—")

    def test_open_virtual_trade_stays_visible_after_24h(self):
        old=dict(self.record,created_at=self.now-24*3600-1,
                 tracking_status="OPEN",entered_at=self.now-3600)
        _, table, _, _=dash.build_rows(
            {"symbols":{"TESTUSDT":old},"history":[old]},self.tickers,
            {},{},self.now)
        self.assertEqual(table[1][4],"STRONG_V2")
        self.assertEqual(table[1][19],"OPEN")

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
        self.assertIn("Aday süresi içinde karar verilemedi", row[38])
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

    def test_both_bot_status_tab_spellings_update_in_one_batch(self):
        # Regression: normalized keys 'Bot Status' and 'BOT_STATUS' collide.
        for titles in (
            ["BOT_STATUS", "Bot Status", "Sinyal"],
            ["Bot Status", "Sinyal", "BOT_STATUS"],
        ):
            with self.subTest(order=titles):
                book = Mock(title="EMA Filter")
                book.worksheets.return_value = [
                    SimpleNamespace(title=title, id=i+1)
                    for i, title in enumerate(titles)]
                client = Mock()
                client.open_by_key.return_value = book
                with patch.object(dash, "_credentials",
                                  return_value=(client, "testing-id")), \
                     patch.object(dash, "_read_previous", return_value={}), \
                     patch.object(dash, "_write_previous"), \
                     patch.object(dash.os, "makedirs"):
                    self.assertTrue(dash.publish(
                        {"symbols": {"TESTUSDT": self.record}, "history": []},
                        self.tickers, {"auto": "AÇIK", "daily": 7,
                                       "max_daily": 50},
                        {"connected": True}, self.now, "/tmp/test-dashboard"))
                book.values_batch_update.assert_called_once()
                data = book.values_batch_update.call_args.args[0]["data"]
                self.assertEqual([d["range"].split("'!")[0] for d in data],
                                 ["'Bot Status", "'Sinyal", "'BOT_STATUS"])
                self.assertEqual(data[0]["values"][1][0],
                                 "Son güncelleme (Paris)")
                self.assertEqual(data[2]["range"], "'BOT_STATUS'!A1:B32")
                self.assertEqual(data[2]["values"][0][0], "Last alive")
                self.assertIn("UTC", data[2]["values"][0][1])
                self.assertIn("CRASH BOUNCE V2", data[2]["values"][1][1])
                self.assertIn("KAPALI", data[2]["values"][2][1])
                self.assertIn("Otomatik alış",
                              [r[0] for r in data[2]["values"]])
                self.assertIn(["Günlük giriş limiti", 50],
                              data[2]["values"])
                self.assertEqual(len(data[2]["values"]), 32)
                self.assertFalse(book.add_worksheet.called)
                self.assertFalse(book.del_worksheet.called)

    def test_old_bot_status_workbook_updates_when_separate_from_ema_filter(self):
        modern = Mock(title="EMA Filter")
        modern.worksheets.return_value = [
            SimpleNamespace(title="Bot Status", id=100),
            SimpleNamespace(title="Sinyal", id=200)]
        old = Mock(title="Original Pre-Signal Sheet")
        old.worksheets.return_value = [
            SimpleNamespace(title="BOT_STATUS", id=300),
            SimpleNamespace(title="PRE_SIGNAL_TEST", id=301)]
        client = Mock()
        client.open_by_key.side_effect = lambda key: (
            modern if key == "modern-book" else
            old if key == "old-book" else None)
        logs = []
        with patch.dict(dash.os.environ, {"GOOGLE_SHEETS_ID": "old-book"}), \
             patch.object(dash, "_credentials",
                          return_value=(client, "modern-book")), \
             patch.object(dash, "_read_previous", return_value={}), \
             patch.object(dash, "_write_previous"), \
             patch.object(dash.os, "makedirs"):
            self.assertTrue(dash.publish(
                {"symbols": {}, "history": []},
                self.tickers, {"daily": 4, "max_daily": 28},
                {"connected": True}, self.now, "/tmp/test-dashboard", logger=logs.append))
        self.assertEqual(modern.values_batch_update.call_count, 1)
        self.assertEqual(old.values_batch_update.call_count, 1)
        legacy = old.values_batch_update.call_args.args[0]["data"]
        self.assertEqual(len(legacy), 1)
        self.assertEqual(legacy[0]["range"], "'BOT_STATUS'!A1:B32")
        self.assertEqual(legacy[0]["values"][0][0], "Last alive")
        self.assertIn(["Günlük giriş limiti", 28], legacy[0]["values"])
        self.assertTrue(any("BOT_STATUS refreshed" in line for line in logs))
        self.assertFalse(old.add_worksheet.called)
        self.assertFalse(old.del_worksheet.called)
        self.assertFalse(modern.add_worksheet.called)
        self.assertFalse(modern.del_worksheet.called)

    def test_external_legacy_status_failure_does_not_block_modern_dashboard(self):
        modern = Mock(title="EMA Filter")
        modern.worksheets.return_value = [
            SimpleNamespace(title="Bot Status", id=100),
            SimpleNamespace(title="Sinyal", id=200)]
        client = Mock()
        def open_book(key):
            if key == "modern-book":
                return modern
            raise PermissionError("sensitive data is never logged")
        client.open_by_key.side_effect = open_book
        logs = []
        with patch.dict(dash.os.environ, {"GOOGLE_SHEETS_ID": "old-book"}), \
             patch.object(dash, "_credentials",
                          return_value=(client, "modern-book")), \
             patch.object(dash, "_read_previous", return_value={}), \
             patch.object(dash, "_write_previous"), \
             patch.object(dash.os, "makedirs"):
            self.assertTrue(dash.publish(
                {"symbols": {}, "history": []}, self.tickers, {},
                {"connected": True}, self.now, "/tmp/test-dashboard",
                logger=logs.append))
        modern.values_batch_update.assert_called_once()
        self.assertEqual(logs, ["[CRASH SHEETS LEGACY] update skipped: PermissionError"])

    def test_only_legacy_bot_status_fallback_remains_live(self):
        book = Mock(title="EMA Filter")
        book.worksheets.return_value = [
            SimpleNamespace(title="Sinyal", id=2),
            SimpleNamespace(title="BOT_STATUS", id=1)]
        client = Mock()
        client.open_by_key.return_value = book
        with patch.object(dash, "_credentials",
                          return_value=(client, "testing-id")), \
             patch.object(dash, "_read_previous", return_value={}), \
             patch.object(dash, "_write_previous"), \
             patch.object(dash.os, "makedirs"):
            self.assertTrue(dash.publish(
                {"symbols": {}, "history": []},
                self.tickers, {}, {"connected": True},
                self.now, "/tmp/test-dashboard"))
        batch = book.values_batch_update.call_args.args[0]["data"]
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0]["range"], "'BOT_STATUS'!A1:B26")
        self.assertEqual(batch[1]["range"], "'Sinyal'!A1:AN61")

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
