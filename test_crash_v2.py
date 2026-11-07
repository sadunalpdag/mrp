"""Offline regression checks for V2-only signal filter."""
import ast
import math
import unittest
from pathlib import Path
from unittest.mock import Mock

import crash_v2


class CrashV2Tests(unittest.TestCase):
    def rows(self, at, low="0.98", close="1.02"):
        end = int(at * 1000) // 900000 * 900000
        return [[t, "1", "1.06", low, close, "100", t + 899999]
                for t in range(end - 480*900000, end, 900000)]

    def test_complete_history_and_recovery(self):
        data = crash_v2.assess_entry(self.rows(2000000000), [], 2000000000)
        self.assertEqual(data["status"], "STRONG_V2")
        self.assertEqual(data["candles"], 480)
        self.assertGreater(data["recovery_4h_pct"], 1)

    def test_missing_latest_closed_candle_fails_closed(self):
        rows = self.rows(2000000000)
        result = crash_v2.assess_entry(rows[:-1], [], 2000000000)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["candles"], 479)
        self.assertEqual(result["missing_reason"], "LAST_CLOSED_CANDLE_MISSING")
        self.assertFalse(result["latest_closed_present"])

    def test_one_or_two_missing_old_candles_are_tolerated_without_imputation(self):
        rows = self.rows(2000000000)
        for missing_at in ((100,), (100, 251), (465, 468)):
            with self.subTest(missing_at=missing_at):
                observed = [r for i, r in enumerate(rows) if i not in missing_at]
                result = crash_v2.assess_entry(observed, [], 2000000000)
                self.assertTrue(result["qualified"])
                self.assertEqual(result["candles"], 480-len(missing_at))
                self.assertEqual(result["missing_candles"], len(missing_at))
                self.assertEqual(result["strength_score"]["observed_candles"],
                                 480-len(missing_at))
                self.assertIsNone(result["five_day_context"])
                self.assertEqual(result["last_closed_open_ms"], rows[-1][0])

    def test_three_missing_candles_still_fail_closed(self):
        rows = self.rows(2000000000)
        result = crash_v2.assess_entry(
            [r for i,r in enumerate(rows) if i not in (100, 120, 250)],
            [], 2000000000)
        self.assertEqual(result["status"], "MISSING_5D_HISTORY")
        self.assertFalse(result["qualified"])
        self.assertEqual(result["candles"], 477)

    def test_closed_ws_tail_can_fill_gap(self):
        rows = self.rows(2000000000)
        result = crash_v2.assess_entry(rows[:-1], rows[-1:], 2000000000)
        self.assertTrue(result["qualified"])

    def test_future_candle_never_accepted(self):
        at = 2000000000
        rows = self.rows(at)
        rows[-1][6] = int(at*1000)+900000
        self.assertFalse(crash_v2.assess_entry(rows, [], at)["qualified"])

    def test_recovery_less_than_one_percent(self):
        data = crash_v2.assess_entry(self.rows(2000000000, "1", "1.005"), [], 2000000000)
        self.assertEqual(data["status"], "FILTERED_V2")

    def _delivery_context(self):
        import time
        tree = ast.parse(Path(__file__).with_name("ema.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_crash_send_pending")
        receipt = Mock(return_value={"ok":True, "attempt_at":100, "accepted_at":101,
                                     "message_id":42})
        env = dict(CRASH_V2_SIGNAL_TTL_SECONDS=280,
                   _crash_save_state=Mock(), _crash_append_data=Mock(),
                   _crash_telegram_receipt=receipt, time=time)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "ema.py", "exec"), env)
        return env, receipt

    def _candidate(self):
        record = dict(signal_id="TEST-100", created_at=100, v2_delivery_phase="CANDIDATE",
                      signal_filter_version="five_day_entry_v2_only", v2_status="STRONG_V2",
                      v2_sent=False, tracking_status="WAIT_ENTRY", symbol="TESTUSDT",
                      change_24h=-15, entry="1.000", tp="1.010", last_price="0.995",
                      v2_features=dict(recovery_4h_pct=1.2, rsi14=24, candles=480,
                                       return_5d_pct=-13),
                      failure_shadow=dict(version='tp_nonclosure_entry_gate_v1',
                          flags=dict(weak_volume=False, falling_rsi_ema=False,
                                     repeat_lower_entry=False)))
        return {"history":[record], "symbols":{"TESTUSDT":dict(record)}}, record

    def test_legacy_candidate_not_resent(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        record.pop("v2_delivery_phase")
        env["_crash_send_pending"](state, 105)
        receipt.assert_not_called()

    def test_gtc_submission_once_after_successful_candidate_delivery(self):
        env, receipt = self._delivery_context()
        executor = Mock(); env['_CRASH_NIGHT'] = executor
        state, record = self._candidate()
        env['_crash_send_pending'](state,105)
        env['_crash_send_pending'](state,110)
        executor.submit.assert_called_once()
        submitted, observed = executor.submit.call_args.args
        self.assertTrue(submitted['v2_sent'])
        self.assertEqual(submitted['tracking_status'],'WAIT_ENTRY')
        self.assertEqual(observed,101)

    def test_risk_size_policy_queues_zero_one_two_but_not_three(self):
        from crash_failure_shadow import entry_block_reason, position_fraction
        scenarios=[
            ((False,False,False),True,1.0),
            ((True,False,False),True,1.0),
            ((True,True,False),True,0.5),
            ((True,True,True),False,0.0),
        ]
        for vals,allowed,fraction in scenarios:
            with self.subTest(flags=vals):
                env,_=self._delivery_context()
                executor=Mock()
                env['_CRASH_NIGHT']=executor
                state,record=self._candidate()
                record['failure_shadow']['flags']=dict(
                    zip(('weak_volume','falling_rsi_ema','repeat_lower_entry'),vals))
                self.assertEqual(position_fraction(record),fraction)
                self.assertEqual(entry_block_reason(record) is None,allowed)
                env['_crash_send_pending'](state,105)
                self.assertEqual(executor.submit.call_count,int(allowed))
                self.assertTrue(record['v2_sent'])

    def test_failed_signal_delivery_never_submits_gtc(self):
        env, receipt = self._delivery_context()
        executor=Mock();env['_CRASH_NIGHT']=executor
        receipt.return_value={'ok':False,'error':'NETWORK','ambiguous':True}
        state, record=self._candidate()
        env['_crash_send_pending'](state,105)
        executor.submit.assert_not_called()

    def test_qualified_candidate_alert_before_any_entry(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        env["_crash_send_pending"](state, 105)
        env["_crash_send_pending"](state, 110)
        receipt.assert_called_once()
        self.assertTrue(record["v2_sent"])
        self.assertEqual(record["v2_telegram_message_id"],42)
        self.assertEqual(record["v2_delivery_status"],"SENT")
        self.assertNotIn("entered_at",record)
        self.assertIn("Giriş henüz gerçekleşmedi",receipt.call_args.args[0])

    def test_fractional_strength_shown_and_does_not_block_telegram(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        weights = {
            "dip_stability":(14.3,25), "rsi_momentum":(11.7,20),
            "volume_confirmation":(9.4,20), "ema25_improvement":(8.6,15),
            "atr_tp_capacity":(5.5,10), "selling_slowdown":(3.2,5),
            "btc_context":(2.4,5)
        }
        record["v2_features"]["strength_score"] = {
            "score":55.1, "coverage_pct":100.0,
            "components":{k:{"points":v,"max_points":w} for k,(v,w) in weights.items()}
        }
        env["_crash_send_pending"](state,105)
        message = receipt.call_args.args[0]
        self.assertIn("GÜÇ PUANI: 55.1/100",message)
        self.assertIn("Dip 14.3/25",message)
        self.assertIn("RSI 11.7/20",message)
        self.assertIn("Hacim 9.4/20",message)
        self.assertIn("EMA25 8.6/15",message)
        self.assertTrue(record["v2_sent"])

    def test_stale_signal_never_sent_late(self):
        env, receipt = self._delivery_context()
        state, record = self._candidate()
        env["_crash_send_pending"](state, 381)
        self.assertEqual(record["v2_delivery_status"],"EXPIRED_NOT_SENT")
        receipt.assert_not_called()

    def test_delayed_telegram_allowed_but_old_candidate_never_autobought(self):
        env, receipt = self._delivery_context()
        executor = Mock()
        env['_CRASH_NIGHT'] = executor
        env['log'] = Mock()
        state, record = self._candidate()
        receipt.return_value = {"ok": True, "attempt_at": 379,
                                "accepted_at": 379, "message_id": 43}
        env["_crash_send_pending"](state, 379)  # Candidate age: 279s.
        self.assertEqual(record["v2_delivery_status"], "SENT")
        executor.submit.assert_not_called()

    def test_pending_history_survives_90s_but_expires_after_280s(self):
        tree = ast.parse(Path(__file__).with_name("ema.py").read_text())
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_crash_review_candidate")
        assess = Mock(return_value={"status": "MISSING_5D_HISTORY",
                                    "qualified": False, "candles": 479})
        env = dict(CRASH_V2_SIGNAL_TTL_SECONDS=280,
                   _crash_v2_assess_candidate=assess,
                   _crash_save_state=Mock(), _crash_append_data=Mock())
        exec(compile(ast.Module(body=[fn], type_ignores=[]),
                     "ema.py", "exec"), env)
        record = dict(symbol="TESTUSDT", signal_id="TEST-100", created_at=100,
                      v2_status="PENDING_5D_HISTORY", v2_delivery_status="WAIT_V2")
        state = {"history": [record], "symbols": {"TESTUSDT": dict(record)}}
        env["_crash_review_candidate"](state, record, 379)
        self.assertEqual(record["v2_status"], "PENDING_5D_HISTORY")
        assess.assert_called_once()
        env["_crash_review_candidate"](state, record, 381)
        self.assertEqual(record["v2_reason"], "HISTORY_NOT_READY_IN_TIME")
        self.assertEqual(record["v2_delivery_status"], "EXPIRED_NOT_SENT")

    def test_timeout_marks_unknown_delivery_no_blind_retry(self):
        env, receipt = self._delivery_context()
        receipt.return_value = {"ok":False, "error":"TIMEOUT_DELIVERY_UNKNOWN",
                                "ambiguous":True}
        state, record = self._candidate()
        env["_crash_send_pending"](state, 105)
        env["_crash_send_pending"](state, 140)
        self.assertEqual(record["v2_delivery_status"],"UNKNOWN_DELIVERY")
        self.assertEqual(receipt.call_count,1)


if __name__ == "__main__":
    unittest.main()

