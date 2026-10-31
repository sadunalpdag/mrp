import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from crash_ai import Analyst, MODEL, cost, export_csv, format_review, snapshot, validate


def record():
    return {'signal_id': 'TESTUSDT-123', 'symbol': 'TESTUSDT', 'created_at': 100,
            'v2_sent': True, 'entry': 10, 'tp': 10.1, 'change_24h': -20,
            'v2_features': {'rsi14': 24, 'candles': 480, 'recovery_4h_pct': 2,
                            'strength_score': {'score': 99, 'metrics': {'rsi_delta_1h': 2}},
                            'five_day_context': {'return_24h_pct': -20}},
            'derivatives_shadow': {'signal_id': 'TESTUSDT-123', 'as_of': 100,
                                   'status': 'COMPLETE', 'oi_change_4h': -3}}


def good():
    return {'ai_score': 52, 'data_quality': 'MEDIUM', 'positives': ['RSI iyileşiyor.'],
            'risks': ['Düşüş sürüyor.'], 'missing_data': ['spot_cvd_4h'], 'summary': 'Belirsiz toparlanma.'}


def response(value=None, status='completed'):
    return SimpleNamespace(status_code=200, json=lambda: {
        'id': 'response1', 'status': status, 'usage': {'input_tokens': 2000, 'output_tokens': 500},
        'output': [{'type': 'message', 'content': [{'type': 'output_text',
                    'text': json.dumps(good() if value is None else value)}]}]})


class AITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 200
        self.post = Mock(return_value=response())

    def analyst(self, **env):
        return Analyst(self.tmp.name, log=lambda x: None, clock=lambda: self.now,
                       post=self.post, env={'OPENAI_API_KEY': 'test-secret', **env})

    def test_allowlist_excludes_scores_secrets_and_future_outcomes(self):
        r = record()
        r.update(secret='DO_NOT_SEND', last_price=1000, finished_at=999, tracking_status='TP')
        r['v2_features']['strength_score']['components'] = {'score': 999}
        data = snapshot(r)
        text = json.dumps(data)
        for bad in ('score', 'DO_NOT_SEND', 'finished_at', 'tracking_status', 'last_price'):
            self.assertNotIn(bad, text)
        self.assertIsNone(data['derivatives']['oi_change_4h'])
        self.assertEqual(data['metrics']['rsi_delta_1h'], 2)

    def test_no_key_no_request(self):
        a = self.analyst(OPENAI_API_KEY='')
        self.assertFalse(a.submit(record()))
        self.assertFalse(a.step())
        self.post.assert_not_called()

    def test_only_delivered_signal(self):
        r = record(); r['v2_sent'] = False
        self.assertFalse(self.analyst().submit(r))

    def test_complete_once_and_restart_no_paid_repeat(self):
        a = self.analyst(); r = record(); before = copy.deepcopy(r)
        self.assertTrue(a.submit(r)); self.assertFalse(a.submit(r))
        self.assertTrue(a.step())
        d = a.results.get_nowait()
        self.assertEqual(d['status'], 'COMPLETE')
        self.assertEqual(d['estimated_usd'], .00375)
        self.assertEqual(r, before)
        b = self.analyst()
        self.assertFalse(b.submit(r)); self.assertFalse(b.step())
        self.assertEqual(self.post.call_count, 1)
        payload = self.post.call_args.kwargs['json']
        self.assertFalse(payload['store'])
        self.assertNotIn('tools', payload)
        self.assertEqual(payload['max_output_tokens'], 1000)
        self.assertNotIn('test-secret', Path(self.tmp.name, 'ai_events.jsonl').read_text())

    def test_wait_for_derivatives_or_timeout_with_missing_not_zero(self):
        a = self.analyst(); r = record(); r['derivatives_shadow']['status'] = 'PENDING'
        a.submit(r); self.assertFalse(a.step())
        r['entry'] = 99  # caller mutations must not change snapshot
        self.now += 121
        self.assertTrue(a.step())
        data = json.loads(self.post.call_args.kwargs['json']['input'])
        self.assertIsNone(data['derivatives']['oi_change_4h'])
        self.assertEqual(data['market']['entry'], 10)

    def test_future_derivatives_rejected(self):
        a = self.analyst(); r=record(); r['derivatives_shadow']['status']='PENDING'
        a.submit(r)
        a.update_derivatives({'signal_id':r['signal_id'], 'as_of':101, 'status':'COMPLETE', 'oi_change_4h':4})
        self.assertFalse(a.step())
        a.update_derivatives({'signal_id':r['signal_id'], 'as_of':100, 'status':'COMPLETE', 'oi_change_4h':-2})
        self.assertTrue(a.step())

    def test_timeout_no_blind_retry(self):
        self.post.side_effect = TimeoutError('secret in untrusted exception')
        a = self.analyst(); a.submit(record()); a.step()
        self.assertIsNone(a.results.get_nowait()['assessment'])
        b = self.analyst(); self.assertFalse(b.submit(record()))
        self.assertNotIn('secret in', Path(self.tmp.name, 'ai_events.jsonl').read_text())
        self.assertEqual(self.post.call_count, 1)

    def test_incomplete_and_invalid_scores_not_reported_as_zero(self):
        for result in (response(status='incomplete'), response(good() | {'ai_score': 101})):
            with tempfile.TemporaryDirectory() as directory:
                a = Analyst(directory, log=lambda x: None, clock=lambda: 200,
                            post=Mock(return_value=result), env={'OPENAI_API_KEY':'x'})
                a.submit(record()); a.step(); d=a.results.get_nowait()
                self.assertIsNone(d['assessment'])
                self.assertIn('alınamadı', format_review(d))
                self.assertNotIn('0/100', format_review(d))

    def test_refusal(self):
        self.post.return_value = SimpleNamespace(status_code=200, json=lambda: {
            'status':'completed', 'output':[{'type':'message','content':[{'type':'refusal','refusal':'x'}]}]})
        a=self.analyst(); a.submit(record()); a.step()
        self.assertIsNone(a.results.get_nowait()['assessment'])

    def test_daily_cap_and_next_day(self):
        a=self.analyst(CRASH_AI_MAX_DAILY='1'); a.submit(record()); a.step()
        self.now += 11
        r=record(); r['signal_id']='OTHER-2'; a.submit(r); self.now += 121; a.step()
        self.assertEqual(self.post.call_count, 1)
        self.now += 86400
        r['signal_id']='OTHER-3'; a.submit(r); self.now += 121; a.step()
        self.assertEqual(self.post.call_count, 2)

    def test_auth_error_disables_and_rate_limit_waits(self):
        self.post.return_value=SimpleNamespace(status_code=401)
        a=self.analyst(); a.submit(record()); a.step()
        self.assertFalse(a.enabled)
        self.assertEqual(a.results.get_nowait()['status'], 'HTTP_401')
        with tempfile.TemporaryDirectory() as d:
            a=Analyst(d, log=lambda x:None, clock=lambda:200,
                      post=lambda *args, **kwargs: SimpleNamespace(status_code=429), env={'OPENAI_API_KEY':'x'})
            a.submit(record()); a.step(); self.assertEqual(a.next_call,500)

    def test_corrupt_journal_disables(self):
        Path(self.tmp.name, 'ai_events.jsonl').write_text('{broken')
        a=self.analyst(); self.assertFalse(a.enabled)

    def test_cost_and_csv(self):
        self.assertIsNone(cost({}, MODEL)); self.assertIsNone(cost({'input_tokens':1,'output_tokens':1}, 'unknown'))
        a=self.analyst(); r=record(); a.submit(r); a.step(); r['ai_review']=a.results.get_nowait()
        self.assertIn('52', export_csv([r])); self.assertIn('test-secret', a.key)
        self.assertNotIn('test-secret', export_csv([r]))
        with self.assertRaises(ValueError):
            validate(good() | {'ai_score': True})

    def test_signal_integration_exception_does_not_block_executor(self):
        from test_crash_v2 import CrashV2Tests
        # Reuse the production AST harness without importing the running bot.
        helper = CrashV2Tests()
        env, receipt = helper._delivery_context()
        state, candidate = helper._candidate()
        env['log'] = Mock()
        env['_CRASH_AI'] = SimpleNamespace(submit=Mock(side_effect=ValueError('AI unavailable')))
        executor = SimpleNamespace(enabled=SimpleNamespace(is_set=lambda: True), submit=Mock())
        env['_CRASH_NIGHT']=executor
        env['_crash_send_pending'](state,105)
        self.assertTrue(state['history'][0]['v2_sent'])
        executor.submit.assert_called_once()
