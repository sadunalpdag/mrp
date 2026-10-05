"""Offline crash-only startup and REST regression tests."""
import ast
import json
import math
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class RateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 2000000000.
        def sleep(seconds):
            self.now += seconds
        tree = ast.parse(Path(__file__).with_name('ema.py').read_text())
        names = {'main', '_crash_public_get', '_crash_set_cooldown', '_crash_wait_until'}
        subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        self.g = dict(time=SimpleNamespace(time=lambda:self.now, sleep=sleep), math=math,
                      json=json, os=os, re=re, SAVE_LOCK=threading.Lock(), log=Mock(),
                      CRASH_BOUNCE_REST_STATE_FILE=self.tmp.name+'/rest.json',
                      BINANCE_FAPI='https://example.invalid', requests=Mock(),
                      BinanceRateLimiter=SimpleNamespace(is_banned=lambda:False, set_ban=Mock()),
                      _crash_cooldown_until=0., _crash_next_request=0.,
                      _crash_weight_window=-1, _crash_weight_used=0,
                      _crash_ip_weight_limit=2400, _crash_backoff=0,
                      _crash_bounce_worker=Mock())
        exec(compile(subset,'ema.py','exec'), self.g)

    def response(self, status=200, headers=None, data=None):
        r=Mock(status_code=status, headers=headers or {})
        r.json.return_value=[] if data is None else data
        if status >= 400:
            r.raise_for_status.side_effect=RuntimeError(str(status))
        self.g['requests'].get.return_value=r
        return r

    def test_main_only_starts_crash_and_respects_persisted_ban(self):
        Path(self.g['CRASH_BOUNCE_REST_STATE_FILE']).write_text(json.dumps({'until':self.now+800}))
        before=self.now
        self.g['main']()
        self.assertEqual(self.now, before+800)
        self.g['_crash_bounce_worker'].assert_called_once()
        self.g['requests'].get.assert_not_called()

    def test_429_blocks_every_request_and_persists_retry_after(self):
        self.response(429, {'Retry-After':'900'})
        before=self.now
        with self.assertRaises(RuntimeError): self.g['_crash_public_get']('/fapi/v1/ticker/24hr')
        self.assertGreaterEqual(self.g['_crash_cooldown_until'], before+905)
        with self.assertRaises(RuntimeError): self.g['_crash_public_get']('/fapi/v1/klines')
        self.assertEqual(self.g['requests'].get.call_count,1)
        self.assertEqual(json.loads(Path(self.g['CRASH_BOUNCE_REST_STATE_FILE']).read_text())['until'],self.g['_crash_cooldown_until'])

    def test_418_honors_ban_timestamp_and_invalid_header(self):
        before=self.now
        self.response(418, {'Retry-After':'invalid'}, {'msg':f'IP banned until {int((before+7200)*1000)}'})
        with self.assertRaises(RuntimeError): self.g['_crash_public_get']('/fapi/v1/ticker/24hr')
        self.assertGreaterEqual(self.g['_crash_cooldown_until'],before+7205)

    def test_weight_budget_spacing_and_ip_header(self):
        self.response()
        before=self.now
        for _ in range(4): self.g['_crash_public_get']('/fapi/v1/ticker/24hr')
        self.assertGreater(self.now,before+10)
        self.response(headers={'X-MBX-USED-WEIGHT-1M':'1300'})
        self.g['_crash_public_get']('/fapi/v1/klines')
        self.assertGreater(self.g['_crash_cooldown_until'],self.now)

    def test_endpoint_allowlist_and_cooldown_corruption_fail_closed(self):
        with self.assertRaises(ValueError): self.g['_crash_public_get']('/fapi/v1/ticker/price')
        Path(self.g['CRASH_BOUNCE_REST_STATE_FILE']).write_text('broken')
        with self.assertRaises(ValueError): self.g['main']()
        self.g['_crash_bounce_worker'].assert_not_called()
        self.g['requests'].get.assert_not_called()


if __name__=='__main__': unittest.main()
