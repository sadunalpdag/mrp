"""Regression tests for read-only 15m source comparison."""
import tempfile
import time
import unittest
from unittest.mock import Mock
from crash_candle_trace import STEP, inspect_candles, CandleAudit


def rows(times):
    return [[t, '1.0', '1.2', '0.9', '1.1', '50', t + STEP - 1]
            for t in times]


class CandleTraceTests(unittest.TestCase):
    def test_latest_ws_only_and_one_old_gap(self):
        end = 901 * STEP
        all_slots = list(range(end - 480 * STEP, end, STEP))
        history = rows(all_slots[:-2])
        ws = rows([end - STEP])
        result = inspect_candles(end / 1000 + 30, ws, history,
                                 {'open_ms': end - STEP})
        self.assertTrue(result['ws_cache_latest'])
        self.assertFalse(result['history_latest'])
        self.assertTrue(result['engine_latest'])
        self.assertEqual(result['engine_count'], 479)
        self.assertEqual(result['missing_count'], 1)
        self.assertTrue(result['actual_ws_event_latest'])

    def test_latest_missing_in_both_caches_is_real_gap(self):
        end = 901 * STEP
        history = rows(range(end - 480 * STEP, end - STEP, STEP))
        result = inspect_candles(end / 1000 + 40, [], history)
        self.assertFalse(result['engine_latest'])
        self.assertIsNone(result['actual_ws_event_latest'])
        self.assertEqual(result['engine_count'], 479)

    def test_history_cache_can_serve_signal_without_new_ws_event(self):
        end = 901 * STEP
        history = rows(range(end - 480 * STEP, end, STEP))
        result = inspect_candles(end / 1000 + 40, [], history)
        self.assertTrue(result['history_latest'])
        self.assertTrue(result['engine_latest'])
        self.assertFalse(result['ws_cache_latest'])
        self.assertEqual(result['engine_count'], 480)

    def test_audit_does_not_request_network_or_write_history(self):
        end = 901 * STEP
        log = Mock()
        with tempfile.TemporaryDirectory() as directory:
            cache = Mock()
            cache.candles.return_value = []
            cache.last_ws_observation.return_value = None
            audit = CandleAudit(directory, log)
            audit.tick([{'symbol': 'XUSDT', 'priceChangePercent': '-20'}],
                       cache, now=end / 1000 + 35)
            for _ in range(50):
                if not audit.lock.locked():
                    break
                time.sleep(.01)
            self.assertTrue(any('rest_requests=0' in str(call)
                                for call in log.call_args_list))
