"""Offline context alignment, missing-data and delivery regression tests."""
import unittest
from crash_context import assess_context, summarize_context
from crash_v2 import assess_entry
import test_crash_v2


class ContextTests(unittest.TestCase):
    def test_quote_volume_uses_four_day_daily_baseline(self):
        candles = [(10, 12, 8, 10, 100)] * 480
        score = {'metrics': {'green_volume_share_1h': .1, 'rsi_delta_1h': -5}}
        context = assess_context(candles, [100]*384 + [200]*96, score)
        self.assertEqual(context['quote_volume_ratio_24h_vs_previous_4d'], 2)
        self.assertEqual(context['selling_pressure_index'], 45)
        self.assertTrue(context['weak_green_and_falling_rsi'])
        self.assertAlmostEqual(context['drawdown_from_5d_high_pct'], -16.6667)

    def test_missing_volume_does_not_block_or_invent_volume(self):
        at = 2000000000
        rows = test_crash_v2.CrashV2Tests().rows(at)
        result = assess_entry(rows, [], at)
        self.assertTrue(result['qualified'])
        self.assertIsNone(result['five_day_context']['quote_volume_ratio_24h_vs_previous_4d'])

    def test_future_rows_cannot_change_context(self):
        at = 2000000000
        rows = test_crash_v2.CrashV2Tests().rows(at)
        for row in rows:
            row.append('100')
        base = assess_entry(rows, [], at)
        future = [rows[-1][0]+900000, '1', '1000', '.1', '999', '1', rows[-1][6]+900000, '99999999']
        self.assertEqual(base, assess_entry(rows, [future], at))

    def test_missing_and_unknown_are_not_unflagged(self):
        result = summarize_context([
            {'tracking_status': 'OPEN', 'entered_at': 1},
            {'tracking_status': 'TP', 'entered_at': 1, 'v2_features': {'five_day_context': {
                'version': 'five_day_context_v1', 'weak_green_and_falling_rsi': True}}}])
        self.assertEqual(result['groups']['unknown']['signals'], 1)
        self.assertEqual(result['groups']['flagged']['statuses'], {'TP': 1})
        self.assertEqual(result['groups']['not_flagged']['signals'], 0)

    def test_flag_does_not_block_telegram(self):
        helper = test_crash_v2.CrashV2Tests()
        env, receipt = helper._delivery_context()
        state, record = helper._candidate()
        record['v2_features']['five_day_context'] = {
            'weak_green_and_falling_rsi': True, 'selling_pressure_index': 90,
            'regime': 'CONTINUING_DECLINE'}
        env['_crash_send_pending'](state, 105)
        self.assertTrue(record['v2_sent'])
        self.assertIn('Zayıf yeşil hacim', receipt.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
