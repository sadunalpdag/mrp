"""Offline WebSocket V2 outcome tests; synthetic prints; no network."""
import unittest
from crash_live_v2 import advance_trade, MODEL


class CrashLiveV2Tests(unittest.TestCase):
    def setUp(self):
        self.r = {'tracking_model':MODEL, 'tracking_status':'WAIT_ENTRY', 'created_at':100,
                  'entry':'1', 'tp':'1.01', 'reference_low':'0.95',
                  'previous_watch_price':.98, 'samples':0, 'coverage_gaps':0,
                  'max_gap_seconds':0}

    def test_entry_then_tp_not_same_event(self):
        self.assertEqual(advance_trade(self.r,1.005,101,1),'OPEN')
        self.assertIsNone(advance_trade(self.r,1.03,101,1))
        self.assertEqual(advance_trade(self.r,1.015,102,2),'TP')
        self.assertEqual(self.r['entry_trade_id'],1)
        self.assertEqual(self.r['tp_trade_id'],2)

    def test_gap_above_tp_is_missed_not_tp(self):
        self.assertEqual(advance_trade(self.r,1.02,132,1),'MISSED_SIGNAL')
        self.assertEqual(self.r['v2_reason'],'ENTRY_ABOVE_TP')
        self.assertNotIn('entered_at',self.r)

    def test_jump_above_tp_is_missed(self):
        self.assertEqual(advance_trade(self.r,1.02,101,1),'MISSED_SIGNAL')

    def test_low_invalidates(self):
        self.assertEqual(advance_trade(self.r,.94,101,1),'INVALIDATED_BEFORE_ENTRY')

    def test_unobserved_gap_cannot_assume_fill(self):
        self.assertIsNone(advance_trade(self.r,1.005,140,1))
        self.assertNotIn('entered_at',self.r)

    def test_replayed_trade_id_ignored(self):
        self.assertIsNone(advance_trade(self.r,.99,101,1))
        self.assertEqual(advance_trade(self.r,1.005,102,2),'OPEN')
        self.assertIsNone(advance_trade(self.r,1.03,104,2))
        self.assertEqual(self.r['tracking_status'],'OPEN')

if __name__ == '__main__':
    unittest.main()
