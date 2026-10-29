import unittest
from crash_data_diagnostics import candle_diagnostic


class DataDiagnosticsTests(unittest.TestCase):
    def test_empty_cache(self):
        text = candle_diagnostic([], 2000000000, '1', '.001')
        self.assertIn('EKSIK_MUM', text)
        self.assertIn('kapali_mum=0/16', text)
        self.assertIn('son_mum_UTC=yok', text)

    def test_stale_and_gap_are_reported_together(self):
        now = 2000000000
        rows = [[(now-4000-(16-i)*900)*1000,'1','1','1','1','1',
                 (now-4000-(15-i)*900)*1000-1] for i in range(16)]
        rows[3][0] += 1000
        text = candle_diagnostic(rows, now, '1', '.001')
        self.assertIn('ESKI_MUM', text)
        self.assertIn('MUM_ZAMAN_BOSLUGU', text)
        self.assertNotIn('EKSIK_MUM', text)

    def test_open_and_malformed_candles_not_counted(self):
        text = candle_diagnostic([[0,1,1,1,1,1,2000000000001], [1]],2000000000,'1','.001')
        self.assertIn('kapali_mum=0/16',text)
        self.assertIn('BOZUK_MUM',text)
