"""Offline proportional Crash Bounce research score regression checks."""
import math
import unittest

from crash_score_v3 import VERSION, WEIGHTS, score_candidate
from crash_v2 import assess_entry


class ProportionalScoreTests(unittest.TestCase):
    def rows(self, at=2_000_000_000, last4_boost=0.0, no_volume=False):
        end = int(at * 1000) // 900000 * 900000
        result = []
        for i, t in enumerate(range(end-480*900000, end, 900000)):
            base = 1.5 - i * 0.0006 + math.sin(i * 0.6) * 0.003
            p = base + (last4_boost*(i-475)/4 if i >= 476 else 0.0)
            op = max(0.1, p - .002*math.sin(i*.7))
            close = max(0.1, p + .003*math.sin(i*.9))
            high = max(op,close) + .012
            low = min(op,close) - .018
            volume = 0.0 if no_volume else 100.0 + (i % 11)*12
            result.append([t,str(op),str(high),str(low),str(close),
                           str(volume),t+899999])
        return result

    def to_ohlcv(self, rows):
        return [tuple(float(r[j]) for j in (1,2,3,4,5)) for r in rows]

    def test_fractional_points_sum_and_ranges(self):
        value=score_candidate(self.to_ohlcv(self.rows()),btc_change_24h=-1.3)
        self.assertEqual(value['version'],VERSION)
        self.assertEqual(value['coverage_pct'],100.0)
        self.assertTrue(0 <= value['score'] <= 100)
        self.assertEqual(set(value['components']),set(WEIGHTS))
        for key, component in value['components'].items():
            self.assertEqual(component['max_points'], WEIGHTS[key])
            self.assertIsNotNone(component['points'])
            self.assertTrue(0 <= component['points'] <= WEIGHTS[key])
        self.assertAlmostEqual(sum(i['points'] for i in value['components'].values()),
                               value['raw_points'],places=2)
        self.assertTrue(any(p['points'] not in (0, p['max_points'])
                            for p in value['components'].values()))

    def test_btc_missing_marked_not_assumed(self):
        rows=self.to_ohlcv(self.rows())
        missing=score_candidate(rows)
        available=score_candidate(rows,btc_change_24h=0)
        self.assertEqual(missing['coverage_pct'],95.0)
        self.assertIsNone(missing['components']['btc_context']['points'])
        self.assertEqual(available['coverage_pct'],100.0)
        self.assertIsNotNone(available['components']['btc_context']['points'])

    def test_absent_volume_is_not_fake_buy_pressure(self):
        value=score_candidate(self.to_ohlcv(self.rows(no_volume=True)))
        self.assertIsNone(value['components']['volume_confirmation']['points'])
        self.assertEqual(value['coverage_pct'],75.0)

    def test_components_change_with_market_conditions(self):
        early=score_candidate(self.to_ohlcv(self.rows(last4_boost=0.0)),btc_change_24h=-2.5)
        bounce=score_candidate(self.to_ohlcv(self.rows(last4_boost=0.040)),btc_change_24h=1.0)
        self.assertNotEqual(early['score'],bounce['score'])
        self.assertGreater(bounce['components']['btc_context']['points'],
                           early['components']['btc_context']['points'])

    def test_v2_integrates_score_and_preserves_existing_gate(self):
        rows=self.rows()
        x=assess_entry(rows,[],2_000_000_000,btc_change_24h=-0.3)
        self.assertEqual(x['candles'],480)
        self.assertIn('strength_score',x)
        self.assertEqual(x['strength_score']['version'],VERSION)
        self.assertIsInstance(x['qualified'],bool)
        damaged=assess_entry(rows[:-1],[],2_000_000_000)
        self.assertFalse(damaged['qualified'])
        self.assertNotIn('strength_score',damaged)

    def test_requires_full_candle_window(self):
        with self.assertRaises(ValueError):
            score_candidate(self.to_ohlcv(self.rows()[:-1]))


if __name__=="__main__":
    unittest.main()
