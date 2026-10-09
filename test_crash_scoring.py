import ast
import math
import unittest
from pathlib import Path
from unittest.mock import Mock
import test_crash_bounce

class ScoreTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path(__file__).with_name('ema.py').read_text())
        names = {'_crash_signal_score', '_crash_score_text', '_crash_apply_filter_migration'}
        self.g = {'math': math, 'CRASH_BOUNCE_FILTER_VERSION': 'rsi_ema_score_v2', '_crash_save_state': Mock()}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],type_ignores=[]),'ema.py','exec'),self.g)

    def test_scores_and_boundaries(self):
        for rsi,slope,score in [(24,-.1,100),(25,-.1,50),(24,0,50),(25,0,0),(70,.3,0)]:
            with self.subTest(rsi=rsi,slope=slope):
                self.assertEqual(self.g['_crash_signal_score']({'rsi14_wilder':rsi,'ema25_slope_pct':slope})['score'],score)

    def test_missing_and_invalid_indicators(self):
        for bad in [None,True,'24',float('nan'),float('inf'),-1,101]:
            value=self.g['_crash_signal_score']({'rsi14_wilder':bad,'ema25_slope_pct':-.1})
            self.assertEqual(value['score'],50)
            self.assertFalse(value['complete'])
        self.assertIn('veri yok',self.g['_crash_score_text']({}))

    def test_migration_preserves_pending_open_and_historical_filtered(self):
        records=[dict(symbol=str(i),signal_id=str(i),tracking_status=status,features={},sent=False) for i,status in enumerate(['WAIT_ENTRY','OPEN','FILTERED_OUT'])]
        state={'history':records,'symbols':{r['symbol']:dict(r) for r in records},'report':{'pending':{}}}
        self.g['_crash_apply_filter_migration'](state,123)
        self.assertEqual([r['tracking_status'] for r in records],['WAIT_ENTRY','OPEN','FILTERED_OUT'])
        self.assertTrue(all(not r['sent'] for r in state['symbols'].values()))
        self.assertNotIn('pending',state['report'])

    def test_all_scores_create_and_deliver_signals(self):
        for features,score in [({'rsi14_wilder':24,'ema25_slope_pct':-.1,'ema25':1.02},100),({'rsi14_wilder':60,'ema25_slope_pct':-.1},50),({'rsi14_wilder':60,'ema25_slope_pct':.1},0),({},0)]:
            fixture=test_crash_bounce.CrashBounceTests()
            fixture.setUp()
            fixture.g['_crash_features'].return_value=features
            state=fixture.revision_fixture()
            self.assertEqual(len(state['history']),1)
            self.assertEqual(state['history'][0]['condition_score']['score'],score)
            self.assertTrue(state['symbols']['AUSDT']['sent'])
            message=fixture.g['tg_send'].call_args.args[0]
            for text in [f'{score}/100','RSI14','EMA25 eğimi','başarı olasılığı değildir']:
                self.assertIn(text,message)

if __name__=='__main__':
    unittest.main()
