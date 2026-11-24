"""Offline safety checks for the combined Bull Flag/Pennant observer."""
import unittest
from unittest.mock import patch
import bull_pattern_research as bp


def row(kind="Bull Pennant", status="forming", position=.08):
    return {"id":"provider-123","exchange":"binance-futures",
            "tf":"4h","symbol":"BTCUSDT",
            "item":{"type":kind,"status":status,
                    "upper":{"price":105},"lower":{"price":100},
                    "flag":{"position":position,"poleStart":{"time":111},
                            "poleEnd":{"time":222}}}}


class PatternTests(unittest.TestCase):
    def test_recognize_both_patterns(self):
        self.assertEqual(bp.extract(row("Bull Flag"),"4h")[1],"bull-flag")
        self.assertEqual(bp.extract(row("Bull Pennant"),"4h")[1],"bull-pennant")

    def test_status_and_support_position(self):
        got=bp.extract(row(status="near",position=.05),"4h")
        self.assertEqual((got[2],got[3],got[4],got[5]),("near",100,105,.05))

    def test_dedup_stable_across_updates(self):
        a=row()
        b=row(status="broken",position=.9)
        self.assertEqual(bp.identity(a,a["item"],"4h"),
                         bp.identity(b,b["item"],"4h"))

    def test_different_pattern_is_different_case(self):
        a,b=row("Bull Flag"),row("Bull Pennant")
        self.assertNotEqual(bp.identity(a,a["item"],"4h"),
                            bp.identity(b,b["item"],"4h"))

    def test_unstable_identifier_is_rejected(self):
        a=row();a.pop("id");a["item"]["flag"]={}
        self.assertIsNone(bp.identity(a,a["item"],"4h"))

    def test_tp_is_only_after_entry(self):
        bars=[[0,100,102,99,101,1],[900000,101,103,100,102,1]]
        self.assertEqual(bp._outcome(bars,100,101,900000),"TP_1_PERCENT")
        self.assertEqual(bp._outcome(bars,100,105,900000),"NO_TP")
        self.assertEqual(bp._outcome(bars,None,None,None),"NO_ENTRY")

    def test_paper_module_has_no_broker_import(self):
        import inspect
        source=inspect.getsource(bp)
        for forbidden in ("from binance", "import binance", "place_order(", "futures_create_order("):
            self.assertNotIn(forbidden,source)


if __name__=="__main__":
    unittest.main()
