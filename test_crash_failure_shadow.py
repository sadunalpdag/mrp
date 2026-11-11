"""Offline prospective TP-risk label and automatic BUY gate regression tests."""
import csv
import io
import tempfile
import unittest
from unittest.mock import Mock

from crash_failure_shadow import (
    VERSION, assess, entry_block_reason, position_fraction, export_csv, format_warning, summarize,
)
from crash_night import NightExecutor
from test_crash_services import FakeBroker, timestamp


def candidate(symbol="AUSDT", at=1_000_000.0, entry="10",
              green=0.50, volume=2.0, rsi=1.0, ema=0.04, atr=2.0):
    return {
        "symbol": symbol, "signal_id": symbol + ":" + str(at),
        "created_at": at, "entry": entry, "tp": "10.10",
        "v2_status": "STRONG_V2", "v2_sent": True,
        "tracking_status": "WAIT_ENTRY",
        "tracking_model": "ws_aggtrade_upcross_v3",
        "v2_features": {"strength_score": {
            "score": 45.0, "metrics": {
                "green_volume_share_1h": green,
                "volume_ratio_vs_prior_12h": volume,
                "rsi_delta_1h": rsi,
                "ema25_slope_change_1h_pp": ema,
                "atr14_pct": atr,
            }
        }}
    }


class NonclosureLabelsTests(unittest.TestCase):
    def risk(self, record, earlier=()):
        shadow = assess(record, earlier)
        record["failure_shadow"] = shadow
        return shadow

    def test_clean_signal_permitted(self):
        r = candidate()
        shadow = self.risk(r)
        self.assertEqual(shadow["flag_count"], 0)
        self.assertEqual(shadow["known_flags"], 4)
        self.assertFalse(shadow["blocks_live_entry"])
        self.assertIsNone(entry_block_reason(r))

    def test_weak_green_share_and_volume_uses_full_size(self):
        r = candidate(green=.17, volume=1.27, rsi=2, ema=.02)
        shadow = self.risk(r)
        self.assertEqual(shadow["flags"], {
            "weak_volume": True, "falling_rsi_ema": False,
            "repeat_lower_entry": False, "low_atr_15m": False})
        self.assertEqual(shadow["flag_count"], 1)
        self.assertIsNone(entry_block_reason(r))
        self.assertEqual(position_fraction(r), 1.0)
        self.assertEqual(shadow["position_fraction"], 1.0)
        self.assertIn("%100", format_warning(shadow))

    def test_falling_rsi_and_ema_uses_full_size(self):
        r = candidate(rsi=-6, ema=-.11)
        shadow = self.risk(r)
        self.assertTrue(shadow["flags"]["falling_rsi_ema"])
        self.assertIsNone(entry_block_reason(r))
        self.assertEqual(position_fraction(r), 1.0)

    def test_repeat_lower_prior_strong_signal_only(self):
        older = candidate(symbol="AUSDT", at=1_000_000-3600, entry="11")
        r = candidate()
        shadow = self.risk(r, [older])
        self.assertTrue(shadow["flags"]["repeat_lower_entry"])
        self.assertLess(shadow["metrics"]["repeat_prior_entry_drop_pct"], -2)
        self.assertIsNone(entry_block_reason(r))
        self.assertEqual(position_fraction(r), 1.0)
        older["created_at"] = 1_000_001
        self.assertFalse(assess(r, [older])["flags"]["repeat_lower_entry"])

    def test_missing_data_is_unknown_not_clean(self):
        r = candidate(volume=None)
        shadow = self.risk(r)
        self.assertIsNone(shadow["flags"]["weak_volume"])
        self.assertEqual(shadow["known_flags"], 3)
        self.assertEqual(entry_block_reason(r), "RISK_METRICS_INCOMPLETE")
        self.assertIn("EKSİK VERİ", format_warning(shadow))
        self.assertEqual(entry_block_reason(candidate()),
                         "RISK_ASSESSMENT_MISSING")

    def test_no_future_leak_and_outcomes_have_separate_counts(self):
        r = candidate()
        self.risk(r)
        r.update(tracking_status="OPEN", entered_at=1_000_010)
        tp = candidate(symbol="BUSDT")
        self.risk(tp)
        tp.update(tracking_status="TP", entered_at=1_000_010)
        timeout = candidate(symbol="CUSDT", green=.1, volume=1.0)
        self.risk(timeout)
        timeout.update(tracking_status="TIMEOUT", entered_at=1_000_020)
        result = summarize([r, tp, timeout])
        self.assertEqual(result["groups"]["all"]["signals"], 3)
        self.assertEqual(result["groups"]["all"]["tp"], 1)
        self.assertEqual(result["groups"]["all"]["timeout"], 1)
        self.assertEqual(result["groups"]["all"]["open_unresolved"], 1)
        self.assertEqual(result["groups"]["weak_volume"]["timeout"], 1)
        rows = list(csv.DictReader(io.StringIO(export_csv([r, tp, timeout]))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["weak_volume"], "True")

    def test_two_risks_half_sized_and_four_risks_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            broker = FakeBroker()
            clock = timestamp("2026-11-03T20:00:00")
            executor = NightExecutor(directory, broker, Mock(return_value={"ok":True}),
                                     Mock(), lambda:clock, enabled=True, max_daily=20)
            self.addCleanup(executor.guard.close)
            older = candidate(symbol="AUSDT", at=clock-3600, entry="11")
            two = candidate(green=.1, volume=1.1, rsi=-6, ema=-.11)
            two.update(created_at=clock-1, signal_id="two-risks")
            shadow = self.risk(two)
            self.assertEqual(shadow["flag_count"],2)
            self.assertEqual(position_fraction(two),.5)
            self.assertIsNone(entry_block_reason(two))
            self.assertIn("%50",format_warning(shadow))
            executor.enter(two,clock)
            self.assertEqual(len(broker.buys),1)
            self.assertEqual(executor.daily_count(),1)
            order=executor.state["orders"]["two-risks"]
            self.assertEqual(order["risk_flag_count"],2)
            self.assertEqual(order["risk_position_fraction"],.5)
            self.assertEqual(order["risk_configured_usdt"],"500")
            self.assertEqual(order["risk_target_usdt"],"250.0")
            self.assertLessEqual(float(order["risk_effective_notional_usdt"]),250)

        with tempfile.TemporaryDirectory() as directory:
            broker = FakeBroker()
            clock = timestamp("2026-11-03T20:00:00")
            executor = NightExecutor(directory, broker, Mock(return_value={"ok":True}),
                                     Mock(), lambda:clock, enabled=True, max_daily=20)
            self.addCleanup(executor.guard.close)
            three = candidate(green=.1, volume=1.1, rsi=-6, ema=-.11, atr=1.0)
            three.update(created_at=clock-1, signal_id="three-risks")
            shadow=self.risk(three,[older])
            self.assertEqual(shadow["flag_count"],4)
            self.assertEqual(shadow["position_fraction"],0)
            self.assertEqual(entry_block_reason(three),"RISK_4_OF_4")
            self.assertIn("YAPILMAYACAK",format_warning(shadow))
            executor.enter(three,clock)
            self.assertEqual(len(broker.buys),0)
            self.assertEqual(executor.daily_count(),0)

    def test_three_risks_trades_thirty_percent_and_half_percent_tp(self):
        with tempfile.TemporaryDirectory() as directory:
            broker = FakeBroker()
            clock = timestamp("2026-11-03T20:00:00")
            executor = NightExecutor(directory, broker, Mock(return_value={"ok": True}),
                                     Mock(), lambda: clock, enabled=True, max_daily=20)
            self.addCleanup(executor.guard.close)
            r = candidate(green=.1, volume=1.1, rsi=-6, ema=-.11, atr=1.0)
            r.update(created_at=clock-1, signal_id="three-risks-thirty")
            shadow = self.risk(r)
            self.assertEqual(shadow["flag_count"], 3)
            self.assertEqual(shadow["known_flags"], 4)
            self.assertEqual(shadow["position_fraction"], .3)
            self.assertEqual(position_fraction(r), .3)
            self.assertIn("%30", format_warning(shadow))
            self.assertIsNone(entry_block_reason(r))
            executor.enter(r, clock)
            self.assertEqual(len(broker.buys), 1)
            order = executor.state["orders"]["three-risks-thirty"]
            self.assertEqual(order["risk_target_usdt"], "150.0")
            self.assertEqual(order["risk_flag_count"], 3)
            self.assertEqual(order["risk_position_fraction"], .3)
            self.assertEqual(order["risk_tp_pct"], "0.005")
            self.assertTrue(order["risk_flags"]["low_atr_15m"])

    def test_atr_missing_and_legacy_three_flags_fail_closed(self):
        r = candidate(atr=None)
        shadow = self.risk(r)
        self.assertEqual(shadow["known_flags"], 3)
        self.assertIsNone(shadow["flags"]["low_atr_15m"])
        self.assertEqual(entry_block_reason(r), "RISK_METRICS_INCOMPLETE")
        old = candidate()
        old["failure_shadow"] = {
            "version": "tp_nonclosure_entry_gate_v1",
            "flags": {"weak_volume": False, "falling_rsi_ema": False,
                      "repeat_lower_entry": False},
        }
        self.assertEqual(entry_block_reason(old), "RISK_ASSESSMENT_MISSING")
        self.assertIsNone(position_fraction(old))

    def test_one_risk_trades_full_notional_and_logs_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            broker=FakeBroker()
            clock=timestamp("2026-11-03T20:00:00")
            executor=NightExecutor(directory,broker,Mock(return_value={"ok":True}),
                                   Mock(),lambda:clock,enabled=True)
            self.addCleanup(executor.guard.close)
            r=candidate(green=.1,volume=1.1)
            r.update(created_at=clock-1,signal_id="one-risk")
            self.risk(r)
            executor.enter(r,clock)
            self.assertEqual(len(broker.buys),1)
            order=executor.state["orders"]["one-risk"]
            self.assertEqual(order["risk_flag_count"],1)
            self.assertEqual(order["risk_target_usdt"],"500.0")
            self.assertEqual(position_fraction(r),1.0)
            result=summarize([r],executor.state["orders"])
            self.assertEqual(result["sizing_cohorts_prospective"]["one_flag_full"]["real_orders_created"],1)
            data=list(csv.DictReader(io.StringIO(export_csv([r],executor.state["orders"]))))
            self.assertEqual(data[0]["real_target_usdt"],"500.0")
            self.assertEqual(data[0]["risk_position_action"],"FULL")

    def test_half_amount_follows_telegram_amount_and_exchange_minimum(self):
        import copy
        from test_crash_services import INFO
        with tempfile.TemporaryDirectory() as directory:
            broker=FakeBroker()
            clock=timestamp("2026-11-03T20:00:00")
            executor=NightExecutor(directory,broker,Mock(return_value={"ok":True}),
                                   Mock(),lambda:clock,enabled=True)
            self.addCleanup(executor.guard.close)
            self.assertIn("700",executor.apply_control("amount:700"))
            two=candidate(green=.1,volume=1.0,rsi=-6,ema=-.11)
            two.update(signal_id="half-700",created_at=clock-1)
            self.risk(two)
            executor.enter(two,clock)
            order=executor.state["orders"]["half-700"]
            self.assertEqual(order["risk_configured_usdt"],"700")
            self.assertEqual(order["risk_target_usdt"],"350.0")
            self.assertLessEqual(float(order["risk_effective_notional_usdt"]),350)
            self.assertEqual(executor.daily_count(),1)

        with tempfile.TemporaryDirectory() as directory:
            broker=FakeBroker()
            info=copy.deepcopy(INFO)
            for filt in info["filters"]:
                if filt["filterType"]=="MIN_NOTIONAL":
                    filt["notional"]="300"
            broker.info=lambda symbol:info
            clock=timestamp("2026-11-03T20:00:00")
            executor=NightExecutor(directory,broker,Mock(return_value={"ok":True}),
                                   Mock(),lambda:clock,enabled=True)
            self.addCleanup(executor.guard.close)
            two=candidate(green=.1,volume=1.0,rsi=-6,ema=-.11)
            two.update(signal_id="half-too-small",created_at=clock-1)
            self.risk(two)
            executor.enter(two,clock)
            self.assertEqual(len(broker.buys),0)
            self.assertEqual(executor.daily_count(),0)
            self.assertNotIn("half-too-small",executor.state["orders"])
            self.assertIn("TP/giriş ön kontrolü başarısız",executor.state["outbox"][-1]["text"])

    def test_preflight_rechecks_3_flags_before_buy(self):
        with tempfile.TemporaryDirectory() as directory:
            broker=FakeBroker()
            clock=timestamp("2026-11-03T20:00:00")
            executor=NightExecutor(directory,broker,Mock(return_value={"ok":True}),
                                   Mock(),lambda:clock,enabled=True)
            self.addCleanup(executor.guard.close)
            two=candidate(green=.1,volume=1.0,rsi=-6,ema=-.11)
            two.update(signal_id="race-to-three",created_at=clock-1)
            self.risk(two)
            def preflight(symbol):
                two["failure_shadow"]["flags"]["repeat_lower_entry"]=True
                return "BOTH"
            broker.preflight=preflight
            executor.enter(two,clock)
            self.assertEqual(len(broker.buys),0)
            self.assertNotIn("race-to-three",executor.state["orders"])

    def test_missing_assessment_blocks_live_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            broker = FakeBroker()
            clock = timestamp("2026-11-03T20:00:00")
            executor = NightExecutor(directory, broker, Mock(return_value={"ok":True}),
                                     Mock(), lambda:clock, enabled=True)
            self.addCleanup(executor.guard.close)
            r = candidate()
            r["signal_id"] = "old-unassessed"
            executor.enter(r, clock)
            self.assertEqual(len(broker.buys), 0)
            self.assertEqual(executor.daily_count(), 0)


if __name__ == "__main__":
    unittest.main()
