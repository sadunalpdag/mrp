import unittest
from profit_close_reconcile import reconcile_close, normalize_positions


def position(qty, side="BOTH", symbol="ETHUSDT"):
    return {"symbol": symbol, "positionSide": side, "positionAmt": str(qty)}


def order(client_id, side="BUY", **extra):
    return dict(symbol="ETHUSDT", side=side, clientOrderId=client_id, **extra)


class ReconcileTests(unittest.TestCase):
    def test_open_position_prevents_done_even_no_orders(self):
        result = reconcile_close([], [position("1")], [], [], set(), set())
        self.assertFalse(result["all_clear"])

    def test_partially_filled_position_prevents_done(self):
        result = reconcile_close([dict(symbol="ETHUSDT", position_side="BOTH")],
                                 [position("0.05")], [], [], set(), {"exit"})
        self.assertFalse(result["positions_flat"])

    def test_pending_owned_close_prevents_done(self):
        result = reconcile_close([], [], [order("exit", side="SELL", reduceOnly=True)],
                                 [], set(), {"exit"})
        self.assertFalse(result["closing_orders_cleared"])

    def test_owned_buy_prevents_done(self):
        result = reconcile_close([], [], [order("entry")], [], {"entry"}, set())
        self.assertFalse(result["owned_entries_cleared"])

    def test_unknown_manual_buy_prevents_global_unlock(self):
        result = reconcile_close([], [], [order("manual")], [], set(), set())
        self.assertTrue(result["positions_flat"])
        self.assertFalse(result["all_clear"])

    def test_verified_clear(self):
        result = reconcile_close([], [], [], [], {"entry"}, {"exit"})
        self.assertTrue(result["all_clear"])

    def test_hedge_mode_and_malformed_exchange_response(self):
        p = normalize_positions([position("1", "LONG"), position("-2", "SHORT")])
        self.assertEqual(len(p), 2)
        with self.assertRaises(ValueError):
            normalize_positions([position("-1", "LONG")])
        with self.assertRaises(ValueError):
            reconcile_close([], [], None, [], set(), set())


if __name__ == "__main__":
    unittest.main()
