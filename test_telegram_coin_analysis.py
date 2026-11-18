import tempfile
import unittest
from unittest.mock import Mock, patch
from crash_telegram_control import TelegramControl
from telegram_coin_analysis import normalize

class CoinTelegramTests(unittest.TestCase):
    def test_symbols(self):
        self.assertEqual(normalize("sol"), "SOLUSDT")
        self.assertEqual(normalize("BTCUSDT"), "BTCUSDT")
        self.assertIsNone(normalize("/otomatik_ac"))
        self.assertIsNone(normalize("buy BTC"))
        self.assertIsNone(normalize("BTC/USDT"))

    def test_owner_only_and_existing_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            c = TelegramControl(directory, "fake", "123", None, Mock(), Mock(), Mock(),
                                lambda:1000, lambda offset:[], counter_only=True,
                                require_owner=False, allow_toggle=True)
            def msg(sender, text):
                return {"message":{"chat":{"id":123},"from":{"id":sender,"is_bot":False},
                                   "date":1000,"text":text}}
            self.assertEqual(c.command(msg(123,"SOL")), "coin_analysis:SOLUSDT")
            self.assertIsNone(c.command(msg(456,"SOL")))
            self.assertEqual(c.command(msg(123,"/otomatik_durum")), "status")

    def test_analysis_never_calls_trade_action(self):
        with tempfile.TemporaryDirectory() as directory:
            action, deliver = Mock(), Mock(return_value={"ok":True})
            c = TelegramControl(directory,"fake","123",None,action,deliver,Mock(),
                                lambda:1000,lambda offset:[{"update_id":1,"message":{
                                "chat":{"id":123},"from":{"id":123,"is_bot":False},
                                "date":1000,"text":"SOL"}}],counter_only=True,
                                require_owner=False)
            with patch("telegram_coin_analysis.analyze",return_value="SANAL") as ai:
                c.step()
            ai.assert_called_once_with("SOLUSDT")
            action.assert_not_called()
            deliver.assert_called_once_with("SANAL")

if __name__ == "__main__":
    unittest.main()
