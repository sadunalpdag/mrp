"""Offline tests: keyless OKX swap last-closed recovery and urgent signal lane."""
import tempfile
import unittest
from unittest.mock import Mock

from crash_data_sources import STEP, tagged
from crash_history import HistoryCollector
from crash_okx import OKXLastClosed
from crash_v2 import assess_entry

COUNT = 480


def candles(start, count):
    return [[t, "1", "1.2", ".8", "1.1", "10", t+STEP-1, "11",
             0, 0, 0, 0] for t in range(start, start+count*STEP, STEP)]


def okx_reply(data):
    response = Mock(status_code=200)
    response.json.return_value = {"code": "0", "data": data}
    return response


def instrument():
    return okx_reply([{"instId": "BTC-USDT-SWAP", "instType": "SWAP",
                       "settleCcy": "USDT", "ctType": "linear", "state": "live"}])


def okx_bars(end, confirmed="1", anchor_close="1.1"):
    return okx_reply([
        [str(end), "1", "1.2", ".8", "1.1", "10", "10", "11", "0"],
        [str(end-STEP), "1", "1.2", ".8", "1.1", "10", "10", "11", confirmed],
        [str(end-2*STEP), "1", "1.2", ".8", anchor_close, "10", "10", "11", "1"],
    ])


class UrgentOKXTests(unittest.TestCase):
    def setUp(self):
        self.end = 20000*86400000
        self.start = self.end-COUNT*STEP
        self.now = self.end/1000+40
        self.get = Mock(side_effect=[instrument(), okx_bars(self.end)])
        self.log = Mock()
        self.okx = OKXLastClosed(self.log, get=self.get, clock=lambda:self.now)
        self.cache = candles(self.start,COUNT-1)

    def test_closed_okx_last_bar_requires_matching_swap_and_binance_anchor(self):
        result = self.okx.fill_last("BTCUSDT",self.start,self.end,self.cache)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], self.end-STEP)
        self.assertEqual(result[0][12], "OKX_SWAP")
        self.assertEqual(self.get.call_count,2)
        self.assertEqual(self.get.call_args.kwargs["params"]["bar"], "15m")
        self.assertFalse(self.get.call_args.kwargs["allow_redirects"])

    def test_unclosed_15m_bar_is_rejected(self):
        self.get.side_effect=[instrument(), okx_bars(self.end, confirmed="0")]
        self.assertEqual(self.okx.fill_last("BTCUSDT",self.start,self.end,self.cache),[])

    def test_nonmatching_swap_is_rejected_before_candles_call(self):
        self.get.side_effect=[okx_reply([{"instId":"BTC-USDT-SWAP",
             "instType":"SPOT","settleCcy":"USDT","ctType":"linear","state":"live"}])]
        self.assertEqual(self.okx.fill_last("BTCUSDT",self.start,self.end,self.cache),[])
        self.get.assert_called_once()

    def test_large_price_discrepancy_is_rejected(self):
        self.get.side_effect=[instrument(), okx_bars(self.end,anchor_close="1.15")]
        self.assertEqual(self.okx.fill_last("BTCUSDT",self.start,self.end,self.cache),[])

    def test_no_binance_adjacent_anchor_never_uses_okx(self):
        self.assertEqual(self.okx.fill_last("BTCUSDT",self.start,self.end,self.cache[:-1]),[])
        self.get.assert_not_called()

    def test_no_requests_before_last_candle_closes(self):
        self.now=self.end/1000-10
        self.assertEqual(self.okx.fill_last("BTCUSDT",self.start,self.end,self.cache),[])
        self.get.assert_not_called()

    def test_recent_signal_is_emitted_despite_research_pacing_and_hourly_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            emit, fetch, repair=Mock(),Mock(),Mock(return_value=[
                tagged(candles(self.end-STEP,1)[0], "OKX_SWAP")])
            worker=HistoryCollector(directory,fetch,emit,lambda:True,self.log,
                                    resolver=Mock(),urgent_repair=repair)
            worker._write(directory+"/BTCUSDT.json",self.cache)
            worker.next_download=self.now+3600
            worker.attempts=[self.now-1]*18
            worker.submit({"symbol":"BTCUSDT","signal_id":"signal1",
                           "created_at":self.end/1000+10},
                          "signal",self.end/1000+10)
            worker.step(now=self.now)
            fetch.assert_not_called()
            worker.resolver.resolve.assert_not_called()
            event=emit.call_args.args[0]
            self.assertEqual(event["candle_count"],480)
            self.assertEqual(event["closed_15m_ohlcv"][-1][0],self.end-STEP)
            self.assertEqual(event["data_sources"]["OKX_SWAP"],1)
            self.assertEqual(assess_entry(event["closed_15m_ohlcv"],[],
                                          self.end/1000+10)["status"],"STRONG_V2")

    def test_old_signal_does_not_use_fast_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            emit, repair=Mock(),Mock()
            worker=HistoryCollector(directory,Mock(),emit,lambda:False,self.log,
                                    resolver=Mock(),urgent_repair=repair)
            worker._write(directory+"/BTCUSDT.json",self.cache)
            worker.next_download=self.now+1000
            worker.submit({"symbol":"BTCUSDT","signal_id":"old"},
                          "signal",self.end/1000-1000)
            worker.step(now=self.now)
            repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
