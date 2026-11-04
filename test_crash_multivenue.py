"""Offline tests for paced Bybit -> Coinbase -> Binance REST fallback."""
import tempfile
import unittest
from unittest.mock import Mock
from crash_coinbase import CoinbaseReference, STEP, MIN_REQUEST_GAP_SECONDS
from crash_data_sources import DataSources, tagged, source_counts


def candles(start,n,price="1.1"):
    return [[t,"1","1.2","0.8",price,"10",t+STEP-1,"11",0,0,0,0]
            for t in range(start,start+n*STEP,STEP)]


def response(rows,status=200):
    r=Mock(status_code=status,headers={})
    r.json.return_value=[
        [int(x[0]/1000),float(x[3]),float(x[2]),float(x[1]),
         float(x[4]),float(x[5])] for x in reversed(rows)]
    return r


class MultiVenueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.start=20000*86400000
        self.end=self.start+3*STEP
        self.now=[self.end/1000+60]
        self.log=Mock()
        self.get=Mock()
        self.cb=CoinbaseReference(self.temp.name,self.log,get=self.get,
                                  clock=lambda:self.now[0])

    def test_coinbase_only_missing_with_anchor_and_source_tag(self):
        self.get.return_value=response(candles(self.start,3))
        existing=[tagged(candles(self.start,1)[0],"BINANCE_ARCHIVE")]
        out=self.cb.fill_missing("BTCUSDT",self.start,self.end,
                                 [self.start+STEP,self.start+2*STEP],existing)
        self.assertEqual(len(out),2)
        self.assertEqual([x[12] for x in out],["COINBASE_SPOT"]*2)
        self.assertEqual(self.get.call_args.args[0],
                         "https://api.exchange.coinbase.com/products/BTC-USD/candles")
        self.assertEqual(self.get.call_args.kwargs["params"]["granularity"],900)
        self.assertFalse(self.get.call_args.kwargs["allow_redirects"])
        self.assertEqual(self.cb.fill_missing("BTCUSDT",self.start,self.end,
                           [self.start+STEP],existing),[])
        self.get.assert_called_once()

    def test_coinbase_404_cached_and_429_cooldown(self):
        existing=candles(self.start,1)
        self.get.return_value=response([],status=404)
        self.assertEqual(self.cb.fill_missing("UNKNOWNUSDT",self.start,self.end,
                        [self.start+STEP],existing),[])
        self.now[0]+=MIN_REQUEST_GAP_SECONDS
        self.assertEqual(self.cb.fill_missing("UNKNOWNUSDT",self.start,self.end,
                        [self.start+STEP],existing),[])
        self.get.assert_called_once()
        self.get.return_value=response([],status=429)
        self.assertEqual(self.cb.fill_missing("BTCUSDT",self.start,self.end,
                        [self.start+STEP],existing),[])
        self.assertGreater(self.cb.global_cooldown,self.now[0])

    def test_spot_mismatch_fails_closed(self):
        self.get.return_value=response(candles(self.start,3,"1.6"))
        self.assertEqual(self.cb.fill_missing("BTCUSDT",self.start,self.end,
                         [self.start+STEP],candles(self.start,1)),[])
        self.assertTrue(any("CROSS_EXCHANGE_PRICE_MISMATCH" in str(c)
                            for c in self.log.call_args_list))

    def test_source_order_bybit_coinbase_rest_and_no_rest_if_filled(self):
        events=[]
        bybit=Mock()
        coinbase=Mock()
        def bybit_fill(*args):
            events.append("BYBIT")
            return []
        def coinbase_fill(*args):
            events.append("COINBASE")
            return [tagged(r,"COINBASE_SPOT") for r in candles(self.start+STEP,2)]
        bybit.fill_missing.side_effect=bybit_fill
        coinbase.fill_missing.side_effect=coinbase_fill
        def rest(*args):
            events.append("BINANCE")
            return []
        ds=DataSources(self.temp.name,rest,lambda:False,self.log,
                       clock=lambda:self.now[0],bybit=bybit,coinbase=coinbase)
        ds.archive=Mock(return_value=[])
        out=ds.resolve("BTCUSDT",self.start,self.end,candles(self.start,1))
        self.assertEqual(events,["BYBIT","COINBASE"])
        self.assertEqual(source_counts(out)["COINBASE_SPOT"],2)

    def test_binance_only_after_both_fail_and_respects_ban(self):
        events=[]
        bybit=Mock(fill_missing=Mock(side_effect=lambda *a:events.append("BYBIT") or []))
        coinbase=Mock(fill_missing=Mock(side_effect=lambda *a:events.append("COINBASE") or []))
        def rest(path,params):
            events.append("BINANCE")
            return candles(params["startTime"],params["limit"])
        blocked=[False]
        ds=DataSources(self.temp.name,rest,lambda:blocked[0],self.log,
                       clock=lambda:self.now[0],bybit=bybit,coinbase=coinbase)
        ds.archive=Mock(return_value=[])
        out=ds.resolve("BTCUSDT",self.start,self.end,candles(self.start,1))
        self.assertEqual(events,["BYBIT","COINBASE","BINANCE"])
        self.assertEqual(source_counts(out)["BINANCE_REST"],2)
        events.clear();blocked[0]=True
        ds.resolve("ETHUSDT",self.start,self.end,candles(self.start,1))
        self.assertEqual(events,["BYBIT","COINBASE"])

    def test_no_unanchored_coinbase_and_never_overwrite_binance(self):
        self.assertEqual(self.cb.fill_missing("BTCUSDT",self.start,self.end,
                         [self.start],[]),[])
        self.get.assert_not_called()
        ds=DataSources(self.temp.name,Mock(),lambda:True,self.log,
                       clock=lambda:self.now[0],coinbase=Mock(
                           fill_missing=Mock(return_value=candles(self.start,3))))
        ds.archive=Mock(return_value=[])
        result=ds.resolve("BTCUSDT",self.start,self.end,candles(self.start,1))
        self.assertEqual(result[0][4],"1.1")


if __name__=="__main__":
    unittest.main()
