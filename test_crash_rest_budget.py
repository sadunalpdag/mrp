"""Offline tests for shared Binance IP request-weight scheduler."""
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
from crash_rest_budget import (
    IpBudget, RateDeferred, estimated_weight, priority_for,
    RESEARCH_CEILING,ACCOUNT_CEILING,PROTECT_CEILING,
)


class SharedIpBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now=[60.0]
        self.b=IpBudget(clock=lambda:self.now[0], sleep=lambda s:self._advance(s))

    def _advance(self,delta):
        self.now[0]+=delta

    def test_estimated_weights_and_priority(self):
        self.assertEqual(estimated_weight('/fapi/v1/openOrders'),40)
        self.assertEqual(estimated_weight('/fapi/v1/openOrders',{'symbol':'BTCUSDT'}),1)
        self.assertEqual(estimated_weight('/fapi/v1/openAlgoOrders'),40)
        self.assertEqual(estimated_weight('/fapi/v1/klines',{'limit':60}),1)
        self.assertEqual(estimated_weight('/fapi/v1/klines',{'limit':480}),2)
        self.assertEqual(estimated_weight('/fapi/v1/klines',{'limit':800}),5)
        self.assertEqual(priority_for('POST','/fapi/v1/order',{'side':'BUY'}),'account')
        self.assertEqual(priority_for('POST','/fapi/v1/order',{'side':'SELL'}),'protect')
        self.assertEqual(priority_for('GET','/fapi/v1/order'),'protect')

    def test_research_paced_without_holding_tp_global_mutex(self):
        self.b.reserve('research',2)
        start=self.now[0]
        self.b.reserve('protect',5)
        self.assertLess(self.now[0]-start,1)
        self.b.reserve('research',2)
        self.assertGreaterEqual(self.now[0]-start,20)
        self.assertEqual(self.b.stats()['reserved_weight'],9)

    def test_server_header_suppresses_research_but_retains_tp_capacity(self):
        self.b.reserve('research',1)
        self.b.observe({'X-MBX-USED-WEIGHT-1M':'600'},200)
        with self.assertRaises(RateDeferred) as caught:
            self.b.reserve('research',1)
        self.assertEqual(caught.exception.reason,'IP_WEIGHT_SOFT_CEILING')
        self.b.reserve('protect',5)
        self.assertLess(self.b.stats()['reserved_weight'], PROTECT_CEILING)
        self.b.observe({'X-MBX-USED-WEIGHT-1M':'1750'},200)
        with self.assertRaises(RateDeferred):
            self.b.reserve('protect',1)

    def test_new_minute_resets_weight(self):
        self.b.observe({'X-MBX-USED-WEIGHT-1M':'700'},200)
        with self.assertRaises(RateDeferred):
            self.b.reserve('research',1)
        self.now[0]=120
        self.b.reserve('research',1)
        self.assertEqual(self.b.stats()['server_weight'],0)

    def test_429_and_418_pause_all_venues_until_retry(self):
        self.b.observe({'X-MBX-USED-WEIGHT-1M':'2500'},429,retry_after='150')
        for tier in ('research','account','protect'):
            with self.assertRaises(RateDeferred):
                self.b.reserve(tier,1)
        self.now[0]+=156
        self.b.reserve('account',1)
        self.b.observe({},418,'2400')
        self.assertGreaterEqual(self.b.blocked_until,self.now[0]+2400)
        with self.assertRaises(RateDeferred):
            self.b.reserve('protect',1)

    def test_actual_broker_request_observes_weight_and_respects_budget(self):
        from crash_night import BinanceBroker, BrokerError
        broker=BinanceBroker('test_key','test_secret')
        response=Mock(status_code=200,headers={'X-MBX-USED-WEIGHT-1M':'690'})
        response.json.return_value={'ok':True}
        req=Mock(return_value=response)
        with patch('crash_rest_budget.BUDGET',self.b),patch.dict(
                'sys.modules',{'requests':SimpleNamespace(request=req)}):
            self.assertEqual(broker.request('GET','/fapi/v1/exchangeInfo',signed=False),{'ok':True})
            self.assertEqual(req.call_count,1)
            with self.assertRaises(BrokerError) as blocked:
                broker.request('GET','/fapi/v1/openOrders')
            self.assertEqual(blocked.exception.code,'-1003')
            self.assertEqual(req.call_count,1)
            self.assertEqual(broker.request('GET','/fapi/v1/order',{'symbol':'BTCUSDT'}),{'ok':True})
            self.assertEqual(req.call_count,2)


if __name__ == '__main__':
    unittest.main()
