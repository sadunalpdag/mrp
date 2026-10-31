import json
from datetime import datetime, timezone
import os
import tempfile
import unittest
from unittest.mock import Mock

from ai_market_agents import (
    AgentsResearch, technical_extra, delta_proxy, derivatives_for_symbol, ROLE_SCHEMA,
)


def candles():
    # 60 closed 1h bars; all fields needed by Binance kline formulas included.
    base = 1700000000 * 1000
    rows = []
    for i in range(60):
        price = 100 + i * 0.1
        rows.append([base + i*3600000, str(price), str(price+1.5),
                     str(price-1.5), str(price+0.25), '100',
                     base + (i+1)*3600000 - 1, '10000', '100', '60',
                     '5500', '0'])
    return rows


class FakeResponse:
    status_code = 200
    def __init__(self, data):
        self.data = data
    def json(self):
        return self.data


class Checks(unittest.TestCase):
    def test_closed_candle_technical_indicators(self):
        r = technical_extra(candles(), now=1800000000)
        self.assertEqual(r['coverage'], 'CLOSED_1H')
        self.assertIn('macd_histogram', r)
        self.assertIsNotNone(r['adx14'])
        self.assertLess(r['support_20h'], r['resistance_20h'])

    def test_taker_delta_is_not_real_cvd(self):
        self.assertAlmostEqual(delta_proxy(candles(), now=1800000000), 10.0)

    def test_derivatives_missing_is_explicit(self):
        def fetch(path, params):
            raise RuntimeError('cooldown')
        r = derivatives_for_symbol('BTCUSDT', candles(), fetch, now=1800000000)
        self.assertIsNone(r['oi_change_4h_pct'])
        self.assertIn('OI_API_UNAVAILABLE', r['missing'])
        self.assertIn('FUNDING_API_UNAVAILABLE', r['missing'])

    def test_four_role_batch_and_capped_one_search(self):
        syms = [f'COIN{i}USDT' for i in range(10)]
        entries = [{'symbol': sym, 'return_4h_pct': 4.0,
                    'atr14_pct': 2.0, 'rsi14_1h': 60.0} for sym in syms]
        schema = {'type': 'object', 'properties': {}, 'required': []}
        received = []
        def post(url, headers, json, timeout):
            received.append(json)
            if json.get('tools'):
                return FakeResponse({'status':'completed','usage': {'input_tokens':1000,'output_tokens':100},
                  'output':[{'type':'web_search_call'}, {'type':'message', 'content':[{'type':'output_text',
                   'text':'BTC market update published ' + datetime.now(timezone.utc).strftime('%Y-%m-%d'), 'annotations':[{'type':'url_citation',
                   'url':'https://example.com/market','title':'Market news'}]}]}]})
            is_final = json['text']['format']['name'] == 'top_final'
            out = ({'analyses':[{'symbol':s,'score':80,'bias':'CONTINUE',
                   'reason':'test','risk':'test','entry_price':100} for s in syms]}
                   if is_final else
                   {'reviews':[{'symbol':s,'score':80,'stance':'BULL',
                   'evidence':'test','risk':'test'} for s in syms]})
            return FakeResponse({'status':'completed','usage':{'input_tokens':1000,'output_tokens':100},
                'output':[{'type':'message','content':[{'type':'output_text','text':__import__('json').dumps(out)}]}]})
        calls = []
        def fetch(path, params):
            calls.append((path, params))
            if 'openInterestHist' in path:
                return [{'sumOpenInterest':v} for v in (100,101,102,103,104,105)]
            if 'fundingRate' in path:
                return [{'fundingRate':'0.0001'}]
            return candles()
        engine = AgentsResearch('example_key', post=post)
        result = engine.evaluate(entries, klines={s:candles() for s in syms},
                fetch=fetch, final_schema=schema, market_prices={s:100 for s in syms})
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(len(result['analyses']), 10)
        self.assertEqual(len(received), 5)
        self.assertEqual(sum(bool(c.get('tools')) for c in received), 1)
        self.assertEqual(result['web_tool_calls'], 1)
        self.assertEqual(len(calls), 20)
        self.assertEqual(result['agents']['news']['status'], 'SOURCED_DATE_REPORTED')
        self.assertTrue(result['estimated_total_usd'] >= 0.01)

    def test_news_without_citations_is_not_counted(self):
        syms = [f'C{i}USDT' for i in range(10)]
        entries=[{'symbol':s} for s in syms]
        def post(url, headers, json, timeout):
            if json.get('tools'):
                return FakeResponse({'status':'completed','usage':{},
                    'output':[{'type':'web_search_call'},{'type':'message','content':[{'type':'output_text','text':'No sources'}]}]})
            is_final=json['text']['format']['name']=='top_final'
            out=({'analyses':[{'symbol':s} for s in syms]} if is_final else
                 {'reviews':[{'symbol':s} for s in syms]})
            return FakeResponse({'status':'completed','usage':{},'output':[{'type':'message',
                'content':[{'type':'output_text','text':__import__('json').dumps(out)}]}]})
        def fetch(path, params):
            return []
        bot=AgentsResearch('key',post=post)
        results=bot.evaluate(entries,klines={},fetch=fetch,final_schema={},market_prices={})
        self.assertEqual(results['agents']['news']['status'],'NO_VERIFIED_RECENT_DATED_SOURCES')

if __name__=='__main__':
    unittest.main()
