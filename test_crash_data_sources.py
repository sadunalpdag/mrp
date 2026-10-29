import ast
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from crash_data_sources import DataSources, STEP, DAY, valid_rows, REST_MAX_CANDLES, source_counts, tagged
from crash_history import HistoryCollector


def rows(start,n):
    return [[t,'1','1.2','.8','1.1','10',t+STEP-1,'11',5,'5','5.5','0'] for t in range(start,start+n*STEP,STEP)]


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.today=20000*DAY;self.now=self.today/1000+36000
        self.rest=Mock();self.log=Mock()
        self.s=DataSources(self.tmp.name,self.rest,lambda:False,self.log,clock=lambda:self.now)

    def test_complete_ws_cache_uses_no_network(self):
        self.s.archive=Mock()
        out=self.s.resolve('BTCUSDT',self.today,self.today+STEP,rows(self.today,1))
        self.assertEqual(len(out),1)
        self.rest.assert_not_called();self.s.archive.assert_not_called()

    def test_archive_precedes_rest_and_preserves_ws_overlap(self):
        start=self.today-DAY
        order=[]
        def archive(symbol,day):
            order.append('ARCHIVE')
            return []
        def rest(*args,**kwargs):
            order.append('REST')
            return rows(start,1)
        self.s.archive=Mock(side_effect=archive)
        self.rest.side_effect=rest
        cache=rows(start+STEP,1);cache[0][4]='1.15'
        out=self.s.resolve('BTCUSDT',start,start+2*STEP,cache)
        self.assertEqual(order,['ARCHIVE','REST'])
        self.assertEqual(out[1][4],'1.15')
        self.assertEqual(out[0][12],'BINANCE_REST')
        self.assertEqual(self.rest.call_args.args[1]['limit'],1)

    def test_archive_and_ws_fill_most_of_480_rest_gets_only_remaining(self):
        end=self.today+40*STEP
        start=end-480*STEP
        archived=[tagged(r,'BINANCE_ARCHIVE') for r in rows(start,309)]
        ws=rows(end-STEP,1)
        self.s.archive=Mock(side_effect=lambda symbol,day:archived)
        self.rest.return_value=rows(start+309*STEP,170)
        out=self.s.resolve('BTCUSDT',start,end,ws)
        self.assertEqual(len(out),480)
        self.assertEqual(source_counts(out),{
            'BINANCE_ARCHIVE':309,'BINANCE_REST':170,'BINANCE_CACHE_WS':1})
        params=self.rest.call_args.args[1]
        self.assertEqual(params['startTime'],start+309*STEP)
        self.assertEqual(params['endTime'],end-STEP-1)
        self.assertEqual(params['limit'],170)

    def test_full_archive_never_uses_rest(self):
        end=self.today
        start=end-480*STEP
        self.s.archive=Mock(side_effect=lambda symbol,day:
            [tagged(r,'BINANCE_ARCHIVE') for r in rows(day,96)])
        out=self.s.resolve('BTCUSDT',start,end,[])
        self.assertEqual(len(out),480)
        self.assertEqual(source_counts(out),{'BINANCE_ARCHIVE':480})
        self.rest.assert_not_called()

    def test_archive_missing_never_requests_480_from_rest(self):
        end=self.today
        start=end-480*STEP
        self.s.archive=Mock(return_value=[])
        self.rest.side_effect=lambda path,params:rows(params['startTime'],params['limit'])
        cache=[]
        calls=[]
        for _ in range(3):
            cache=self.s.resolve('BTCUSDT',start,end,cache)
            calls.append(self.rest.call_args.args[1]['limit'])
        self.assertEqual(calls,[REST_MAX_CANDLES,REST_MAX_CANDLES,96])
        self.assertEqual(len(cache),480)
        self.assertLess(max(calls),480)

    def test_ban_still_uses_archive_but_not_today(self):
        self.s.blocked=lambda:True
        start=self.today-STEP
        self.s.archive=Mock(return_value=rows(start,1))
        out=self.s.resolve('BTCUSDT',start,self.today+STEP,[])
        self.assertEqual(len(out),1)
        self.rest.assert_not_called()
        self.s.archive.assert_called_once_with('BTCUSDT',self.today-DAY)

    def test_rest_error_falls_through_and_invalid_archive_is_rejected(self):
        self.rest.side_effect=RuntimeError('offline')
        start=self.today-DAY
        bad=rows(start,1);bad[0][3]='2'
        self.s.archive=Mock(return_value=bad)
        self.assertEqual(self.s.resolve('BTCUSDT',start,start+STEP,[]),[])

    def test_zip_checksum_header_and_cache(self):
        start=self.today-DAY
        content=io.BytesIO()
        with zipfile.ZipFile(content,'w') as z:
            z.writestr('candles.csv','open_time,open,high,low,close,volume,close_time\n'+
                '\n'.join(','.join(map(str,r)) for r in rows(start,96)))
        data=content.getvalue()
        self.s._download=Mock(side_effect=[data,(hashlib.sha256(data).hexdigest()+' file.zip').encode()])
        result=self.s.archive('BTCUSDT',start)
        self.assertEqual(len(result),96);self.assertEqual(result[0][12],'BINANCE_ARCHIVE')
        self.s.archive('BTCUSDT',start)
        self.assertEqual(self.s._download.call_count,2)

    def test_bad_checksum_and_missing_file_are_deferred(self):
        self.s._download=Mock(side_effect=[b'bad',b'0'*64])
        self.assertEqual(self.s.archive('BTCUSDT',self.today-DAY),[])
        self.assertEqual(self.s.archive('BTCUSDT',self.today-DAY),[])
        self.assertEqual(self.s._download.call_count,2)

    def test_collector_fallback_runs_during_rest_pause_and_merges_ws(self):
        start=self.today-480*STEP;end=self.today
        self.s.blocked=lambda:True
        self.s.archive=Mock(side_effect=lambda sym,day:rows(day,96))
        ws=rows(end-STEP,1);ws[0][4]='1.19'
        emit=Mock();merge=Mock()
        c=HistoryCollector(self.tmp.name+'/history',self.rest,emit,lambda:True,self.log,
            resolver=self.s,ws_rows=lambda sym:ws,on_rows=merge)
        c.submit({'symbol':'BTCUSDT','signal_id':'warmup:BTCUSDT'},'warmup',end/1000)
        c.step(now=self.now)
        saved=json.loads(Path(c.directory,'BTCUSDT.json').read_text())
        self.assertEqual(len(saved),480);self.assertEqual(saved[-1][4],'1.19')
        self.assertEqual(c.jobs,{})
        emit.assert_not_called();merge.assert_called_once();self.rest.assert_not_called()

    def test_initial_four_hour_scan_queues_even_under_ban(self):
        tree=ast.parse(Path(__file__).with_name('ema.py').read_text())
        names={'_crash_get_candles','_crash_cached_candles_valid'}
        g=dict(time=SimpleNamespace(time=lambda:self.now),CRASH_DATA_FALLBACK_ENABLED=1,
            _CRASH_WS_CACHE=SimpleNamespace(candles=lambda s:[],merge_candles=Mock()),
            CRASH_BOUNCE_RESEARCH_DIR=self.tmp.name,os=__import__('os'),json=json,
            _crash_queue_history=Mock(),log=Mock(),_crash_cooldown_until=self.now+3600)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'ema.py','exec'),g)
        self.assertEqual(g['_crash_get_candles']('BTCUSDT'),[])
        self.assertEqual(g['_crash_queue_history'].call_args.args[1],'warmup')

    def test_no_open_or_future_candles(self):
        start=self.today
        self.rest.return_value=rows(start,2)
        self.now=(start+STEP)/1000
        out=self.s.resolve('BTCUSDT',start,start+2*STEP,[])
        self.assertEqual(len(out),1)

    def test_transport_has_no_keys_and_rejects_redirects(self):
        response=Mock(status_code=200)
        response.iter_content.return_value=[b'ok']
        self.s.get=Mock(return_value=response)
        self.assertEqual(self.s._download('https://data.binance.vision/test'),b'ok')
        self.assertEqual(self.s.get.call_args.kwargs['headers'],{})
        self.assertFalse(self.s.get.call_args.kwargs['allow_redirects'])
        response.close.assert_called_once()
