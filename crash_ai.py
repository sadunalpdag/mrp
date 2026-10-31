"""OpenAI signal-time research only. No exchange calls or trading dependencies."""
import csv
import hashlib
import io
import json
import math
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODEL = 'gpt-5.4-mini-2026-03-17'
DERIVATIVE_FIELDS = ('oi_change_1h', 'oi_change_4h', 'spot_cvd_1h', 'spot_cvd_4h',
                     'perp_cvd_1h', 'perp_cvd_4h', 'funding_rate', 'price_change_4h')
METRICS = ('rsi_delta_1h', 'ema25_slope_pct', 'ema25_slope_change_1h_pp',
           'green_volume_share_1h', 'atr14_pct', 'btc_24h_change_pct',
           'new_lows_4h', 'bars_since_low_15m', 'volume_ratio_vs_prior_12h',
           'return_1h_pct', 'previous_4h_return_per_hour_pct')
SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'ai_score': {'type': ['integer', 'null'], 'minimum': 0, 'maximum': 100},
    'data_quality': {'type': 'string', 'enum': ['LOW', 'MEDIUM', 'HIGH']},
    'positives': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 2},
    'risks': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 2},
    'missing_data': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 8},
    'summary': {'type': 'string'}},
    'required': ['ai_score', 'data_quality', 'positives', 'risks', 'missing_data', 'summary']}


def num(v):
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError, OverflowError):
        return None


def snapshot(record):
    """Allowlist only frozen market inputs; never send account data, outcomes or scores."""
    if not re.fullmatch(r'[A-Z0-9_]{1,40}', record['symbol']):
        raise ValueError('INVALID_SYMBOL')
    f = record.get('v2_features') or {}
    context = f.get('five_day_context') or {}
    metrics = (f.get('strength_score') or {}).get('metrics') or {}
    return {'symbol': record['symbol'], 'as_of': num(record.get('created_at')),
            'market': {k: num(record.get(k)) for k in
                       ('change_24h', 'entry', 'tp', 'reference_low', 'btc_change_24h')},
            'v2': {k: num(f.get(k)) for k in
                   ('rsi14', 'recovery_4h_pct', 'low_4h', 'return_5d_pct', 'candles')},
            'metrics': {k: num(metrics.get(k)) for k in METRICS},
            'context_5d': {k: num(context.get(k)) for k in
                          ('previous_4d_return_pct', 'return_24h_pct',
                           'drawdown_from_5d_high_pct', 'range_position_5d_pct',
                           'quote_volume_ratio_24h_vs_previous_4d')},
            'derivatives': {k: None for k in DERIVATIVE_FIELDS}}


def validate(value):
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('INVALID_SCHEMA')
    score = value['ai_score']
    if score is not None and (type(score) is not int or not 0 <= score <= 100):
        raise ValueError('INVALID_SCORE')
    if value['data_quality'] not in ('LOW', 'MEDIUM', 'HIGH'):
        raise ValueError('INVALID_QUALITY')
    for key, limit in (('positives', 2), ('risks', 2), ('missing_data', 8)):
        if (not isinstance(value[key], list) or len(value[key]) > limit
                or any(not isinstance(x, str) or len(x) > 160 for x in value[key])):
            raise ValueError('INVALID_ITEMS')
    if not isinstance(value['summary'], str) or len(value['summary']) > 240:
        raise ValueError('INVALID_SUMMARY')
    return value


def cost(usage, model):
    if model not in (MODEL, 'gpt-5.4-mini'):
        return None
    if 'input_tokens' not in usage or 'output_tokens' not in usage:
        return None
    # Conservative: cached tokens charged at full input price. USD, 2026-10-31 rates.
    return round((usage['input_tokens'] * .75 + usage['output_tokens'] * 4.5) / 1e6, 8)


class Analyst:
    def __init__(self, directory, log=print, clock=time.time, post=None, env=None):
        env = os.environ if env is None else env
        self.key = env.get('OPENAI_API_KEY', '').strip()
        self.enabled = env.get('CRASH_AI_ENABLED', '1') == '1' and bool(self.key)
        self.model = env.get('CRASH_AI_MODEL', MODEL)
        self.limit = max(1, min(100, int(env.get('CRASH_AI_MAX_DAILY', '50'))))
        self.prompt = Path(__file__).with_name('crash_ai_prompt.txt').read_text(encoding='utf-8')
        self.version = hashlib.sha256(self.prompt.encode()).hexdigest()[:16]
        self.clock, self.log, self.post = clock, log, post
        self.pending, self.seen, self.counts = {}, set(), {}
        self.results = queue.Queue()
        self.lock = threading.RLock()
        self.next_call = 0
        self.path = Path(directory) / 'ai_events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    # Fail closed: do not duplicate paid requests after damaged journal.
                    self.enabled = False
                    self.log('[CRASH AI DISABLED] INVALID_JOURNAL')
                    break
                if event.get('event') == 'ATTEMPT':
                    self.seen.add(event['signal_id'])
                    d = event['day']
                    self.counts[d] = self.counts.get(d, 0) + 1
                elif event.get('event') == 'RESULT':
                    self.results.put(event['result'])
        self.log('[CRASH AI] ' + ('READY | '+self.model if self.enabled else 'OFF | key/config/journal'))

    def journal(self, event):
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())

    def submit(self, record):
        if not self.enabled or not record.get('v2_sent'):
            return False
        sid = record['signal_id']
        with self.lock:
            if sid in self.seen or sid in self.pending or len(self.pending) >= 100:
                return False
            self.pending[sid] = {'input': snapshot(record), 'queued_at': self.clock(), 'ready': False}
            self.update_derivatives(record.get('derivatives_shadow') or {})
        return True

    def update_derivatives(self, d):
        with self.lock:
            job = self.pending.get(d.get('signal_id'))
            if (not job or num(d.get('as_of')) != job['input']['as_of']
                    or d.get('status') == 'PENDING'):
                return
            job['input']['derivatives'] = {k: num(d.get(k)) for k in DERIVATIVE_FIELDS}
            job['ready'] = True

    def step(self):
        now = self.clock()
        with self.lock:
            if not self.enabled or now < self.next_call:
                return False
            item = next(((sid, j) for sid, j in self.pending.items()
                         if j['ready'] or now-j['queued_at'] >= 120), None)
            if item is None:
                return False
            sid, job = item
            day = datetime.fromtimestamp(now, ZoneInfo('Europe/Paris')).date().isoformat()
            result = {'signal_id': sid, 'symbol': job['input']['symbol'], 'as_of': job['input']['as_of'],
                      'model': self.model, 'prompt_version': self.version, 'input': job['input'],
                      'input_hash': hashlib.sha256(json.dumps(job['input'], sort_keys=True).encode()).hexdigest(),
                      'attempted_at': now, 'status': 'DAILY_CAP', 'usage': {}, 'estimated_usd': None,
                      'assessment': None, 'signal_gate_changed': False}
            allowed = self.counts.get(day, 0) < self.limit
            # Persist before HTTP. Never automatically retry a potentially billed request.
            if allowed:
                self.journal({'event': 'ATTEMPT', 'day': day, 'signal_id': sid})
                self.counts[day] = self.counts.get(day, 0)+1
            del self.pending[sid]
            self.seen.add(sid)
            self.next_call = now+10
        if allowed:
            try:
                post = self.post
                if post is None:
                    import requests
                    post = requests.post
                payload = {'model': self.model, 'store': False, 'instructions': self.prompt,
                           'input': json.dumps(job['input'], ensure_ascii=False, allow_nan=False),
                           'reasoning': {'effort': 'none'}, 'max_output_tokens': 1000,
                           'text': {'format': {'type': 'json_schema', 'name': 'signal_review',
                                               'strict': True, 'schema': SCHEMA}}}
                response = post('https://api.openai.com/v1/responses',
                                headers={'Authorization': 'Bearer '+self.key}, json=payload, timeout=(5, 45))
                if response.status_code != 200:
                    result['status'] = 'HTTP_'+str(response.status_code)
                    if response.status_code in (401, 403):
                        self.enabled = False
                    if response.status_code == 429:
                        self.next_call = self.clock()+300
                else:
                    body = response.json()
                    result['usage'] = {k: v for k, v in (body.get('usage') or {}).items()
                                       if k in ('input_tokens', 'output_tokens', 'total_tokens')
                                       and type(v) is int and v >= 0}
                    result['estimated_usd'] = cost(result['usage'], self.model)
                    result['response_id'] = body.get('id')
                    if body.get('status') != 'completed':
                        raise ValueError('INCOMPLETE_RESPONSE')
                    content = [c for o in body.get('output', []) if o.get('type') == 'message'
                               for c in o.get('content', [])]
                    if any(c.get('type') == 'refusal' for c in content):
                        raise ValueError('MODEL_REFUSAL')
                    text = ''.join(c.get('text', '') for c in content if c.get('type') == 'output_text')
                    result['assessment'] = validate(json.loads(text))
                    if any(job['input'][group].get(key) is None for group, key in
                           (('market', 'entry'), ('market', 'tp'), ('v2', 'rsi14'))):
                        result['assessment']['ai_score'] = None
                        result['assessment']['data_quality'] = 'LOW'
                    result['status'] = 'COMPLETE'
            except Exception as error:
                # Never log request headers, keys, raw server text or exception URLs.
                result['status'] = 'UNAVAILABLE_'+type(error).__name__
        result['finished_at'] = self.clock()
        self.journal({'event': 'RESULT', 'result': result})
        self.results.put(result)
        self.log(f"[CRASH AI RESULT] {result['symbol']} status={result['status']} usd={result['estimated_usd']}")
        return True

    def start(self):
        if not self.enabled:
            return
        def run():
            while True:
                try:
                    self.step()
                except Exception as error:
                    self.log('[CRASH AI WORKER ERR] '+type(error).__name__)
                time.sleep(1)
        threading.Thread(target=run, name='crash-openai-shadow', daemon=True).start()


def format_review(result):
    esc = lambda x: str(x)
    head = f"🤖 OPENAI SİNYAL ANALİZİ | {esc(result['symbol'])}\nSinyal: {esc(result['signal_id'])}\n"
    a = result.get('assessment')
    if not a:
        return head+'Değerlendirme alınamadı: '+esc(result['status'])+'\nV2 sinyali ve emir kararları değişmedi.'
    score = 'yetersiz veri' if a['ai_score'] is None else str(a['ai_score'])+'/100'
    lines = [head+'AI puanı: '+score+' | Veri kalitesi: '+a['data_quality'], esc(a['summary'])]
    lines += ['✅ '+esc(x) for x in a['positives']]
    lines += ['⚠️ '+esc(x) for x in a['risks']]
    if a['missing_data']:
        lines.append('Eksik: '+esc(', '.join(a['missing_data'])))
    lines.append('Deneysel kalite puanı; TP olasılığı değildir. V2/emir kararına etkisi yok.')
    return '\n'.join(lines)


def export_csv(records):
    output = io.StringIO()
    fields = ('signal_id', 'symbol', 'as_of', 'model', 'prompt_version', 'status', 'ai_score',
              'data_quality', 'input_tokens', 'output_tokens', 'estimated_usd',
              'tracking_status', 'shadow_mae_pct', 'shadow_new_low_observed')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        d = record.get('ai_review')
        if not d:
            continue
        row = {k: d.get(k) for k in fields}
        row.update({k: (d.get('assessment') or {}).get(k) for k in ('ai_score', 'data_quality')})
        row.update({k: (d.get('usage') or {}).get(k) for k in ('input_tokens', 'output_tokens')})
        row.update({k: record.get(k) for k in ('tracking_status', 'shadow_mae_pct', 'shadow_new_low_observed')})
        writer.writerow(row)
    return output.getvalue()
