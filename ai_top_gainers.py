"""Independent four-hour futures gainers report. Never places trades."""
import json
import math
import os
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

INTERVAL = 4 * 3600
MODEL = "gpt-5.4-mini-2026-03-17"
SCHEMA = {"type":"object","additionalProperties":False,
 "properties":{"analyses":{"type":"array","maxItems":10,"items":{"type":"object",
 "additionalProperties":False,"properties":{
 "symbol":{"type":"string"},"score":{"type":"integer","minimum":0,"maximum":100},
 "bias":{"type":"string","enum":["CONTINUE","PULLBACK","UNCERTAIN"]},
 "reason":{"type":"string"},"risk":{"type":"string"}},
 "required":["symbol","score","bias","reason","risk"]}}},
 "required":["analyses"]}
INSTRUCTIONS = """You are an independent Binance USDT-M futures market research analyst.
Evaluate ONLY the ten supplied top four-hour gainers. Base each claim solely on
provided timestamped data; do not invent news, CVD, funding, OI or real-time data.
Use bull, bear and risk perspectives, then give one concise conclusion per coin.
The score measures subjective continuation strength, NOT win probability.
Consider momentum, RSI, EMA25, EMA50, realized volume, ATR and overextension.
Do not suggest leverage or execute trades. Turkish output, max 110 characters
for each reason and risk. All ten listed symbols must appear exactly once.
If evidence is sparse, use UNCERTAIN and conservative scores."""

def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError, OverflowError):
        return None

def ema(values, period):
    a=2/(period+1); v=values[0]
    for n in values[1:]:
        v=a*n+(1-a)*v
    return v

def features(symbol, rows, ret, volume_24h):
    # Futures 1h klines; Binance may include an unclosed trailing candle.
    nowms=int(time.time()*1000)
    closed=[r for r in rows if len(r)>7 and int(r[6])<nowms]
    if len(closed)<28:
        return {"symbol":symbol,"return_4h_pct":round(ret,3),"data":"insufficient"}
    closes=[float(r[4]) for r in closed]
    highs=[float(r[2]) for r in closed]
    lows=[float(r[3]) for r in closed]
    volumes=[float(r[7]) for r in closed]
    deltas=[closes[i]-closes[i-1] for i in range(1,len(closes))]
    ups=[max(x,0) for x in deltas[-14:]]
    downs=[max(-x,0) for x in deltas[-14:]]
    gain=sum(ups)/14; loss=sum(downs)/14
    rsi=100 if loss==0 else 100-100/(1+gain/loss)
    tr=[max(highs[i]-lows[i], abs(highs[i]-closes[i-1]),
            abs(lows[i]-closes[i-1])) for i in range(1,len(closes))]
    prevvol=sum(volumes[-28:-4])/24
    return {"symbol":symbol,"return_4h_pct":round(ret,3),
            "price":closes[-1],"rsi14_1h":round(rsi,2),
            "ema25_1h":round(ema(closes,25),8),
            "ema50_1h":round(ema(closes,50),8) if len(closes)>=50 else None,
            "atr14_pct":round(100*sum(tr[-14:])/14/closes[-1],3),
            "volume_4h_to_prior_24h_hourly":round(sum(volumes[-4:])/4/prevvol,3) if prevvol>0 else None,
            "quote_volume_24h":number(volume_24h)}

class TopGainers:
    def __init__(self, directory, cache, fetch, telegram, log=print,
                 post=None, clock=time.time, env=None):
        self.env=os.environ if env is None else env
        self.enabled=self.env.get("TOP_GAINERS_ENABLED","1")=="1"
        self.key=self.env.get("OPENAI_API_KEY","").strip()
        self.cache,self.fetch,self.telegram,self.log,self.post,self.clock=cache,fetch,telegram,log,post,clock
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/"top_gainers_state.json"
        self.state={"snapshots":[],"completed_slots":[],"reports":[]}
        if self.path.exists():
            loaded=json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded,dict) or not isinstance(loaded.get("snapshots"),list):
                raise ValueError("TOP_GAINERS_INVALID_STATE")
            self.state=loaded
        self.lock=threading.Lock()

    def save(self):
        tmp=self.path.with_suffix(".tmp")
        with tmp.open("w",encoding="utf-8") as f:
            json.dump(self.state,f,ensure_ascii=False,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.path)

    def collect(self, now):
        data=self.cache.snapshot(now)
        prices={}
        for t in data:
            sym=t.get("s","")
            price=number(t.get("c"))
            if sym.endswith("USDT") and price and price>0:
                prices[sym]=price
        return prices

    def step(self):
        if not self.enabled:
            return False
        now=self.clock()
        slot=int(now//INTERVAL)
        with self.lock:
            if slot in self.state.get("completed_slots",[]):
                return False
            prices=self.collect(now)
            if len(prices)<50:
                self.log("[TOP GAINERS] WS snapshot unavailable; retry later")
                return False
            older=self.state["snapshots"][-1] if self.state["snapshots"] else None
            # Mark slot before expensive calls, at-most-once AI / Telegram attempt.
            self.state.setdefault("completed_slots",[]).append(slot)
            self.state["completed_slots"]=self.state["completed_slots"][-100:]
            self.state["snapshots"].append({"at":now,"slot":slot,"prices":prices})
            self.state["snapshots"]=self.state["snapshots"][-8:]
            self.save()
        if not older or slot-older["slot"]!=1:
            self.log("[TOP GAINERS] 4h reference not available; first cycle is warmup")
            return True
        old=older["prices"]
        ranked=sorted(((s,100*(p/old[s]-1)) for s,p in prices.items()
                       if s in old and old[s]>0),key=lambda x:x[1],reverse=True)
        # Pick top ten from active perpetual USDT markets, excluding known non-trading aliases.
        ranked=[x for x in ranked if x[0].isascii() and x[0].endswith("USDT")][:10]
        if len(ranked)<10:
            self.log("[TOP GAINERS] fewer than ten comparable instruments")
            return True
        inputs=[]
        for symbol,ret in ranked:
            try:
                rows=self.fetch("/fapi/v1/klines",{"symbol":symbol,"interval":"1h","limit":60})
                inputs.append(features(symbol,rows,ret,None))
            except Exception as error:
                self.log("[TOP GAINERS KLINES] "+symbol+" "+type(error).__name__)
                inputs.append({"symbol":symbol,"return_4h_pct":round(ret,3),"data":"unavailable"})
        report={"slot":slot,"at":now,"ranking":ranked,"inputs":inputs,
                "status":"NO_AI_KEY" if not self.key else "AI_UNAVAILABLE",
                "analyses":[],"usage":{}}
        if self.key:
            try:
                import requests
                post=self.post or requests.post
                response=post("https://api.openai.com/v1/responses",
                  headers={"Authorization":"Bearer "+self.key},
                  json={"model":self.env.get("TOP_GAINERS_MODEL",MODEL),
                        "store":False,"instructions":INSTRUCTIONS,
                        "input":json.dumps({"as_of_utc":datetime.fromtimestamp(now,timezone.utc).isoformat(),
                                            "gainers":inputs},allow_nan=False),
                        "reasoning":{"effort":"none"},"max_output_tokens":2200,
                        "text":{"format":{"type":"json_schema","name":"gainers_review",
                                          "strict":True,"schema":SCHEMA}}},
                  timeout=(5,60))
                if response.status_code==200:
                    body=response.json()
                    if body.get("status")!="completed":
                        raise ValueError("INCOMPLETE")
                    content="".join(c.get("text","") for o in body.get("output",[])
                        if o.get("type")=="message" for c in o.get("content",[])
                        if c.get("type")=="output_text")
                    parsed=json.loads(content)["analyses"]
                    bysymbol={a["symbol"]:a for a in parsed}
                    expected={s for s,_ in ranked}
                    if len(parsed)!=10 or set(bysymbol)!=expected:
                        raise ValueError("INVALID_SYMBOL_COVERAGE")
                    report["analyses"]=parsed
                    report["usage"]=body.get("usage",{})
                    report["status"]="COMPLETE"
                else:
                    report["status"]="HTTP_"+str(response.status_code)
            except Exception as error:
                report["status"]="AI_"+type(error).__name__
        lines=["🤖 AI TOP GAINERS | 4H | USDT-M FUTURES",
               "Bağımsız araştırma • emir açılmaz",
               "AI durumu: "+report["status"]]
        indexed={a["symbol"]:a for a in report["analyses"]}
        for i,(symbol,ret) in enumerate(ranked,1):
            a=indexed.get(symbol)
            direction={"CONTINUE":"Yükseliş","PULLBACK":"Düzeltme","UNCERTAIN":"Belirsiz"}.get((a or {}).get("bias"),"Veri")
            lines.append(f"{i}. {symbol} {ret:+.2f}% | AI {a['score']}/100 | {direction}" if a else f"{i}. {symbol} {ret:+.2f}% | AI yok")
            if a:
                lines.append(str(a["reason"])[:110]+" | Risk: "+str(a["risk"])[:110])
        lines.append("AI puanı başarı olasılığı değildir. Crash Bounce V2 değişmedi.")
        receipt=self.telegram("\n".join(lines))
        report["telegram"]={k:receipt.get(k) for k in ("ok","error","ambiguous","message_id")}
        with self.lock:
            self.state.setdefault("reports",[]).append(report)
            self.state["reports"]=self.state["reports"][-90:]
            self.save()
        self.log("[TOP GAINERS] "+report["status"]+" telegram="+str(receipt.get("ok")))
        return True

    def start(self):
        if not self.enabled:
            self.log("[TOP GAINERS] OFF")
            return
        def run():
            while True:
                try:self.step()
                except Exception as error:
                    self.log("[TOP GAINERS ERROR] "+type(error).__name__)
                time.sleep(60)
        threading.Thread(target=run,daemon=True,name="ai-top-gainers-report").start()
