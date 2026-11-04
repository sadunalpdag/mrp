"""Independent four-hour futures rank 11-20 report. Never places trades."""
import json
import math
import os
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

INTERVAL = 4 * 3600
RANK_START = 10   # 0-based, first selected symbol is rank 11
RANK_STOP = 20    # exclusive, last selected symbol is rank 20
MODEL = "gpt-5.4-mini-2026-03-17"
SCHEMA = {"type":"object","additionalProperties":False,
 "properties":{"analyses":{"type":"array","maxItems":10,"items":{"type":"object",
 "additionalProperties":False,"properties":{
 "symbol":{"type":"string"},"score":{"type":"integer","minimum":0,"maximum":100},
 "bias":{"type":"string","enum":["CONTINUE","PULLBACK","UNCERTAIN"]},
 "reason":{"type":"string"},"risk":{"type":"string"},
 "decision":{"type":"string","enum":["BUY","NEUTRAL","AVOID"]},
 "entry_reason":{"type":"string"},
 "entry_price":{"type":["number","null"]}},
 "required":["symbol","score","bias","reason","risk","decision","entry_reason","entry_price"]}}},
 "required":["analyses"]}
INSTRUCTIONS = """You are an independent Binance USDT-M futures market research analyst.
Evaluate ONLY the ten supplied four-hour gainers ranked 11 through 20. Base each claim solely on
provided timestamped data; do not invent news, CVD, funding, OI or real-time data.
Use bull, bear and risk perspectives, then give one concise conclusion per coin.
The score measures subjective continuation strength, NOT win probability.
Consider momentum, RSI, EMA25, EMA50, realized volume, ATR and overextension.
Do not suggest leverage or execute trades. Turkish output, max 110 characters
for each reason and risk. All ten listed symbols must appear exactly once.
Give every coin a BUY (paper LONG candidate), NEUTRAL (wait), or AVOID (avoid LONG,
not an instruction to sell or short) decision and a concise entry_reason.
BUY requires a supported entry_price no higher than supplied market price,
prefer a retracement using EMA25/ATR/support; never invent an entry if data are
insufficient: choose NEUTRAL or AVOID and entry_price=null.
NEUTRAL/AVOID must have entry_price=null. Entry may be at most 10 percent below market.
Do not forecast certainty. If evidence is sparse, use UNCERTAIN and conservative scores."""

def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError, OverflowError):
        return None

def normalize_recommendation(assessment, market_price):
    """Fail closed: only a valid BUY candidate can create a paper LONG."""
    a=dict(assessment)
    decision=a.get("decision")
    why=str(a.get("entry_reason") or "").strip()
    entry=number(a.get("entry_price"))
    market=number(market_price)
    if decision not in ("BUY","NEUTRAL","AVOID"):
        a["decision"]="NEUTRAL"
        a["entry_price"]=None
        a["entry_reason"]="AI kararı doğrulanamadı; LONG giriş önerilmedi."
    elif decision=="BUY" and (market is None or market<=0 or entry is None
                               or not market*0.90 <= entry <= market):
        a["decision"]="NEUTRAL"
        a["entry_price"]=None
        a["entry_reason"]=("Geçerli LONG giriş fiyatı yok. "+why).strip()[:160]
    elif decision=="BUY":
        a["entry_price"]=entry
        a["entry_reason"]=why or "AI, teknik destek ve riske göre LONG giriş önerdi."
    else:
        a["entry_price"]=None
        a["entry_reason"]=why or (
            "Yükselişin zayıflama riski: LONG giriş önerilmiyor." if decision=="AVOID"
            else "Risk veya veri yetersiz: uygun LONG giriş seviyesi yok.")
    return a


def format_top_gainers_report(report, ranked):
    """Compact Telegram report with an explicit decision for every assessed coin."""
    lines=["🤖 AI TOP GAINERS | 4H | RANK 11–20 | USDT-M FUTURES",
           "Sanal araştırma • gerçek emir açılmaz",
           "AI durumu: "+str(report.get("status","UNKNOWN"))]
    news=(report.get("agents") or {}).get("news",{}).get("status","UNKNOWN")
    lines.append("Analistler: Teknik | Haber ("+news+") | Türev | Risk")
    lines.append("Kaynağı/tarihi doğrulanmayan haber karar kanıtı sayılmaz.")
    indexed={a["symbol"]:a for a in report.get("analyses",[])}
    totals={"BUY":0,"NEUTRAL":0,"AVOID":0}
    labels={"BUY":"🟢 AL ADAYI","NEUTRAL":"⚪ NÖTR / BEKLE",
            "AVOID":"🔴 SAT / KAÇIN (LONG açma)"}
    missing=0
    for index,(symbol,ret) in enumerate(ranked,RANK_START+1):
        a=indexed.get(symbol)
        if not a:
            missing+=1
            lines.append(f"{index}. {symbol} {ret:+.2f}% | AI değerlendirmesi yok")
            continue
        decision=a.get("decision","NEUTRAL")
        if decision not in totals:
            decision="NEUTRAL"
        entry=number(a.get("entry_price"))
        if decision=="BUY" and (entry is None or entry<=0):
            decision="NEUTRAL"
        totals[decision]+=1
        trend={"CONTINUE":"Yükseliş","PULLBACK":"Düzeltme",
               "UNCERTAIN":"Belirsiz"}.get(a.get("bias"),"Belirsiz")
        lines.append(f"{index}. {symbol} {ret:+.2f}% | {labels[decision]} | "
                     f"AI {a['score']}/100 | {trend}")
        lines.append("Yorum: "+str(a.get("reason",""))[:50]+
                     " | Risk: "+str(a.get("risk",""))[:40])
        if decision=="BUY":
            lines.append(f"LONG sanal giriş: {entry:.8g} | TP +%1: {entry*1.01:.8g}")
            lines.append("Giriş gerekçesi: "+str(a.get("entry_reason",""))[:50])
        else:
            lines.append("Giriş yok: "+str(a.get("entry_reason") or
                         "AI uygun LONG giriş seviyesi önermedi.")[:62])
    lines.append(f"📊 Kararlar: AL {totals['BUY']} | NÖTR {totals['NEUTRAL']} | "
                 f"SAT/KAÇIN {totals['AVOID']} | Analiz yok {missing}")
    lines.append("SAT/KAÇIN otomatik satış/SHORT değildir. AI puanı başarı olasılığı değildir.")
    lines.append("Crash Bounce V2 ve gerçek emirler değişmedi.")
    message="\n".join(lines)
    if len(message.encode("utf-16-le"))//2>4096:
        raise ValueError("TOP_GAINERS_TELEGRAM_MESSAGE_TOO_LONG")
    return message


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
            "return_1h_pct":round(100*(closes[-1]/closes[-2]-1),3),
            "closed_4h_return_pct":round(100*(closes[-1]/closes[-5]-1),3),
            "ema25_slope_pct":round(100*(ema(closes,25)/ema(closes[:-3],25)-1),3),
            "ema25_1h":round(ema(closes,25),8),
            "ema50_1h":round(ema(closes,50),8) if len(closes)>=50 else None,
            "atr14_pct":round(100*sum(tr[-14:])/14/closes[-1],3),
            "volume_4h_to_prior_24h_hourly":round(sum(volumes[-4:])/4/prevvol,3) if prevvol>0 else None,
            "quote_volume_24h":number(volume_24h)}

class TopGainers:
    def __init__(self, directory, cache, fetch, telegram, log=print,
                 post=None, clock=time.time, env=None, market_symbols=None):
        self.env=os.environ if env is None else env
        # Opt-in only. The main Crash worker never starts this module after
        # the operator disabled AI Gainers; a standalone run is also off by default.
        self.enabled=self.env.get("TOP_GAINERS_ENABLED","0")=="1"
        self.key=self.env.get("OPENAI_API_KEY","").strip()
        self.cache,self.fetch,self.telegram,self.log,self.post,self.clock=cache,fetch,telegram,log,post,clock
        self.market_symbols=market_symbols
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/"top_gainers_state.json"
        self.state={"snapshots":[],"completed_slots":[],"reports":[],
                    "trades":[],"btc":[],"daily":[],"btc_slots":[],"daily_slots":[]}
        if self.path.exists():
            loaded=json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded,dict) or not isinstance(loaded.get("snapshots"),list):
                raise ValueError("TOP_GAINERS_INVALID_STATE")
            self.state=loaded
        for key in ("trades","btc","daily","btc_slots","daily_slots"):
            self.state.setdefault(key,[])
        self.lock=threading.Lock()

    def save(self):
        tmp=self.path.with_suffix(".tmp")
        with tmp.open("w",encoding="utf-8") as f:
            json.dump(self.state,f,ensure_ascii=False,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.path)

    def collect(self, now):
        """Normalize actual CrashBounceWebSocketCache snapshot fields.

        The shared cache exposes symbol/lastPrice, not Binance raw s/c keys.
        Keep raw keys supported for compatibility with other cache stubs.
        """
        data=self.cache.snapshot(now)
        prices={}
        for t in data:
            if not isinstance(t,dict):
                continue
            sym=t.get("symbol") or t.get("s") or ""
            raw_price=t.get("lastPrice")
            if raw_price is None:
                raw_price=t.get("c")
            price=number(raw_price)
            if isinstance(sym,str) and sym.endswith("USDT") and price is not None and price>0:
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
                self.log(f"[TOP GAINERS] WS snapshot unavailable; valid_prices={len(prices)}; retry later")
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
        # Never rank dated contracts or symbols missing from validated active perpetual metadata.
        allowed=set(self.market_symbols() or ()) if self.market_symbols else set(prices)
        eligible=[x for x in ranked if x[0] in allowed and x[0].isascii()
                  and x[0].endswith("USDT")]
        ranked=eligible[RANK_START:RANK_STOP]
        if len(ranked)<RANK_STOP-RANK_START:
            self.log("[TOP GAINERS] fewer than twenty comparable instruments; cannot rank 11-20")
            return True
        inputs=[]
        klines={}
        for symbol,ret in ranked:
            try:
                rows=self.fetch("/fapi/v1/klines",{"symbol":symbol,"interval":"1h","limit":60})
                klines[symbol]=rows
                inputs.append(features(symbol,rows,ret,None))
            except Exception as error:
                self.log("[TOP GAINERS KLINES] "+symbol+" "+type(error).__name__)
                inputs.append({"symbol":symbol,"return_4h_pct":round(ret,3),"data":"unavailable"})
        report={"slot":slot,"at":now,"ranking":ranked,"inputs":inputs,
                "status":"NO_AI_KEY" if not self.key else "AI_UNAVAILABLE",
                "analyses":[],"usage":{}}
        if self.key:
            try:
                from ai_market_agents import AgentsResearch
                recent_btc = next((p.get("direction") for p in reversed(self.state.get("btc", []))
                                   if p.get("status") == "COMPLETE"
                                   and p.get("direction") in ("UP", "DOWN", "SIDEWAYS")
                                   and 0 <= now-p.get("as_of", 0) <= 5400), None)
                engine = AgentsResearch(
                    self.key, model=self.env.get("TOP_GAINERS_MODEL", MODEL),
                    post=self.post, log=self.log,
                    web_enabled=self.env.get("TOP_GAINERS_NEWS_ENABLED", "1") == "1",
                    max_calls_per_scan=5, max_web_calls_per_scan=1)
                outcome = engine.evaluate(
                    inputs, klines=klines, fetch=self.fetch, final_schema=SCHEMA,
                    btc=recent_btc,
                    spot_enabled=self.env.get("TOP_GAINERS_SPOT_DELTA_ENABLED", "0") == "1",
                    market_prices={s: prices[s] for s, _ in ranked})
                report["status"] = outcome["status"]
                report["agents"] = outcome["agents"]
                report["derivative_metrics"] = outcome["derivative_metrics"]
                report["agent_usage"] = outcome["usage"]
                report["estimated_model_usd"] = outcome["estimated_model_usd"]
                report["web_tool_calls"] = outcome["web_tool_calls"]
                report["estimated_search_usd"] = outcome["estimated_search_usd"]
                report["estimated_total_usd"] = outcome["estimated_total_usd"]
                if report["status"] == "COMPLETE":
                    report["analyses"] = [
                        normalize_recommendation(a, prices[a["symbol"]])
                        for a in outcome["analyses"]
                    ]
            except Exception as error:
                self.log("[TOP GAINERS MULTIAGENT ERROR] " + type(error).__name__)
                report["status"] = "AI_" + type(error).__name__
        receipt=self.telegram(format_top_gainers_report(report,ranked))
        report["telegram"]={k:receipt.get(k) for k in ("ok","error","ambiguous","message_id")}
        with self.lock:
            if receipt.get("ok"):
                for a in report["analyses"]:
                    if a.get("decision") != "BUY" or a.get("entry_price") is None:
                        continue
                    entry=float(a["entry_price"])
                    self.state["trades"].append({"id":f"{slot}:{a['symbol']}",
                        "symbol":a["symbol"],"created_at":now,"entry":entry,
                        "tp":entry*1.01,"ai_score":a["score"],"ai_bias":a["bias"],
                        "status":"WAIT_ENTRY","entry_at":None,"tp_at":None,
                        "first_observed_at":None,"last_observed_at":None,
                        "observations":0,"last_price":prices[a["symbol"]]})
            self.state.setdefault("reports",[]).append(report)
            self.state["reports"]=self.state["reports"][-90:]
            self.save()
        self.log("[TOP GAINERS] "+report["status"]+" telegram="+str(receipt.get("ok")))
        return True


    def follow_step(self):
        """Ticker-snapshot paper follow-up. Never imply fills during missing observations."""
        now=self.clock()
        prices=self.collect(now)
        if not prices:
            return
        modified=False
        with self.lock:
            for t in self.state["trades"]:
                if t["status"] not in ("WAIT_ENTRY","OPEN"):
                    continue
                sym=t["symbol"]
                price=prices.get(sym)
                if price is None or now <= t["created_at"]:
                    if t["status"]=="WAIT_ENTRY" and now-t["created_at"]>86400:
                        t.update(status="NO_ENTRY_24H",finished_at=now)
                        modified=True
                    elif t["status"]=="OPEN" and now-t["entry_at"]>86400:
                        t.update(status="TIMEOUT_AFTER_ENTRY_24H",finished_at=now)
                        modified=True
                    continue
                t["observations"]+=1
                t["first_observed_at"]=t["first_observed_at"] or now
                t["last_observed_at"]=now
                t["last_price"]=price
                if t["status"]=="WAIT_ENTRY":
                    if now-t["created_at"]>86400:
                        t.update(status="NO_ENTRY_24H",finished_at=now)
                    elif price<=t["entry"]:
                        t.update(status="OPEN",entry_at=now)
                elif t["status"]=="OPEN":
                    if now-t["entry_at"]>86400:
                        t.update(status="TIMEOUT_AFTER_ENTRY_24H",finished_at=now)
                    elif price>=t["tp"]:
                        t.update(status="TP_1PCT",tp_at=now,finished_at=now)
                modified=True
            if modified:self.save()

    def _openai(self, instructions, payload, schema, name, max_tokens=400):
        if not self.key:
            raise RuntimeError("NO_AI_KEY")
        import requests
        post=self.post or requests.post
        response=post("https://api.openai.com/v1/responses",
            headers={"Authorization":"Bearer "+self.key},
            json={"model":self.env.get("TOP_GAINERS_MODEL",MODEL),
                  "store":False,"instructions":instructions,
                  "input":json.dumps(payload,allow_nan=False),
                  "reasoning":{"effort":"none"},"max_output_tokens":max_tokens,
                  "text":{"format":{"type":"json_schema","name":name,
                                    "strict":True,"schema":schema}}},
            timeout=(5,60))
        if response.status_code!=200:
            raise RuntimeError("HTTP_"+str(response.status_code))
        body=response.json()
        if body.get("status")!="completed":
            raise ValueError("INCOMPLETE")
        content="".join(c.get("text","") for o in body.get("output",[])
            if o.get("type")=="message" for c in o.get("content",[])
            if c.get("type")=="output_text")
        return json.loads(content),body.get("usage",{})

    def btc_step(self):
        """One closed-hour BTC forecast; evaluate only against later observed ticks."""
        now=self.clock(); slot=int(now//3600)
        with self.lock:
            for pred in self.state["btc"]:
                if pred.get("actual_direction") or not pred.get("reference_price"):
                    continue
                if now < pred["as_of"]+3600:
                    continue
                # No retrospective price lookup: if running late, label missing coverage.
                live=self.collect(now).get("BTCUSDT")
                if live is None or now>pred["as_of"]+4500:
                    if now>pred["as_of"]+4500:
                        pred.update(actual_direction="NO_OBSERVATION",correct=None)
                    continue
                change=100*(live/pred["reference_price"]-1)
                actual="UP" if change>0.2 else "DOWN" if change< -0.2 else "SIDEWAYS"
                pred.update(actual_direction=actual,actual_change_pct=round(change,3),
                            correct=actual==pred["direction"])
            if slot in self.state["btc_slots"]:
                return
            # Do not consume an hourly forecast slot before its inputs are ready.
            # Reserve immediately before the paid model call, not at worker startup.
        record={"slot":slot,"as_of":now,"status":"NO_AI_KEY","direction":None,
                "confidence":None,"reason":"","actual_direction":None,"correct":None}
        try:
            rows=self.fetch("/fapi/v1/klines",{"symbol":"BTCUSDT","interval":"1h","limit":60})
            details=features("BTCUSDT",rows,0,None)
            live=self.collect(self.clock()).get("BTCUSDT")
            if "rsi14_1h" not in details or not live:
                # Startup race: retry next minute instead of losing the entire hour.
                self.log("[BTC HOURLY] MISSING_MARKET_DATA; retry later")
                return
            with self.lock:
                if slot in self.state["btc_slots"]:
                    return
                self.state["btc_slots"].append(slot)
                self.state["btc_slots"]=self.state["btc_slots"][-240:]
                self.save()
            if self.key:
                schema={"type":"object","additionalProperties":False,"properties":{
                    "direction":{"type":"string","enum":["UP","DOWN","SIDEWAYS"]},
                    "confidence":{"type":"integer","minimum":0,"maximum":100},
                    "reason":{"type":"string"}},"required":["direction","confidence","reason"]}
                result,usage=self._openai(
                    "Predict BTCUSDT next 1h direction from supplied data only. "
                    "UP/DOWN/SIDEWAYS; neutral band +/-0.2 percent. No certainty; "
                    "confidence is subjective, not calibrated probability. "
                    "Use only supplied closed candles and snapshot indicators. "
                    "Turkish reason <=160 chars. No trading instructions.",
                    {"timestamp_utc":datetime.fromtimestamp(now,timezone.utc).isoformat(),
                     "features":details},schema,"btc_hourly_forecast")
                if len(result["reason"])>160:
                    result["reason"]=result["reason"][:160]
                record.update(status="COMPLETE",direction=result["direction"],
                              confidence=result["confidence"],reason=result["reason"],
                              reference_price=live,usage=usage)
            else:
                record.update(status="NO_AI_KEY",reference_price=live)
        except Exception as error:
            record["status"]="ERROR_"+type(error).__name__
            # Transient REST failures should not burn the hour's only forecast.
            if slot not in self.state["btc_slots"]:
                self.log("[BTC HOURLY] INPUT_ERROR_"+type(error).__name__+"; retry later")
                return
        with self.lock:
            self.state["btc"].append(record)
            self.state["btc"]=self.state["btc"][-720:]
            self.save()
        if record["direction"]:
            self.telegram("₿ BTC AI | 1H\nYön: "+record["direction"]+
                          f" | AI puanı: {record['confidence']}/100\n"+
                          record["reason"]+"\nGelecek 1 saat için gölge tahmin. Emir açılmaz.")
        self.log("[BTC HOURLY] "+record["status"])

    def daily_step(self):
        """Independent 21:00 Europe/Paris report, once per local day."""
        from zoneinfo import ZoneInfo
        now=self.clock()
        local=datetime.fromtimestamp(now,ZoneInfo("Europe/Paris"))
        if local.hour<21:
            return
        day=local.date().isoformat()
        with self.lock:
            if day in self.state["daily_slots"]:
                return
            self.state["daily_slots"].append(day)
            self.state["daily_slots"]=self.state["daily_slots"][-40:]
            self.save()
            since=now-86400
            trades=[t.copy() for t in self.state["trades"] if t["created_at"]>=since]
            finished=[t.copy() for t in self.state["trades"]
                      if t.get("finished_at",0)>=since]
            trailing=[t.copy() for t in self.state["trades"]
                      if t["created_at"]>=now-7*86400]
            btc=[b.copy() for b in self.state["btc"] if b["as_of"]>=since]
            btc7=[b.copy() for b in self.state["btc"] if b["as_of"]>=now-7*86400]
        entered=[t for t in trades if t["entry_at"] is not None]
        resolved=[t for t in finished if t["status"] in ("TP_1PCT","TIMEOUT_AFTER_ENTRY_24H")]
        tp=[t for t in resolved if t["status"]=="TP_1PCT"]
        trailing_entered=[t for t in trailing if t.get("entry_at") is not None]
        noentry=sum(t["status"]=="NO_ENTRY_24H" for t in finished)
        evaluated=[b for b in btc if b.get("correct") is not None]
        correct=sum(1 for b in evaluated if b["correct"])
        evaluated7=[b for b in btc7 if b.get("correct") is not None]
        correct7=sum(1 for b in evaluated7 if b["correct"])
        resolved_pct=f"{100*len(tp)/len(resolved):.1f}%" if resolved else "henüz yok"
        avg_time=(sum((t["tp_at"]-t["entry_at"])/3600 for t in tp)/len(tp)) if tp else None
        avg_text=f"{avg_time:.2f} saat" if avg_time is not None else "henüz yok"
        msg=(f"📊 AI PİYASA GÜNLÜK RAPOR | {day} (Paris)\n"
             f"Yeni LONG aday: {len(trades)} | giriş gözlenen: {len(entered)}\n"
             f"Son 24h sonuçlanan giriş: {len(resolved)} | TP +%1: {len(tp)}\n"
             f"Sonuçlananlarda TP: {resolved_pct} | ort. TP: {avg_text}\n"
             f"Giriş olmadan 24h dolan: {noentry}\n"
             f"Son 7 gün: {len(trailing)} aday | {len(trailing_entered)} gözlenen giriş\n")
        if evaluated:
            msg+=f"BTC 1H (24h): {correct}/{len(evaluated)} doğru ({100*correct/len(evaluated):.1f}%)\n"
        else:
            msg+="BTC 1H (24h): doğrulanabilir tahmin yok\n"
        if evaluated7:
            msg+=f"BTC 1H (7g): {correct7}/{len(evaluated7)} doğru ({100*correct7/len(evaluated7):.1f}%)\n"
        msg+="Yalnız WS örnekleri: kaçırılan giriş/TP olabilir; otomatik işlem yok."
        research_today=[r for r in self.state.get("reports",[])
                        if r.get("at",0)>=since]
        costs=[r.get("estimated_total_usd") for r in research_today]
        spent=sum(c for c in costs if isinstance(c,(int,float)))
        msg+=f"\nTop Gainers AI maliyeti (tahmini): ${spent:.3f} / 24s"
        receipt=self.telegram(msg)
        with self.lock:
            self.state["daily"].append({"day":day,"at":now,"trades":len(trades),
               "entered":len(entered),"tp":len(tp),"resolved":len(resolved),
               "no_entry":noentry,"avg_tp_hours":avg_time,
               "btc_evaluated":len(evaluated),"btc_correct":correct,
               "telegram":{k:receipt.get(k) for k in ("ok","error","ambiguous","message_id")}})
            self.state["daily"]=self.state["daily"][-90:]
            self.save()
        self.log("[AI DAILY] "+day+" delivered="+str(receipt.get("ok")))

    def start(self):
        if not self.enabled:
            self.log("[TOP GAINERS] OFF")
            return
        def run():
            while True:
                research = getattr(self, "_research_thread", None)
                if research is None or not research.is_alive():
                    self._research_thread = threading.Thread(
                        target=self.step, daemon=True, name="top-gainers-4h-research")
                    self._research_thread.start()
                for method in (self.follow_step,self.btc_step,self.daily_step):
                    try:method()
                    except Exception as error:
                        self.log("[TOP GAINERS ERROR] "+method.__name__+" "+type(error).__name__)
                time.sleep(60)
        threading.Thread(target=run,daemon=True,name="ai-top-gainers-report").start()
