"""Read-only, on-demand Telegram coin analysis. Never places orders."""
import json
import os
import re
import time
import requests

SYMBOL = re.compile(r"^[A-Z0-9]{2,20}(?:USDT)?$")
_last = {}

def normalize(value):
    raw = value.strip().upper()
    if not SYMBOL.fullmatch(raw):
        return None
    return raw if raw.endswith("USDT") else raw + "USDT"

def analyze(symbol):
    """Use public CLOSED candles; Gemini only interprets verified market data."""
    if not os.getenv("GEMINI_API_KEY"):
        return "Gemini API anahtarı yapılandırılmamış."
    now = time.monotonic()
    if now - _last.get(symbol, -1e12) < 30:
        return "Aynı coin için 30 saniye bekleyip tekrar deneyin."
    _last[symbol] = now
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": symbol, "interval": "15m", "limit": 121},
                         timeout=(4, 12))
        if r.status_code == 400:
            return symbol + ": USDT-M paritesi bulunamadı."
        r.raise_for_status()
        candles = r.json()
        if not isinstance(candles, list) or len(candles) < 61:
            return symbol + ": analiz için yeterli mum yok."
        epoch = int(time.time() * 1000)
        closed = [{"t":int(c[0]),"o":float(c[1]),"h":float(c[2]),
                   "l":float(c[3]),"c":float(c[4]),"v":float(c[5])}
                  for c in candles if int(c[6]) < epoch]
        if len(closed) < 60:
            return symbol + ": kapanmış mum verisi yetersiz."
        bars = closed[-100:]
        closes = [b["c"] for b in bars]
        gains = [max(closes[i]-closes[i-1],0) for i in range(-14,0)]
        losses = [max(closes[i-1]-closes[i],0) for i in range(-14,0)]
        avg_gain,avg_loss = sum(gains)/14,sum(losses)/14
        rsi = 100 if avg_loss == 0 else 100 - 100/(1+avg_gain/avg_loss)
        vol = bars[-1]["v"]
        avg_vol = sum(b["v"] for b in bars[-21:-1])/20
        payload = {"symbol":symbol,"timeframe":"15m","last_closed_price":closes[-1],
                   "rsi14":round(rsi,2),"rvol20":round(vol/avg_vol,2) if avg_vol else None,
                   "support_20":min(b["l"] for b in bars[-20:]),
                   "resistance_20":max(b["h"] for b in bars[-20:]),
                   "last_closed_candles":bars[-60:]}
        model = os.getenv("CB_AI_GEMINI_MODEL","gemini-2.5-flash-lite")
        if not re.fullmatch(r"[A-Za-z0-9._-]{3,80}",model):
            return "Gemini model ayarı geçersiz."
        prompt = ("Türkçe kısa teknik analiz yap. Yalnızca verilen kapanmış mumlara dayan. "
                  "LONG, SHORT veya BEKLE yönü; formasyon (kanıt yoksa yok); destek, direnç; "
                  "destekten dönüş ve hacimli kırılım için koşullu giriş bölgeleri; "
                  "her uygun giriş için yüzde 1 TP; RSI ve hacim yorumu ver. "
                  "BTC yönü verilmediğinden BTC etkisini bilinmiyor yaz. "
                  "Seviyeleri uydurma, yetersiz veri varsa BEKLE de. "
                  "Bu analiz sanaldır, emir açılmaz. Veri: " + json.dumps(payload,separators=(",",":")))
        result = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"+model+":generateContent",
            headers={"x-goog-api-key":os.environ["GEMINI_API_KEY"]},
            json={"contents":[{"parts":[{"text":prompt}]}],
                  "generationConfig":{"temperature":0.1,"maxOutputTokens":650}},
            timeout=(5,28))
        result.raise_for_status()
        parts = result.json()["candidates"][0]["content"]["parts"]
        answer = "".join(p.get("text","") for p in parts).strip()
        return ("🤖 GEMINI | "+symbol+" | 15m\n"+answer[:3200]+
                "\n📌 Yalnızca analiz; gerçek emir yok.") if answer else "Gemini boş yanıt döndürdü."
    except (requests.RequestException,ValueError,KeyError,IndexError,TypeError):
        return symbol + ": piyasa verisi veya Gemini yanıtı alınamadı. Biraz sonra tekrar deneyin."
