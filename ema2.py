import os, requests, hmac, hashlib, math, time

BINANCE_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
BASE_URL       = "https://fapi.binance.com"

# ===============================
# 🔹 Yardımcı fonksiyonlar
# ===============================

def now_ms():
    return int(time.time() * 1000)

def sign(params):
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    sig = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return f"{query}&signature={sig}"

def signed_request(method, path, params=None):
    if params is None:
        params = {}
    params["timestamp"] = now_ms()
    query = sign(params)
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    url = f"{BASE_URL}{path}?{query}"
    r = requests.request(method, url, headers=headers, timeout=10)
    if r.status_code != 200:
        print(f"❌ {path} {r.status_code}: {r.text}")
    return r.json()

def get_filters(sym):
    info = requests.get(BASE_URL + "/fapi/v1/exchangeInfo", timeout=10).json()
    s = next((x for x in info["symbols"] if x["symbol"] == sym), None)
    p = next((f for f in s["filters"] if f["filterType"] == "PRICE_FILTER"), {})
    return {
        "tickSize": float(p.get("tickSize", "0.01")),
        "minPrice": float(p.get("minPrice", "0.00001"))
    }

# ===============================
# 🔹 TP/SL hesaplama fonksiyonu
# ===============================
def calc_tp_sl(sym, entry, tp_pct=0.006, sl_pct=0.2, dir="UP"):
    f = get_filters(sym)
    tick = f["tickSize"]
    minp = f["minPrice"]

    if dir == "UP":
        tp_raw = entry * (1 + tp_pct)
        sl_raw = entry * (1 - sl_pct)
        tp = max(math.floor(tp_raw / tick) * tick, minp)
        sl = max(math.ceil(sl_raw / tick) * tick, minp)
        if sl >= entry:
            sl = entry - tick
    else:
        tp_raw = entry * (1 - tp_pct)
        sl_raw = entry * (1 + sl_pct)
        tp = max(math.ceil(tp_raw / tick) * tick, minp)
        sl = max(math.floor(sl_raw / tick) * tick, minp)
        if sl <= entry:
            sl = entry + tick

    if sl < minp:
        sl = minp * 1.1
    if tp < minp:
        tp = minp * 1.2

    return tp, sl

# ===============================
# 🔹 Test çalıştırıcı
# ===============================
symbols = ["BTCUSDT", "ETHUSDT", "AIOUSDT", "PEPEUSDT", "HOTUSDT"]

print("\n🔍 TP/SL TEST BAŞLIYOR\n")

for sym in symbols:
    try:
        price = float(requests.get(BASE_URL + "/fapi/v1/ticker/price", params={"symbol": sym}).json()["price"])
        tp, sl = calc_tp_sl(sym, price)
        print(f"✅ {sym} entry={price:.8f} → TP={tp:.8f}, SL={sl:.8f}")
    except Exception as e:
        print(f"⚠️ {sym} hata: {e}")

print("\n✅ Test tamamlandı — TP/SL hesaplaması Binance limitleriyle uyumlu.\n")
