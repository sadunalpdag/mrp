import hmac, hashlib, time, requests

# ===============================
# 🔑 BURAYA API BİLGİLERİNİ GİR
# ===============================
API_KEY    = "YOUR_API_KEY_HERE"
API_SECRET = "YOUR_SECRET_KEY_HERE"

# Binance Futures endpoint
BASE_URL = "https://fapi.binance.com"

def signed_request(method, path, params=None):
    if params is None:
        params = {}
    params['timestamp'] = int(time.time() * 1000)
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(API_SECRET.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    query += f"&signature={signature}"
    headers = {"X-MBX-APIKEY": API_KEY}
    r = requests.request(method, f"{BASE_URL}{path}?{query}", headers=headers)
    return r

def check_futures_connection():
    print("🔍 Checking Binance Futures API access...")
    resp = signed_request("GET", "/fapi/v2/account")
    if resp.status_code == 200:
        print("✅ Futures API connection successful!")
        data = resp.json()
        print(f"Total wallet balance: {data['totalWalletBalance']} USDT")
    else:
        print("❌ Connection failed!")
        print(f"HTTP {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    check_futures_connection()
