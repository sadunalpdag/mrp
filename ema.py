EMA ULTRA ema.py dosyasına mevcut hiçbir logic’i silmeden yeni bir `detect_distribution_pattern()` fonksiyonu ekle.

Amaç:
Pump sonrası blow-off top / MM distribution / exit liquidity yapısını tespit etmek.

Pattern şartları:

* Son 6-12 mum içinde güçlü impulse (%12+ veya config ile ayarlanabilir)
* Tepe mumunda anlamlı upper wick
* Hacim spike (ortalamanın üstünde)
* Sonraki 1-3 mumda kırmızı follow-through
* Peak’ten belirgin rejection

Return:
{
"detected": bool,
"score": int,
"state": "DISTRIBUTION" | "NONE",
"bias": "LONG_EXIT" | "SHORT_BIAS" | "NEUTRAL",
"action": "NO_LONG" | "WAIT_PULLBACK" | "LOOK_FOR_SHORT_CONFIRMATION",
"impulse_pct": float,
"peak_price": float,
"last_price": float,
"rejection_pct": float,
"wick_ratio": float,
"volume_ratio": float,
"reason": str
}

Entegrasyon:

* Distribution varsa yeni long açma
* Breakout long sinyali varsa baskıla
* State akışına `OVEREXTENDED -> DISTRIBUTION` veya `BREAKOUT_PENDING -> FAKE_BREAKOUT_LONG` benzeri uyumlu geçiş ekle
* Log üret

Örnek log:
🚫 DISTRIBUTION DETECTED — ENJUSDT
Bias: LONG_EXIT | State: OVEREXTENDED → DISTRIBUTION
Peak: 0.07317 | Last: 0.05285
Impulse: %64.9 | Rejection: %27.8
WickRatio: 2.8 | VolRatio: 3.9
Action: NO_LONG / LOOK_FOR_SHORT_CONFIRMATION

Kurallar:

* Hiçbir mevcut fonksiyon silinmeyecek
* Mevcut stratejiler korunacak
* Placeholder yok
* Tam implementasyon ver
* ema.py stiline sadık kal
