# EMA-Structure Stratejisi - Telegram Mesaj Formatı

## 📊 Telegram'a Gönderilen Mesaj Örnekleri

### LONG (BUY) Sinyali:

```
📊 EMA-STRUCTURE BUY BTCUSDT UP qty:0.01
Power:65.30
Entry:43250.50000000
TP hedefi:1.60$ (0.640%)
time:2025-11-06T23:34:00+03:00
```

### SHORT (SELL) Sinyali:

```
📊 EMA-STRUCTURE SELL ETHUSDT DOWN qty:0.1
Power:68.45
Entry:2250.75000000
TP hedefi:1.70$ (0.680%)
time:2025-11-06T23:35:00+03:00
```

## 🔍 Mesaj Bileşenleri

| Bileşen | Açıklama |
|---------|----------|
| `📊 EMA-STRUCTURE BUY/SELL` | Strateji etiketi (tag) |
| `BTCUSDT` | İşlem yapılan sembol |
| `UP/DOWN` | İşlem yönü (direction) |
| `qty:0.01` | İşlem miktarı (quantity) |
| `Power:65.30` | Güç skoru (power score: 60-100) |
| `Entry:43250.50000000` | Giriş fiyatı (entry price) |
| `TP hedefi:1.60$` | Take Profit hedefi (USD) |
| `(0.640%)` | TP yüzdesi |
| `time:2025-11-06T23:34:00` | İşlem zamanı (timestamp) |

## ⚠️ Önemli Notlar

### ✅ Stop Loss Yok
- EMA-Structure stratejisi artık **STOP LOSS kullanmıyor**
- Sadece **Take Profit (TP)** var
- TP: Sabit %0.6 hedef (diğer stratejiler gibi)

### 📊 Strateji Özellikleri
- **Emoji**: 📊 (EMA-Structure'a özel)
- **Tag**: "📊 EMA-STRUCTURE BUY" veya "📊 EMA-STRUCTURE SELL"
- **Power Score**: 60-100 arası
  - Base: 60
  - EMA50 momentum: +150 max
  - RSI deviation: ±25 max
  - Confirmation candle bonus: +5

### 🎯 Telegram'da Görünmeyen Bilgiler

Signal objesinde şunlar da var (sadece log dosyalarında):

```json
{
  "has_confirmation": true,
  "touched_ema": true,
  "rsi": 58.2,
  "atr": 125.50,
  "kind": "EMA_STRUCTURE",
  "tier": "EMA_STRUCTURE"
}
```

Bu bilgiler şuralarda saklanır:
- `ai_signals.json` - Tüm sinyaller
- `sim_positions.json` - Simülasyon pozisyonları
- `ai_rl_log.json` - Gerçek işlem kayıtları

## 📱 Diğer Stratejilerle Karşılaştırma

| Strateji | Emoji | Tag | Stop Loss |
|----------|-------|-----|-----------|
| UTSTC | 🟢 | UT/STC BUY/SELL | Var |
| MACD | 📈 | EMA/MACD BUY/SELL | Var |
| FVG | 🟩 | FVG BREAK BUY/SELL | Var |
| PULLBACK | 📘 | EMA PULLBACK BUY/SELL | Var |
| KIVANC | 🧩 | KIVANC BUY/SELL | Var |
| **EMA-STRUCTURE** | **📊** | **EMA-STRUCTURE BUY/SELL** | **YOK** ✅ |

## 🔔 Nasıl Tanırsınız?

EMA-Structure sinyallerini şunlardan tanıyabilirsiniz:

1. **Emoji**: 📊 (Chart with increasing bars)
2. **Tag**: "EMA-STRUCTURE" içerir
3. **Stop Loss mesajı yok** (sadece TP var)
4. **Power genelde 60-75 arası** (seçici strateji)

## 📊 İstatistik Takibi

Telegram'dan `/status` komutuyla:
```
📊 STATUS bar:1234 auto:✅ long:3 short:2 
sim_open:5 sim_closed:45
```

Telegram'dan `/report` komutuyla:
- `ai_signals.json` - Tüm EMA-Structure sinyalleri
- `ai_analysis.json` - `ema_structure_signals_total` sayacı
- `sim_closed.json` - Kapalı EMA-Structure işlemleri

## 🎯 Örnek Senaryo

1. Bot EMA-Structure kurulumu tespit eder
2. Telegram'a sinyal gönderir:
   ```
   📊 EMA-STRUCTURE BUY BTCUSDT UP qty:0.01
   Power:65.30
   Entry:43250.50000000
   TP hedefi:1.60$ (0.640%)
   time:2025-11-06T23:34:00+03:00
   ```
3. İşlem açılır (market order)
4. Take Profit emri yerleştirilir
5. **Stop Loss YOK** - sadece TP ile çalışır
6. TP'ye ulaştığında kapatılır
7. 6 saatlik TrendLock devreye girer

---

**Version**: v15.9.53  
**Last Updated**: 2025-11-06
