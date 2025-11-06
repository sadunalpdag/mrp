# Boring Trading Strategy - Fair Value Gap (FVG) Break

## 📋 Genel Bakış

Bu strateji, YouTube video transkriptinde açıklanan "Boring Strategy"yi (Sıkıcı Strateji) uygular. Strateji, sabırlı ve disiplinli bir yaklaşımla günde 90 dakikadan az sürede tutarlı karlar elde etmeyi hedefler.

**Strateji Adı:** Boring Strategy (Sıkıcı Strateji)  
**Zaman Dilimleri:** 15 dakika ve 5 dakika  
**Çalışma Saatleri:** 9:30 AM - 12:00 PM EST  
**Risk/Ödül Oranı:** 1:2  

## 📊 Beklenen Performans (30 Günlük Backtest)

Transkriptte paylaşılan backtest sonuçlarına göre:
- **Kazanma Oranı:** %81
- **Toplam İşlem:** 16
- **Kazanan İşlem:** 13 (~13/16 = 81%)
- **Kaybeden İşlem:** 3
- **Maksimum Düşüş:** $1,600
- **Toplam Kar:** $15,000

## 🎯 Strateji Kuralları

### Adım 1: Range Belirleme (9:30-9:45 EST)

1. Trading View'da 15 dakikalık grafik açın
2. 9:30-9:45 EST mumunu bekleyin (bu mum 9:30'da başlar, 9:45'te kapanır)
3. Bu mumun **high** (en yüksek) ve **low** (en düşük) seviyelerini işaretleyin
4. Bu, gün içindeki **trading range**'inizi oluşturur

### Adım 2: Yön Teyidi (5 Dakikalık Grafik)

1. 5 dakikalık grafiğe geçin
2. **Fair Value Gap (FVG)** oluşmasını bekleyin
3. FVG, **3 mumluk bir pattern**dir:
   - **Mum 1:** Konsolidasyon
   - **Mum 2:** Güçlü genişleme hareketi
   - **Mum 3:** Devam, Mum 1'in fitili ile Mum 3'ün fitili arasında gap oluşur

#### FVG Tanıma:
- **Bullish FVG:** Mum 3'ün low'u > Mum 1'in high'ı
- **Bearish FVG:** Mum 3'ün high'ı < Mum 1'in low'u

#### Geçerli FVG:
- FVG, 15 dakikalık range'in high veya low seviyesini **kırmalı**
- 3 mumdan en az biri range **içinde**, en az biri range **dışında** kapanmalı
- Bu, FVG pattern'inin range'e dokunduğundan emin olur

### Adım 3: Giriş Ayarları

1. **Limit Order:** FVG'nin ortası (FVG top + FVG bottom) / 2
2. **Stop Loss:** 
   - Bullish için: FVG'nin ilk mumunun low'u
   - Bearish için: FVG'nin ilk mumunun high'ı
3. **Take Profit:** 2:1 risk/ödül oranı
   - TP = Entry + 2 * (Entry - SL) // Bullish için
   - TP = Entry - 2 * (SL - Entry) // Bearish için
4. **Zaman Kısıtı:** Giriş 12:00 PM EST'den önce gerçekleşmeli

## 💻 Kullanım

### Kurulum

```bash
pip install pytz
```

### Örnek Backtest Çalıştırma

```bash
python boring_strategy.py
```

### Python Kodu ile Kullanım

```python
from boring_strategy import BoringStrategy, generate_sample_data

# Stratejiyi başlat
strategy = BoringStrategy()

# Örnek veri üret veya gerçek market verisi kullan
candles_15min, candles_5min = generate_sample_data()

# Backtest çalıştır
result = strategy.backtest(candles_15min, candles_5min)

# Sonuçları göster
print(f"Sonuç: {result['outcome']}")
print(f"Kazanç: {result.get('gain_pct', 0):.2f}%")
```

## 📈 Backtest Çıktısı

Program çalıştırıldığında aşağıdaki gibi detaylı bir çıktı verir:

```
============================================================
BORING STRATEGY - Fair Value Gap (FVG) Break Backtest
============================================================

📊 15-Min Range (9:30-9:45):
   High: $101.50
   Low:  $99.50

🔍 Fair Value Gap Detected:
   Direction: UP
   FVG Top:    $102.50
   FVG Bottom: $100.80
   FVG Mid:    $101.65

📈 Trade Parameters:
   Direction: UP
   Entry:     $101.65
   Stop Loss: $100.30
   Take Profit: $104.35
   Risk:      $1.35
   Reward:    $2.70
   R:R Ratio: 1:2

✅ Trade Execution:
   Entry Filled: Yes
   Outcome: TP
   Exit Price: $104.35
   Gain: 2.66%
```

## 🔍 Önemli Notlar

### Strateji Neden "Sıkıcı"?

1. **Tekrarlanan Kurulum:** Her gün aynı kurulum
2. **Sabır Gerektirir:** Geçerli FVG için bekleme
3. **Basitlik:** Karmaşık göstergeler yok
4. **Disiplin:** Kurallara sıkı bağlılık

### Başarı İçin İpuçları

1. **Range İçinde Sabırlı Olun:** Küçük mum kapanışlarında acele etmeyin
2. **FVG Bekleyin:** Sadece wicks değil, tam 3 mumluk pattern gerekli
3. **Overtrading'den Kaçının:** Günde sadece 1-2 geçerli setup olabilir
4. **Zaman Yönetimi:** 9:45'te hazır olun, 12:00'den önce giriş yapın
5. **Mikroyönetim Yapmayın:** Trade'i kendi haline bırakın

### Yaygın Hatalar

❌ **Sadece wick kırılmasıyla trade açmak** - FVG pattern'i gerekli  
❌ **Range içinde trade açmak** - FVG range'i kırmalı  
❌ **Erken TP almak** - 2:1 oranı koruyun  
❌ **SL'yi hareket ettirmek** - İlk SL'de kalın  
❌ **Sabırsızlık** - Her gün setup olmayabilir  

## 📝 Kod Yapısı

### `BoringStrategy` Class

#### Metodlar:

1. **`get_first_15min_range(candles_15min)`**
   - 9:30-9:45 EST mumunun high/low değerlerini alır
   
2. **`detect_fair_value_gap(candles_5min, start_idx)`**
   - 5 dakikalık mumlarda FVG pattern'i tespit eder
   
3. **`check_fvg_breaks_range(fvg, range_high, range_low)`**
   - FVG'nin range'i kırıp kırmadığını kontrol eder
   
4. **`calculate_trade_params(fvg)`**
   - Entry, SL, TP hesaplar (2:1 risk/reward)
   
5. **`backtest(candles_15min, candles_5min)`**
   - Tam backtest simülasyonu yapar

## 🎓 Öğrenme Kaynakları

Strateji hakkında daha fazla bilgi için:
- Video transkripti (problem statement'ta)
- Fair Value Gap (FVG) kavramı
- Price Action Trading temelleri
- Risk Management prensipleri

## ⚠️ Risk Uyarısı

Bu strateji eğitim amaçlıdır. Gerçek para ile trading yapmadan önce:
- Demo hesapta pratik yapın
- Risk yönetimini öğrenin
- Sadece kaybedebileceğiniz para ile trade yapın
- Profesyonel finansal danışmanlık alın

**Geçmiş performans gelecekteki sonuçları garanti etmez.**

## 📄 Lisans

Bu kod eğitim ve araştırma amaçlıdır.

## 🤝 Katkı

Geliştirmeler ve öneriler için pull request gönderilebilir.

---

**Önemli:** Bu strateji, video transkriptinden alınan bilgilere dayanmaktadır. Gerçek trading ortamında kullanmadan önce kapsamlı testler yapılmalıdır.
