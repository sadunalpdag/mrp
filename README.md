# MRP - Multi-Strategy Trading Bot

Python ile yazılmış otomatik trading botu

## Özellikler

### 📊 Strateji Yönetimi
- 12 farklı aktif strateji
- Strateji bazlı limit kontrolleri
- Telegram üzerinden strateji aktif/pasif yapma

### ⏰ Saatlik Performans Takibi (YENİ!)
- Saatlere göre otomatik performans analizi
- 2 haftalık öğrenme periyodu
- Kötü performans gösteren saatlerde otomatik kapanma
- Telegram komutları ile manuel kontrol

Detaylı bilgi için: [HOURLY_ANALYSIS.md](HOURLY_ANALYSIS.md)

### 💰 Kar Hedefi Yönetimi
- Otomatik kar hedefi takibi
- Hedefe ulaşınca tüm pozisyonları kapatma
- Saatlik ilerleme raporları

### 📈 Gelişmiş Durum Raporlaması
- Strateji bazlı açık pozisyon sayıları
- Hedefe olan mesafe gösterimi
- Detaylı performans istatistikleri

## Telegram Komutları

### Temel Komutlar
- `/status` - Bot durumu ve detaylı pozisyon bilgisi
- `/balance` - Bakiye ve kar durumu
- `/strategies` - Tüm stratejiler ve durumları
- `/closeall` - Tüm pozisyonları kapat

### Saatlik Analiz Komutları (YENİ!)
- `/hourlystats` - Saatlik performans istatistikleri
- `/blockhour <saat> [block|unblock]` - Saati blokla/aç
- `/resethourlystats` - İstatistikleri sıfırla
- `/forcehourlyanalysis` - Analizi hemen aktifleştir

### Yapılandırma
- `/set <parametre> <değer>` - Parametre değiştir
- `/setlimits <tip> <değer>` - Limit ayarla
- `/enable <strateji>` - Stratejiyi aç
- `/disable <strateji>` - Stratejiyi kapat

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python ema.py
```

## Gerekli Çevre Değişkenleri

```
BOT_TOKEN=<telegram_bot_token>
CHAT_ID=<telegram_chat_id>
BINANCE_API_KEY=<binance_api_key>
BINANCE_SECRET_KEY=<binance_secret_key>
DATA_DIR=./data
```
