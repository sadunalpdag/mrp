# Backtest for New Trading Strategies

Bu backtest scripti, yeni eklenen 3 stratejiyi (LO_ORB, NYR, ICT_P3) test eder.

## Stratejiler

1. **LO_ORB (London Breakout)**: Londra açılışındaki (08:00-08:30 GMT) range kırılımı stratejisi
2. **NYR (New York Reversal)**: NY açılışında (13:00-14:00 GMT) liquidity sweep ve reversal stratejisi  
3. **ICT_P3 (ICT Power of 3)**: Accumulation, Manipulation, Distribution stratejisi

## Kullanım

### Tam Backtest (Tüm coinler, 3 ay)
```bash
python3 backtest_new_strategies.py
```

Bu komut:
- Binance Futures'taki tüm USDT paritelerini test eder
- 3 aylık (90 gün) geçmiş verilerle backtest yapar
- Sonuçları `backtest_results_new_strategies.json` ve `backtest_trades_new_strategies.json` dosyalarına kaydeder

### Hızlı Test (İlk 20 coin, 1 hafta)
```bash
python3 backtest_new_strategies.py --quick
```

Bu komut:
- İlk 20 coini test eder
- 1 haftalık verilerle hızlı test yapar
- Stratejilerin çalıştığını doğrulamak için idealdir

### Belirli Coinleri Test Etme
```bash
python3 backtest_new_strategies.py --symbols BTC ETH BNB
```

veya

```bash
python3 backtest_new_strategies.py --symbols BTCUSDT ETHUSDT BNBUSDT
```

### Özel Zaman Aralığı
```bash
python3 backtest_new_strategies.py --days 30  # 30 günlük backtest
```

## Çıktı Dosyaları

### backtest_trades_new_strategies.json
Tüm işlemlerin detaylı listesi:
- Giriş/çıkış fiyatları
- TP/SL seviyeleri
- Kâr/zarar yüzdeleri
- İşlem süreleri
- Strateji bilgileri

### backtest_results_new_strategies.json
İstatistiksel analiz:
- Toplam işlem sayısı
- Kazanma oranı (win rate)
- Ortalama kâr/zarar
- Strateji bazlı performans
- Yön bazlı (long/short) performans

## Örnek Çıktı

```
================================================================================
BACKTEST RESULTS - NEW STRATEGIES (LO_ORB, NYR, ICT_P3)
================================================================================

Overall Statistics:
  Total Trades: 150
  Win Rate: 55.33%
  Total PnL: 45.2%
  Average PnL per Trade: 0.301%
  Average Winner: 0.6%
  Average Loser: -0.5%
  Average Duration: 12.5 hours

By Strategy:

  LO_ORB:
    Trades: 60 (W:35, L:25)
    Win Rate: 58.33%
    Total PnL: 21.5%
    Avg PnL: 0.358%
    Avg Winner: 0.65%
    Avg Loser: -0.48%
    Avg Duration: 10.2 hours

  NYR:
    Trades: 45 (W:25, L:20)
    Win Rate: 55.56%
    Total PnL: 12.3%
    Avg PnL: 0.273%
    Avg Winner: 0.58%
    Avg Loser: -0.52%
    Avg Duration: 14.8 hours

  ICT_P3:
    Trades: 45 (W:23, L:22)
    Win Rate: 51.11%
    Total PnL: 11.4%
    Avg PnL: 0.253%
    Avg Winner: 0.62%
    Avg Loser: -0.51%
    Avg Duration: 13.1 hours

================================================================================
```

## Gereksinimler

- Python 3.8+
- requests kütüphanesi
- ema.py (strateji fonksiyonları)
- İnternet bağlantısı (Binance API'ye erişim için)

## Notlar

- ema.py dosyasında hiçbir değişiklik yapmaz
- Backtest geçmiş verilerle yapılır, gerçek işlem yapmaz
- Sonuçlar yalnızca bilgilendirme amaçlıdır
- API rate limit'leri nedeniyle büyük testler zaman alabilir (birkaç saat)
- Backtest sonuçları gelecekteki performansı garanti etmez

## Sorun Giderme

### "No module named 'ema'" hatası
```bash
cd /home/runner/work/mrp/mrp
python3 backtest_new_strategies.py
```

### API rate limit hatası
`--quick` modunu kullanın veya daha az coin test edin:
```bash
python3 backtest_new_strategies.py --quick
```

### İnternet bağlantısı yok
Fallback sembol listesi otomatik olarak kullanılır (15 popüler coin).
