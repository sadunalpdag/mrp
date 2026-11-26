# On-Chain Data Strategy System

Bu modül, on-chain verileri (Coin Metrics) ve fiyat verileri (Binance) kullanarak 3 buy / 3 sell limitleri ile strateji sinyalleri üretir.

## Özellikler

- ✅ **Coin Metrics API Entegrasyonu**: On-chain metrikler (AdrActCnt, TxCnt, vb.)
- ✅ **Binance API Entegrasyonu**: OHLCV fiyat verileri
- ✅ **Z-score Rejim Filtresi**: On-chain aktivite normalizasyonu
- ✅ **SMA Trend Filtresi**: 50 günlük moving average ile trend belirleme
- ✅ **Pozisyon Limitleri**: 3 buy / 3 sell maksimum
- ✅ **Backtest Framework**: Sharpe, drawdown, return metrikleri
- ✅ **Çoklu Coin Desteği**: Birden fazla coin için paralel çalışma

## Kurulum

```bash
pip install -r requirements.txt
```

Gerekli kütüphaneler:
- pandas
- numpy
- requests

## Kullanım

### 1. Temel Kullanım

```python
from onchain_strategy import (
    get_asset_metrics,
    binance_spot_klines,
    build_signals,
    backtest_summary
)

# On-chain data çek
on_df = get_asset_metrics("btc", ("AdrActCnt",), start="2023-01-01")

# Fiyat datası çek
px_df = binance_spot_klines("BTCUSDT", "1d", 1000)

# Sinyal üret ve backtest yap
strategy_df = build_signals(px_df, on_df, "AdrActCnt", max_positions=3)
summary = backtest_summary(strategy_df)

print(f"Total Return: {summary['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {summary['sharpe_approx']:.2f}")
print(f"Max Drawdown: {summary['max_drawdown']*100:.2f}%")
```

### 2. Çoklu Coin Stratejisi

```python
from onchain_strategy import run_multi_coin_strategy

# Birden fazla coin için strateji çalıştır
symbols = [
    ('btc', 'BTCUSDT'),
    ('eth', 'ETHUSDT'),
    ('bnb', 'BNBUSDT'),
]

results = run_multi_coin_strategy(symbols=symbols, start_date="2023-01-01")

# Sonuçları görüntüle
for symbol, data in results.items():
    summary = data["summary"]
    print(f"{symbol}: Return={summary['total_return']*100:.2f}%, "
          f"Sharpe={summary['sharpe_approx']:.2f}")
```

### 3. Komut Satırından Çalıştırma

```bash
# Örnek kullanım ile çalıştır
python onchain_strategy.py

# Test suite çalıştır
python test_onchain_strategy.py
```

## Strateji Mantığı

### Trend Belirleme
- **Long Regime**: Fiyat > 50 günlük SMA
- **Short Regime**: Fiyat < 50 günlük SMA

### Rejim Filtresi (On-chain Z-score)
- **Risk-On**: Z-score > 0.5 (aktivite ortalamanın üstünde)
- **Risk-Off**: Z-score < -0.5 (aktivite ortalamanın altında)

### Sinyal Üretimi
- **Long Signal**: Trend = Long + Risk-On
- **Short Signal**: Trend = Short + Risk-Off
- **Exit Signal**: Diğer durumlar

### Pozisyon Limitleri
- Maksimum 3 long pozisyon
- Maksimum 3 short pozisyon
- Limit dolunca yeni pozisyon açılmaz

## API Kaynakları

### Coin Metrics API
- **Base URL**: `https://community-api.coinmetrics.io/v4`
- **Rate Limit**: Community tier için sınırlı
- **Dokümantasyon**: https://docs.coinmetrics.io/

#### Kullanılabilir Metrikler
- `AdrActCnt`: Active addresses count
- `TxCnt`: Transaction count
- `FlowInExNtv`: Exchange inflow
- `FlowOutExNtv`: Exchange outflow
- `NVTAdj`: Network value to transactions ratio
- Ve daha fazlası...

### Binance API
- **Spot API**: `https://api.binance.com/api/v3`
- **Futures API**: `https://fapi.binance.com/fapi/v1`
- **Rate Limit**: Weight-based limiting
- **Dokümantasyon**: https://binance-docs.github.io/apidocs/

## Backtest Metrikleri

### Total Return
Toplam getiri oranı (başlangıç sermayesine göre).

### Sharpe Ratio
Risk-adjusted return metriği. Yüksek değer daha iyi.
- **> 1.0**: İyi
- **> 2.0**: Çok iyi
- **> 3.0**: Mükemmel

### Max Drawdown
En büyük sermaye düşüşü (peak-to-trough). Negatif değer, risk göstergesi.

### Position Changes
Toplam pozisyon değişim sayısı (açılış/kapanış).

## Örnek Çıktı

```
================================================================================
ON-CHAIN DATA STRATEGY SYSTEM
================================================================================

Processing BTC / BTCUSDT
============================================================
Fetching on-chain data for btc...
Fetching price data for BTCUSDT...
Generating signals and running backtest...

📊 Results for BTCUSDT:
  Total Return: 45.23%
  Sharpe Ratio: 1.85
  Max Drawdown: -12.34%
  Position Changes: 28
  Days: 365

📈 Son 5 Sinyal:
      time        close        sma50  on_z  signal  position  strategy_ret
2024-01-01  42000.00  41500.00  0.85       1       1.0      0.0123
2024-01-02  42500.00  41550.00  0.92       1       1.0      0.0119
2024-01-03  42800.00  41600.00  1.05       1       1.0      0.0071
2024-01-04  42600.00  41650.00  0.88       1       1.0     -0.0047
2024-01-05  43000.00  41700.00  0.95       1       1.0      0.0094
```

## Notlar

- ⚠️ **Bu bir demo/araştırma kodudur**. Production kullanımı için ek güvenlik, hata yönetimi ve optimizasyon gereklidir.
- ⚠️ **API rate limitlerine** dikkat edin. Çok fazla istek yapmayın.
- ⚠️ **Backtest sonuçları** geçmiş performansı gösterir, gelecek garantisi değildir.
- ⚠️ **Paper trading** ile test etmeden gerçek parayla kullanmayın.

## Lisans

MIT License - Eğitim ve araştırma amaçlı kullanım için.

## İletişim

Sorular ve öneriler için repository'nin issue bölümünü kullanabilirsiniz.
