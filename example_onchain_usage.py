#!/usr/bin/env python3
"""
Example Usage: On-chain Strategy System
========================================
Bu dosya onchain_strategy modülünün kullanım örneklerini gösterir.
"""

import sys
from datetime import datetime, timedelta

# Import functions from onchain_strategy
from onchain_strategy import (
    list_assets,
    list_metrics,
    get_asset_metrics,
    binance_spot_klines,
    build_signals,
    backtest_summary,
    run_multi_coin_strategy
)

def example_1_list_available_data():
    """Example 1: Kullanılabilir asset ve metrikleri listele"""
    print("\n" + "="*80)
    print("ÖRNEK 1: Kullanılabilir Data Kaynakları")
    print("="*80)
    
    try:
        # Coin Metrics'te mevcut assetleri listele
        print("\n📋 İlk 20 Asset:")
        assets = list_assets(limit=20)
        print(f"   {', '.join(assets)}")
        
        # Mevcut metrikleri listele
        print("\n📋 İlk 30 On-chain Metrik:")
        metrics = list_metrics(limit=30)
        for i, metric in enumerate(metrics[:30], 1):
            if i % 5 == 0:
                print(f"   {metric}")
            else:
                print(f"   {metric}", end=", ")
        print()
        
    except Exception as e:
        print(f"⚠️  API erişimi başarısız (sandbox ortamında beklenen durum): {e}")
        print("   Production ortamında API'ler çalışacaktır.")

def example_2_single_asset_strategy():
    """Example 2: Tek bir asset için strateji"""
    print("\n" + "="*80)
    print("ÖRNEK 2: BTC için Tek Asset Stratejisi")
    print("="*80)
    
    try:
        asset = "btc"
        symbol = "BTCUSDT"
        on_metric = "AdrActCnt"
        start_date = "2023-01-01"
        
        print(f"\n🔍 {asset.upper()} için veri çekiliyor...")
        print(f"   Metric: {on_metric}")
        print(f"   Start Date: {start_date}")
        
        # 1. On-chain data çek
        print(f"\n   Step 1: On-chain data (Coin Metrics)")
        on_df = get_asset_metrics(asset, (on_metric,), start=start_date)
        print(f"   ✓ {len(on_df)} günlük on-chain data alındı")
        
        # 2. Fiyat datası çek
        print(f"\n   Step 2: Fiyat datası (Binance)")
        px_df = binance_spot_klines(symbol, "1d", 1000)
        print(f"   ✓ {len(px_df)} günlük fiyat datası alındı")
        
        # 3. Strateji sinyalleri üret
        print(f"\n   Step 3: Sinyal üretimi ve backtest")
        strategy_df = build_signals(px_df, on_df, on_metric, max_positions=3)
        summary = backtest_summary(strategy_df)
        
        # 4. Sonuçları göster
        print(f"\n📊 BACKTEST SONUÇLARI")
        print(f"   {'─'*60}")
        print(f"   Total Return:     {summary['total_return']*100:>8.2f}%")
        print(f"   Sharpe Ratio:     {summary['sharpe_approx']:>8.2f}")
        print(f"   Max Drawdown:     {summary['max_drawdown']*100:>8.2f}%")
        print(f"   Position Changes: {summary['position_changes']:>8d}")
        print(f"   Total Days:       {summary['rows']:>8d}")
        
        # 5. Son sinyalleri göster
        print(f"\n📈 SON 10 SİNYAL")
        print(f"   {'─'*60}")
        cols = ['time', 'close', 'sma50', 'on_z', 'signal', 'position']
        available_cols = [c for c in cols if c in strategy_df.columns]
        print(strategy_df[available_cols].tail(10).to_string(index=False))
        
    except Exception as e:
        print(f"\n⚠️  Hata: {e}")
        print("   Sandbox ortamında API erişimi kısıtlıdır.")
        print("   Production ortamında bu örnek çalışacaktır.")

def example_3_multiple_assets():
    """Example 3: Çoklu asset stratejisi"""
    print("\n" + "="*80)
    print("ÖRNEK 3: Çoklu Asset Portföy Stratejisi")
    print("="*80)
    
    try:
        symbols = [
            ('btc', 'BTCUSDT'),
            ('eth', 'ETHUSDT'),
            ('bnb', 'BNBUSDT'),
        ]
        
        print(f"\n🔍 {len(symbols)} asset için strateji çalıştırılıyor...")
        results = run_multi_coin_strategy(symbols=symbols, start_date="2023-01-01")
        
        # Sonuçları tablo halinde göster
        print(f"\n📊 PORTFÖY SONUÇLARI")
        print(f"   {'='*80}")
        print(f"   {'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Trades':>8}")
        print(f"   {'-'*80}")
        
        for symbol, data in results.items():
            s = data['summary']
            print(f"   {symbol:<12} {s['total_return']*100:>9.2f}% "
                  f"{s['sharpe_approx']:>8.2f} {s['max_drawdown']*100:>9.2f}% "
                  f"{s['position_changes']:>8d}")
        
        print(f"   {'='*80}")
        
    except Exception as e:
        print(f"\n⚠️  Hata: {e}")
        print("   Sandbox ortamında API erişimi kısıtlıdır.")
        print("   Production ortamında bu örnek çalışacaktır.")

def example_4_custom_parameters():
    """Example 4: Özel parametrelerle strateji"""
    print("\n" + "="*80)
    print("ÖRNEK 4: Özel Parametrelerle Strateji")
    print("="*80)
    
    print("\n🔧 Farklı on-chain metrikleri kullanarak karşılaştırma:")
    print("   " + "-"*70)
    
    try:
        metrics_to_test = [
            "AdrActCnt",   # Active addresses
            "TxCnt",       # Transaction count
            "FlowInExNtv", # Exchange inflow
        ]
        
        asset = "btc"
        symbol = "BTCUSDT"
        
        print(f"\n   Asset: {asset.upper()}/{symbol}")
        print(f"\n   {'Metric':<20} {'Return':>10} {'Sharpe':>8} {'Trades':>8}")
        print(f"   {'-'*70}")
        
        for metric in metrics_to_test:
            try:
                on_df = get_asset_metrics(asset, (metric,), start="2023-01-01")
                px_df = binance_spot_klines(symbol, "1d", 500)
                strategy_df = build_signals(px_df, on_df, metric, max_positions=3)
                summary = backtest_summary(strategy_df)
                
                print(f"   {metric:<20} {summary['total_return']*100:>9.2f}% "
                      f"{summary['sharpe_approx']:>8.2f} {summary['position_changes']:>8d}")
            except Exception as e:
                print(f"   {metric:<20} {'N/A':>9}  {'N/A':>8}  {'N/A':>8}  ({e})")
        
        print(f"   {'-'*70}")
        
    except Exception as e:
        print(f"\n⚠️  Hata: {e}")
        print("   Sandbox ortamında API erişimi kısıtlıdır.")

def main():
    """Ana fonksiyon - Tüm örnekleri çalıştır"""
    print("\n" + "="*80)
    print("ON-CHAIN STRATEGY SYSTEM - KULLANIM ÖRNEKLERİ")
    print("="*80)
    print("\nBu dosya onchain_strategy modülünün çeşitli kullanım örneklerini gösterir.")
    print("Sandbox ortamında API erişimi kısıtlıdır, ancak kod yapısı gösterilmiştir.")
    
    # Örnekleri sırayla çalıştır
    example_1_list_available_data()
    example_2_single_asset_strategy()
    example_3_multiple_assets()
    example_4_custom_parameters()
    
    print("\n" + "="*80)
    print("✅ TÜM ÖRNEKLER TAMAMLANDI")
    print("="*80)
    print("\n💡 İpucu: Production ortamında bu örnekler gerçek API'lerden veri çekecektir.")
    print("💡 İpucu: Kendi stratejinizi oluşturmak için build_signals() fonksiyonunu inceleyin.")
    print("💡 İpucu: Detaylı dokümantasyon için ONCHAIN_STRATEGY_README.md dosyasına bakın.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program kullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
