"""
On-chain Data Strategy System
===============================
Bu modül on-chain verileri (Coin Metrics) ve fiyat verileri (Binance) kullanarak
3 buy / 3 sell limitleri ile strateji sinyalleri üretir.

Özellikler:
- Coin Metrics API entegrasyonu (on-chain metrikler)
- Binance API entegrasyonu (OHLCV fiyat verileri)
- Z-score tabanlı rejim filtresi
- SMA tabanlı trend filtresi
- Backtest framework
- Pozisyon limitleri (3 buy / 3 sell max)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# 1) COIN METRICS API - On-chain Data
# ==============================================================================

CM_BASE = "https://community-api.coinmetrics.io/v4"

def cm_get(path, params=None):
    """
    Coin Metrics API'ye GET request yapar.
    
    Args:
        path: API endpoint path (örn: "/catalog/assets")
        params: Query parametreleri (dict)
    
    Returns:
        JSON response (dict)
    """
    r = requests.get(f"{CM_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def list_assets(limit=200):
    """
    Kullanılabilir asset listesini döndürür.
    
    Args:
        limit: Maksimum asset sayısı
    
    Returns:
        Asset isimleri listesi (örn: ['btc', 'eth', ...])
    """
    j = cm_get("/catalog/assets", {"limit": limit})
    return [x["asset"] for x in j.get("data", [])]

def list_metrics(limit=2000):
    """
    Kullanılabilir metrik listesini döndürür.
    
    Args:
        limit: Maksimum metrik sayısı
    
    Returns:
        Metrik isimleri listesi (örn: ['AdrActCnt', 'TxCnt', ...])
    """
    j = cm_get("/catalog/metrics", {"limit": limit})
    return [x["metric"] for x in j.get("data", [])]

def get_asset_metrics(asset="btc", metrics=("AdrActCnt", "TxCnt"), 
                      start="2024-01-01", end=None, freq="1d"):
    """
    Belirli bir asset için on-chain metrikleri zaman serisi olarak çeker.
    
    Args:
        asset: Asset ismi (örn: 'btc')
        metrics: Metrik isimleri tuple'ı (örn: ('AdrActCnt', 'TxCnt'))
        start: Başlangıç tarihi (ISO format: '2024-01-01')
        end: Bitiş tarihi (None ise şimdiye kadar)
        freq: Frekans ('1d' günlük)
    
    Returns:
        pandas DataFrame with columns: time, asset, metric values
    """
    params = {
        "assets": asset,
        "metrics": ",".join(metrics),
        "frequency": freq,
        "start_time": start,
    }
    if end:
        params["end_time"] = end
    
    j = cm_get("/timeseries/asset-metrics", params)
    df = pd.DataFrame(j.get("data", []))
    
    if df.empty:
        return df
    
    # metrics value columns are in "values" dict
    values = pd.json_normalize(df["values"])
    out = pd.concat([df.drop(columns=["values"]), values], axis=1)
    out["time"] = pd.to_datetime(out["time"])
    return out.sort_values("time").reset_index(drop=True)

# ==============================================================================
# 2) BINANCE API - Price Data (OHLCV)
# ==============================================================================

def binance_spot_klines(symbol="BTCUSDT", interval="1d", limit=1000):
    """
    Binance Spot API'den OHLCV verilerini çeker.
    
    Args:
        symbol: Trading pair (örn: 'BTCUSDT')
        interval: Zaman aralığı ('1d', '4h', '1h', vb.)
        limit: Maksimum bar sayısı (max: 1000)
    
    Returns:
        pandas DataFrame with columns: time, open, high, low, close, volume
    """
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "num_trades", "tbbav", "tbqav", "ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    
    return df[["open_time", "open", "high", "low", "close", "volume"]].rename(columns={"open_time": "time"})

# ==============================================================================
# 3) SIGNAL GENERATION - Z-score + SMA Strategy
# ==============================================================================

def zscore(s, window=60):
    """
    Rolling z-score hesaplar (standardizasyon).
    
    Args:
        s: pandas Series
        window: Rolling window boyutu
    
    Returns:
        pandas Series with z-scores
    """
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=1)
    return (s - m) / sd

def build_signals(px_df, on_df, on_metric="AdrActCnt", max_positions=3):
    """
    On-chain metrik ve fiyat verilerini birleştirerek trading sinyalleri üretir.
    
    Strategy Logic:
    - Trend: 50 günlük SMA üstü = Long regime, altı = Short regime
    - Rejim Filtresi: On-chain metrik z-score > 0.5 = Risk-on, < -0.5 = Risk-off
    - Long Signal: Trend long + Risk-on
    - Short Signal: Trend short + Risk-off
    - Position Limits: 3 buy / 3 sell max
    
    Args:
        px_df: Price DataFrame (columns: time, close)
        on_df: On-chain DataFrame (columns: time, on_metric)
        on_metric: On-chain metric column name
        max_positions: Maksimum pozisyon sayısı (buy veya sell için)
    
    Returns:
        DataFrame with signals, positions, returns, equity
    """
    # Merge daily data by time
    df = px_df.merge(on_df[["time", on_metric]], on="time", how="inner").copy()
    
    # Calculate indicators
    df["sma50"] = df["close"].rolling(50).mean()
    df["trend"] = np.where(df["close"] > df["sma50"], 1, -1)  # 1 = long regime, -1 = short regime
    df["on_z"] = zscore(df[on_metric].astype(float), 60)
    
    # Regime filter
    df["risk_on"] = df["on_z"] > 0.5
    df["risk_off"] = df["on_z"] < -0.5
    
    # Generate raw signals
    df["signal"] = 0
    df.loc[(df["trend"] == 1) & (df["risk_on"]), "signal"] = 1
    df.loc[(df["trend"] == -1) & (df["risk_off"]), "signal"] = -1
    
    # Apply position limits (3 buy / 3 sell max)
    # Interpretation: Maximum 3 separate long entries and 3 separate short entries
    # Each signal cycle (entry to exit) counts as one trade
    df["position"] = 0.0
    long_entries_used = 0
    short_entries_used = 0
    in_position = False
    
    for i in range(len(df)):
        current_signal = df.iloc[i]["signal"]
        prev_position = df.iloc[i-1]["position"] if i > 0 else 0.0
        
        if current_signal == 1:
            # Long signal
            if not in_position:
                # New entry
                if long_entries_used < max_positions:
                    df.iloc[i, df.columns.get_loc("position")] = 1.0
                    long_entries_used += 1
                    in_position = True
                else:
                    df.iloc[i, df.columns.get_loc("position")] = 0.0
            else:
                # Continue holding
                df.iloc[i, df.columns.get_loc("position")] = prev_position
        elif current_signal == -1:
            # Short signal
            if not in_position:
                # New entry
                if short_entries_used < max_positions:
                    df.iloc[i, df.columns.get_loc("position")] = -1.0
                    short_entries_used += 1
                    in_position = True
                else:
                    df.iloc[i, df.columns.get_loc("position")] = 0.0
            else:
                # Continue holding
                df.iloc[i, df.columns.get_loc("position")] = prev_position
        else:
            # Exit signal
            df.iloc[i, df.columns.get_loc("position")] = 0.0
            in_position = False
    
    # Calculate returns
    df["ret"] = df["close"].pct_change().fillna(0)
    df["strategy_ret"] = df["position"] * df["ret"]
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    
    return df

# ==============================================================================
# 4) BACKTEST FRAMEWORK - Performance Metrics
# ==============================================================================

def backtest_summary(df):
    """
    Backtest sonuçlarının özet istatistiklerini hesaplar.
    
    Args:
        df: Strategy DataFrame (build_signals output)
    
    Returns:
        Dict with performance metrics:
        - total_return: Toplam getiri
        - sharpe_approx: Yaklaşık Sharpe oranı (yıllık)
        - max_drawdown: Maksimum düşüş
        - position_changes: Pozisyon değişim sayısı
        - rows: Toplam gün sayısı
    """
    if df.empty or "equity" not in df.columns:
        return {
            "total_return": 0.0,
            "sharpe_approx": 0.0,
            "max_drawdown": 0.0,
            "position_changes": 0,
            "rows": 0,
        }
    
    total = df["equity"].iloc[-1] - 1
    daily = df["strategy_ret"]
    sharpe = (daily.mean() / (daily.std(ddof=1) + 1e-12)) * np.sqrt(365)
    max_dd = (df["equity"] / df["equity"].cummax() - 1).min()
    trades = (df["position"].diff().abs() > 0).sum()
    
    return {
        "total_return": float(total),
        "sharpe_approx": float(sharpe),
        "max_drawdown": float(max_dd),
        "position_changes": int(trades),
        "rows": int(len(df)),
    }

# ==============================================================================
# 5) MULTI-COIN STRATEGY - Tüm coinler için strateji
# ==============================================================================

def run_multi_coin_strategy(symbols=None, start_date="2024-01-01", on_metric="AdrActCnt"):
    """
    Birden fazla coin için strateji çalıştırır ve sonuçları toplar.
    
    Args:
        symbols: List of (asset, symbol) tuples, örn: [('btc', 'BTCUSDT'), ('eth', 'ETHUSDT')]
                 None ise varsayılan BTC ve ETH kullanılır
        start_date: Başlangıç tarihi
        on_metric: Kullanılacak on-chain metrik
    
    Returns:
        Dict with results for each coin
    """
    if symbols is None:
        symbols = [('btc', 'BTCUSDT'), ('eth', 'ETHUSDT')]
    
    results = {}
    
    for asset, symbol in symbols:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {asset.upper()} / {symbol}")
            print(f"{'='*60}")
            
            # Fetch on-chain data
            print(f"Fetching on-chain data for {asset}...")
            on_df = get_asset_metrics(asset=asset, metrics=(on_metric,), start=start_date)
            
            if on_df.empty:
                print(f"⚠️  No on-chain data available for {asset}")
                continue
            
            # Fetch price data
            print(f"Fetching price data for {symbol}...")
            px_df = binance_spot_klines(symbol=symbol, interval="1d", limit=1000)
            
            if px_df.empty:
                print(f"⚠️  No price data available for {symbol}")
                continue
            
            # Generate signals and backtest
            print(f"Generating signals and running backtest...")
            strategy_df = build_signals(px_df, on_df, on_metric=on_metric, max_positions=3)
            summary = backtest_summary(strategy_df)
            
            # Store results
            results[symbol] = {
                "asset": asset,
                "summary": summary,
                "dataframe": strategy_df
            }
            
            # Print summary
            print(f"\n📊 Results for {symbol}:")
            print(f"  Total Return: {summary['total_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {summary['sharpe_approx']:.2f}")
            print(f"  Max Drawdown: {summary['max_drawdown']*100:.2f}%")
            print(f"  Position Changes: {summary['position_changes']}")
            print(f"  Days: {summary['rows']}")
            
        except Exception as e:
            print(f"❌ Error processing {asset}/{symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

# ==============================================================================
# 6) MAIN EXECUTION - Example Usage
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ON-CHAIN DATA STRATEGY SYSTEM")
    print("=" * 80)
    print("\nBu sistem on-chain verileri (Coin Metrics) ve fiyat verileri (Binance)")
    print("kullanarak 3 buy / 3 sell limitleri ile trading sinyalleri üretir.\n")
    
    # Example 1: List available assets and metrics
    print("\n1️⃣  Mevcut Coin Metrics Assets (ilk 10):")
    try:
        assets = list_assets(limit=10)
        print(f"   {', '.join(assets)}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print("\n2️⃣  Mevcut On-chain Metrics (ilk 20):")
    try:
        metrics = list_metrics(limit=20)
        print(f"   {', '.join(metrics[:20])}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Example 2: Single coin strategy
    print("\n3️⃣  Tek Coin Strateji Örneği (BTC):")
    print("   " + "-" * 60)
    try:
        # Fetch data
        on_df = get_asset_metrics("btc", ("AdrActCnt",), start="2023-01-01")
        px_df = binance_spot_klines("BTCUSDT", "1d", 1000)
        
        # Generate signals
        strategy_df = build_signals(px_df, on_df, "AdrActCnt", max_positions=3)
        summary = backtest_summary(strategy_df)
        
        # Print results
        print(f"\n   📊 BTC Strateji Sonuçları:")
        print(f"   • Total Return: {summary['total_return']*100:.2f}%")
        print(f"   • Sharpe Ratio: {summary['sharpe_approx']:.2f}")
        print(f"   • Max Drawdown: {summary['max_drawdown']*100:.2f}%")
        print(f"   • Position Changes: {summary['position_changes']}")
        print(f"   • Days: {summary['rows']}")
        
        # Show last 5 signals
        print(f"\n   📈 Son 5 Sinyal:")
        signal_cols = ["time", "close", "sma50", "on_z", "signal", "position", "strategy_ret"]
        available_cols = [c for c in signal_cols if c in strategy_df.columns]
        print(strategy_df[available_cols].tail(5).to_string(index=False))
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 3: Multi-coin strategy
    print("\n\n4️⃣  Çoklu Coin Strateji Örneği:")
    print("   " + "-" * 60)
    try:
        symbols = [
            ('btc', 'BTCUSDT'),
            ('eth', 'ETHUSDT'),
        ]
        results = run_multi_coin_strategy(symbols=symbols, start_date="2023-01-01")
        
        print(f"\n\n📊 GENEL ÖZET")
        print("=" * 80)
        for symbol, data in results.items():
            summary = data["summary"]
            print(f"\n{symbol}:")
            print(f"  Return: {summary['total_return']*100:>8.2f}% | Sharpe: {summary['sharpe_approx']:>6.2f} | DD: {summary['max_drawdown']*100:>7.2f}%")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ Program tamamlandı!")
    print("=" * 80)
