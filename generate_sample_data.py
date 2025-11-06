#!/usr/bin/env python3
"""
Generate synthetic market data for backtesting when API is not available
Creates realistic OHLC data with trends and volatility
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_ohlc(bars=2000, base_price=50000, volatility=0.015, trend=0.0001):
    """
    Generate realistic OHLC data with trends and ranging periods
    
    Args:
        bars: Number of candles to generate
        base_price: Starting price
        volatility: Price volatility (0.015 = 1.5%)
        trend: Base trend factor (0.0001 = slight uptrend)
    """
    data = []
    current_time = int((datetime.now() - timedelta(hours=bars)).timestamp() * 1000)
    price = base_price
    
    # Create varying market conditions
    trend_changes = random.randint(3, 6)  # Number of trend changes
    trend_periods = bars // trend_changes
    current_trend = trend
    
    for i in range(bars):
        # Change trend periodically to create realistic market structure
        if i % trend_periods == 0 and i > 0:
            current_trend = random.choice([trend * 3, trend, -trend, -trend * 2, 0])
        
        # Add some cyclical behavior
        cycle = np.sin(i / 100) * volatility * 0.3
        
        # Random walk with trend
        change = np.random.randn() * volatility + current_trend + cycle
        price = price * (1 + change)
        
        # Generate OHLC for this candle with more realistic structure
        open_price = price
        
        # Create candle body
        body_change = np.random.randn() * volatility * 0.6
        close_price = open_price * (1 + body_change)
        
        # Add wicks - more realistic with smaller wicks most of the time
        upper_wick = abs(np.random.randn()) * volatility * 0.4
        lower_wick = abs(np.random.randn()) * volatility * 0.4
        
        high_price = max(open_price, close_price) * (1 + upper_wick)
        low_price = min(open_price, close_price) * (1 - lower_wick)
        
        # Ensure high/low are correct
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Binance kline format: [time, open, high, low, close, volume, close_time, ...]
        candle = [
            current_time,                    # 0: Open time
            str(open_price),                 # 1: Open
            str(high_price),                 # 2: High
            str(low_price),                  # 3: Low
            str(close_price),                # 4: Close
            str(random.uniform(1000, 10000)),# 5: Volume
            current_time + 3600000,          # 6: Close time
            "0",                             # 7: Quote asset volume
            100,                             # 8: Number of trades
            "0",                             # 9: Taker buy base volume
            "0",                             # 10: Taker buy quote volume
            "0"                              # 11: Ignore
        ]
        
        data.append(candle)
        current_time += 3600000  # 1 hour in milliseconds
        price = close_price
    
    return data


def save_sample_data():
    """Generate and save sample data for multiple symbols"""
    symbols_config = [
        ("BTCUSDT", 50000, 0.012, 0.0002),   # Bitcoin - moderate volatility, slight uptrend
        ("ETHUSDT", 3000, 0.015, 0.0001),    # Ethereum - normal volatility
        ("BNBUSDT", 400, 0.018, -0.00005),   # BNB - slightly more volatile, sideways
        ("SOLUSDT", 150, 0.020, 0.0003),     # Solana - more volatile, stronger uptrend
        ("ADAUSDT", 0.5, 0.016, 0.00008),    # Cardano - moderate volatility, slight uptrend
    ]
    
    sample_data = {}
    
    for symbol, base_price, volatility, trend in symbols_config:
        print(f"Generating data for {symbol}...")
        data = generate_realistic_ohlc(
            bars=2200,  # ~3 months of hourly data
            base_price=base_price,
            volatility=volatility,
            trend=trend
        )
        sample_data[symbol] = data
    
    # Save to file
    with open("sample_market_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"\n✅ Sample data saved to sample_market_data.json")
    print(f"   Total symbols: {len(sample_data)}")
    print(f"   Bars per symbol: {len(sample_data['BTCUSDT'])}")
    
    # Print sample statistics
    for symbol in sample_data:
        closes = [float(candle[4]) for candle in sample_data[symbol]]
        start_price = closes[0]
        end_price = closes[-1]
        change_pct = ((end_price / start_price) - 1) * 100
        print(f"   {symbol}: ${start_price:.2f} → ${end_price:.2f} ({change_pct:+.2f}%)")



if __name__ == "__main__":
    save_sample_data()
