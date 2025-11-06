"""
Example: How to use Boring Strategy with real market data

This example shows how to integrate the Boring Strategy with 
real market data from an exchange API (e.g., Binance).
"""

from boring_strategy import BoringStrategy
from datetime import datetime, timezone
import pytz


def convert_binance_kline_to_candle(kline):
    """
    Convert Binance kline format to strategy candle format.
    
    Binance kline format:
    [
        timestamp,
        open,
        high,
        low,
        close,
        volume,
        close_time,
        quote_asset_volume,
        number_of_trades,
        taker_buy_base_asset_volume,
        taker_buy_quote_asset_volume,
        ignore
    ]
    
    Args:
        kline: List or tuple from Binance API
        
    Returns:
        Dict in format expected by BoringStrategy
    """
    return {
        'timestamp': int(kline[0]),
        'open': float(kline[1]),
        'high': float(kline[2]),
        'low': float(kline[3]),
        'close': float(kline[4])
    }


def example_with_binance_data():
    """
    Example showing how to use with Binance API data.
    
    Note: This is pseudocode - you need to implement actual API calls
    """
    # Pseudocode for getting data from Binance
    # In real implementation, you would use requests or python-binance library
    
    print("Example: Using Boring Strategy with Binance Data")
    print("=" * 60)
    print()
    
    # 1. Get 15-minute candles for the day
    # symbol = "BTCUSDT"
    # interval_15m = "15m"
    # start_time = today at 9:30 AM EST
    # end_time = today at 9:45 AM EST
    #
    # klines_15m = binance_client.get_klines(
    #     symbol=symbol,
    #     interval=interval_15m,
    #     startTime=start_time,
    #     endTime=end_time
    # )
    #
    # candles_15min = [convert_binance_kline_to_candle(k) for k in klines_15m]
    
    # 2. Get 5-minute candles from 9:30 AM to current time
    # interval_5m = "5m"
    # end_time_current = current time
    #
    # klines_5m = binance_client.get_klines(
    #     symbol=symbol,
    #     interval=interval_5m,
    #     startTime=start_time,
    #     endTime=end_time_current
    # )
    #
    # candles_5min = [convert_binance_kline_to_candle(k) for k in klines_5m]
    
    # 3. Run the strategy
    # strategy = BoringStrategy()
    # result = strategy.backtest(candles_15min, candles_5min)
    #
    # if 'error' not in result:
    #     print(f"Trade Setup Found!")
    #     print(f"Direction: {result['trade_params']['direction']}")
    #     print(f"Entry: ${result['trade_params']['entry']:.2f}")
    #     print(f"Stop Loss: ${result['trade_params']['sl']:.2f}")
    #     print(f"Take Profit: ${result['trade_params']['tp']:.2f}")
    # else:
    #     print(f"No valid setup: {result['error']}")
    
    print("See boring_strategy.py for actual implementation")
    print("Install python-binance: pip install python-binance")
    print()


def example_live_scanning():
    """
    Example of how to scan for setups in real-time.
    
    This would be called every 5 minutes during trading hours.
    """
    print("Example: Live Scanning for Boring Strategy Setups")
    print("=" * 60)
    print()
    
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    print(f"Current time (EST): {now.strftime('%H:%M')}")
    print()
    
    # Check if we're in trading hours
    if now.hour < 9 or (now.hour == 9 and now.minute < 45):
        print("⏰ Market not open yet or before 9:45 AM EST")
        print("   Waiting for 9:45 AM to start scanning...")
        return
    
    if now.hour >= 12:
        print("⏰ Past 12:00 PM EST - trading window closed")
        print("   No new setups will be taken after 12:00 PM")
        return
    
    print("✅ Within trading hours (9:45 AM - 12:00 PM EST)")
    print("   Scanning for Fair Value Gap setups...")
    print()
    
    # Pseudocode for live scanning:
    #
    # 1. Get today's 9:30-9:45 15-min candle
    # 2. Get all 5-min candles from 9:30 to now
    # 3. Run strategy.backtest()
    # 4. If valid setup found:
    #    - Check if we haven't already entered this setup
    #    - Place limit order at FVG mid
    #    - Set stop loss
    #    - Set take profit
    # 5. If no setup yet:
    #    - Wait 5 minutes
    #    - Scan again
    
    print("Pseudocode implementation:")
    print("""
    while True:
        now = datetime.now(eastern)
        
        if 9.45 <= now.hour + now.minute/60 < 12:
            # Get data
            candles_15min = get_range_candle()
            candles_5min = get_5min_candles()
            
            # Check for setup
            result = strategy.backtest(candles_15min, candles_5min)
            
            if 'error' not in result and result['entry_filled'] == False:
                # New setup found! Place order
                place_limit_order(
                    price=result['trade_params']['entry'],
                    sl=result['trade_params']['sl'],
                    tp=result['trade_params']['tp']
                )
                break  # Exit after placing order
        
        # Wait 5 minutes before next scan
        time.sleep(300)
    """)
    print()


def example_multiple_symbols():
    """
    Example of scanning multiple symbols for setups.
    """
    print("Example: Scanning Multiple Symbols")
    print("=" * 60)
    print()
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    
    print(f"Scanning {len(symbols)} symbols for setups...")
    print()
    
    for symbol in symbols:
        print(f"📊 {symbol}:")
        print(f"   1. Get 9:30-9:45 range")
        print(f"   2. Get 5-min candles")
        print(f"   3. Check for FVG break")
        print(f"   4. Calculate entry/sl/tp")
        print()
    
    print("Implementation note:")
    print("- Process symbols in parallel for efficiency")
    print("- Apply position sizing rules")
    print("- Track multiple positions if allowed")
    print()


def main():
    """
    Main function to run all examples
    """
    print()
    print("=" * 60)
    print("BORING STRATEGY - Integration Examples")
    print("=" * 60)
    print()
    
    example_with_binance_data()
    print("\n" + "=" * 60 + "\n")
    
    example_live_scanning()
    print("\n" + "=" * 60 + "\n")
    
    example_multiple_symbols()
    
    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print()
    print("1. Install required packages:")
    print("   pip install python-binance pytz")
    print()
    print("2. Get Binance API keys (or use your exchange)")
    print()
    print("3. Test with demo/testnet first")
    print()
    print("4. Implement proper error handling and logging")
    print()
    print("5. Add position management and risk controls")
    print()
    print("6. Paper trade for 30 days minimum")
    print()
    print("7. Only then consider live trading with small size")
    print()


if __name__ == "__main__":
    main()
