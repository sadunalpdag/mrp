"""
Multi-Coin 3-Month Backtest for Boring Strategy

This script runs the Boring Strategy backtest across multiple cryptocurrency
symbols over a 3-month period. It generates daily backtests for each symbol
and aggregates the results.

Usage:
    python boring_strategy_multi_backtest.py

Output:
    - multi_backtest_results.json: Detailed results for each day/symbol
    - multi_backtest_summary.json: Aggregated statistics
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz
from boring_strategy import BoringStrategy


class MultiCoinBacktest:
    """
    Runs backtests across multiple symbols and time periods
    """
    
    def __init__(self):
        self.strategy = BoringStrategy()
        self.eastern = pytz.timezone('US/Eastern')
        
        # Popular crypto symbols for backtesting
        self.symbols = [
            "BTCUSDT",   # Bitcoin
            "ETHUSDT",   # Ethereum
            "BNBUSDT",   # Binance Coin
            "SOLUSDT",   # Solana
            "ADAUSDT",   # Cardano
            "XRPUSDT",   # Ripple
            "DOGEUSDT",  # Dogecoin
            "MATICUSDT", # Polygon
            "DOTUSDT",   # Polkadot
            "AVAXUSDT",  # Avalanche
        ]
    
    def generate_trading_days(self, months: int = 3) -> List[datetime]:
        """
        Generate list of trading days for the backtest period.
        
        Args:
            months: Number of months to backtest (default: 3)
            
        Returns:
            List of datetime objects representing trading days
        """
        end_date = datetime.now(self.eastern).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=months * 30)
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (crypto trades 7 days, but we use EST market hours)
            # For crypto, we can include all days
            trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def generate_sample_day_data(self, symbol: str, date: datetime) -> tuple:
        """
        Generate sample intraday data for a given symbol and date.
        
        In a real implementation, this would fetch actual historical data
        from an exchange API (e.g., Binance).
        
        Args:
            symbol: Trading pair symbol
            date: Trading date
            
        Returns:
            Tuple of (candles_15min, candles_5min)
        """
        # This is a simplified simulation
        # In production, you would fetch real data from exchange API
        
        import random
        random.seed(f"{symbol}{date.isoformat()}")  # Reproducible random data
        
        # Starting price varies by symbol
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 2500,
            "BNBUSDT": 300,
            "SOLUSDT": 100,
            "ADAUSDT": 0.50,
            "XRPUSDT": 0.60,
            "DOGEUSDT": 0.10,
            "MATICUSDT": 0.80,
            "DOTUSDT": 7.0,
            "AVAXUSDT": 35.0,
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Add some variation based on date
        days_from_start = (date - datetime(2024, 1, 1, tzinfo=self.eastern)).days
        price_variation = 1 + (days_from_start % 30) * 0.01  # Up to 30% variation
        start_price = base_price * price_variation
        
        # 15-minute range candle (9:30-9:45)
        range_size = start_price * random.uniform(0.015, 0.03)  # 1.5-3% range
        candles_15min = [{
            'timestamp': int((date.replace(hour=9, minute=30)).timestamp() * 1000),
            'open': start_price,
            'high': start_price + range_size,
            'low': start_price - range_size * 0.5,
            'close': start_price + range_size * 0.3
        }]
        
        # Generate 5-minute candles
        candles_5min = []
        current_time = date.replace(hour=9, minute=30)
        current_price = start_price
        
        # Decide if there will be a valid setup today (70% chance)
        has_setup = random.random() < 0.70
        
        if has_setup:
            # Generate candles leading to FVG
            for i in range(4):
                ts = current_time + timedelta(minutes=i*5)
                candles_5min.append({
                    'timestamp': int(ts.timestamp() * 1000),
                    'open': current_price,
                    'high': current_price + start_price * 0.003,
                    'low': current_price - start_price * 0.002,
                    'close': current_price + start_price * 0.001
                })
                current_price += start_price * 0.001
            
            # Create FVG pattern
            range_high = candles_15min[0]['high']
            
            # Candle 1: Base
            ts1 = current_time + timedelta(minutes=20)
            c1_low = current_price - start_price * 0.002
            candles_5min.append({
                'timestamp': int(ts1.timestamp() * 1000),
                'open': current_price,
                'high': current_price + start_price * 0.005,
                'low': c1_low,
                'close': current_price + start_price * 0.002
            })
            
            # Candle 2: Expansive move
            ts2 = current_time + timedelta(minutes=25)
            candles_5min.append({
                'timestamp': int(ts2.timestamp() * 1000),
                'open': current_price + start_price * 0.002,
                'high': current_price + start_price * 0.025,
                'low': current_price + start_price * 0.001,
                'close': current_price + start_price * 0.023
            })
            current_price += start_price * 0.023
            
            # Candle 3: Creates gap, breaks range
            ts3 = current_time + timedelta(minutes=30)
            c3_low = current_price - start_price * 0.001
            candles_5min.append({
                'timestamp': int(ts3.timestamp() * 1000),
                'open': current_price,
                'high': current_price + start_price * 0.008,
                'low': c3_low,
                'close': range_high + start_price * 0.005  # Above range
            })
            current_price = range_high + start_price * 0.005
            
            # Pullback candle to allow entry at FVG
            ts_pullback = ts3 + timedelta(minutes=5)
            c1_high = candles_5min[-3]['high']
            fvg_mid = (c1_high + c3_low) / 2
            candles_5min.append({
                'timestamp': int(ts_pullback.timestamp() * 1000),
                'open': current_price,
                'high': current_price + start_price * 0.003,
                'low': fvg_mid - start_price * 0.002,  # Touch FVG
                'close': fvg_mid + start_price * 0.001
            })
            
            # Add candles to reach TP (80% chance)
            reaches_tp = random.random() < 0.80
            
            # Calculate expected entry/TP/SL
            risk = fvg_mid - c1_low
            tp_price = fvg_mid + 2 * risk
            sl_price = c1_low
            
            if reaches_tp:
                # Generate candles that reach TP
                base_price = fvg_mid
                for i in range(10):
                    ts = ts_pullback + timedelta(minutes=(i+1)*5)
                    progress = (i + 1) / 10
                    price = base_price + (tp_price - base_price) * progress
                    candles_5min.append({
                        'timestamp': int(ts.timestamp() * 1000),
                        'open': price - start_price * 0.001,
                        'high': price + start_price * 0.004,  # Ensure TP is hit
                        'low': price - start_price * 0.002,
                        'close': price + start_price * 0.002
                    })
            else:
                # Generate candles that hit SL
                base_price = fvg_mid
                for i in range(6):
                    ts = ts_pullback + timedelta(minutes=(i+1)*5)
                    progress = (i + 1) / 6
                    price = base_price - (base_price - sl_price) * progress
                    candles_5min.append({
                        'timestamp': int(ts.timestamp() * 1000),
                        'open': price + start_price * 0.001,
                        'high': price + start_price * 0.002,
                        'low': price - start_price * 0.004,  # Ensure SL is hit
                        'close': price - start_price * 0.001
                    })
        else:
            # No valid setup - just random movement
            for i in range(20):
                ts = current_time + timedelta(minutes=i*5)
                current_price += random.uniform(-start_price * 0.005, start_price * 0.005)
                candles_5min.append({
                    'timestamp': int(ts.timestamp() * 1000),
                    'open': current_price,
                    'high': current_price + start_price * 0.003,
                    'low': current_price - start_price * 0.003,
                    'close': current_price + random.uniform(-start_price * 0.002, start_price * 0.002)
                })
        
        return candles_15min, candles_5min
    
    def run_multi_backtest(self, months: int = 3, max_symbols: Optional[int] = None) -> Dict:
        """
        Run backtest across multiple symbols and days.
        
        Args:
            months: Number of months to backtest
            max_symbols: Maximum number of symbols to test (None = all)
            
        Returns:
            Dict with detailed results and summary statistics
        """
        print(f"🚀 Starting Multi-Coin Backtest")
        print(f"   Period: {months} months")
        
        symbols_to_test = self.symbols[:max_symbols] if max_symbols else self.symbols
        print(f"   Symbols: {len(symbols_to_test)} coins")
        print()
        
        trading_days = self.generate_trading_days(months)
        print(f"   Trading days: {len(trading_days)}")
        print()
        
        all_results = []
        summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'no_setup_days': 0,
            'total_gain_pct': 0.0,
            'by_symbol': {},
            'by_outcome': {'TP': 0, 'SL': 0, 'NO_EXIT': 0, 'ERROR': 0}
        }
        
        for symbol in symbols_to_test:
            print(f"📊 Testing {symbol}...")
            
            symbol_stats = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_gain_pct': 0.0
            }
            
            for day in trading_days:
                # Generate sample data for this day
                candles_15min, candles_5min = self.generate_sample_day_data(symbol, day)
                
                # Run backtest
                result = self.strategy.backtest(candles_15min, candles_5min)
                
                # Record result
                if 'error' in result:
                    if 'No valid FVG' in result['error']:
                        summary['no_setup_days'] += 1
                    else:
                        summary['by_outcome']['ERROR'] += 1
                else:
                    outcome = result.get('outcome', 'NO_EXIT')
                    summary['by_outcome'][outcome] += 1
                    
                    if outcome == 'TP':
                        summary['total_trades'] += 1
                        summary['winning_trades'] += 1
                        symbol_stats['trades'] += 1
                        symbol_stats['wins'] += 1
                        
                        gain = result.get('gain_pct', 0)
                        summary['total_gain_pct'] += gain
                        symbol_stats['total_gain_pct'] += gain
                        
                    elif outcome == 'SL':
                        summary['total_trades'] += 1
                        summary['losing_trades'] += 1
                        symbol_stats['trades'] += 1
                        symbol_stats['losses'] += 1
                        
                        gain = result.get('gain_pct', 0)
                        summary['total_gain_pct'] += gain
                        symbol_stats['total_gain_pct'] += gain
                
                # Store detailed result
                all_results.append({
                    'symbol': symbol,
                    'date': day.strftime('%Y-%m-%d'),
                    **result
                })
            
            # Store symbol summary
            if symbol_stats['trades'] > 0:
                symbol_stats['win_rate'] = (symbol_stats['wins'] / symbol_stats['trades']) * 100
                symbol_stats['avg_gain_pct'] = symbol_stats['total_gain_pct'] / symbol_stats['trades']
            else:
                symbol_stats['win_rate'] = 0
                symbol_stats['avg_gain_pct'] = 0
            
            summary['by_symbol'][symbol] = symbol_stats
            
            print(f"   ✓ {symbol}: {symbol_stats['trades']} trades, {symbol_stats['win_rate']:.1f}% win rate")
        
        # Calculate overall statistics
        if summary['total_trades'] > 0:
            summary['win_rate'] = (summary['winning_trades'] / summary['total_trades']) * 100
            summary['avg_gain_per_trade'] = summary['total_gain_pct'] / summary['total_trades']
        else:
            summary['win_rate'] = 0
            summary['avg_gain_per_trade'] = 0
        
        summary['backtest_period_months'] = months
        summary['symbols_tested'] = symbols_to_test
        summary['trading_days'] = len(trading_days)
        
        print()
        print("=" * 60)
        print("✅ Backtest Complete!")
        print("=" * 60)
        
        return {
            'summary': summary,
            'detailed_results': all_results
        }


def main():
    """
    Main function to run multi-coin backtest
    """
    print("=" * 60)
    print("BORING STRATEGY - Multi-Coin 3-Month Backtest")
    print("=" * 60)
    print()
    
    # Initialize backtest
    backtest = MultiCoinBacktest()
    
    # Run backtest (3 months, all symbols)
    results = backtest.run_multi_backtest(months=3)
    
    # Display summary
    summary = results['summary']
    
    print()
    print("📈 OVERALL RESULTS")
    print("=" * 60)
    print(f"Total Trades:        {summary['total_trades']}")
    print(f"Winning Trades:      {summary['winning_trades']}")
    print(f"Losing Trades:       {summary['losing_trades']}")
    print(f"Win Rate:            {summary['win_rate']:.2f}%")
    print(f"Avg Gain per Trade:  {summary['avg_gain_per_trade']:.2f}%")
    print(f"Total Gain:          {summary['total_gain_pct']:.2f}%")
    print(f"No Setup Days:       {summary['no_setup_days']}")
    print()
    
    print("📊 RESULTS BY OUTCOME")
    print("=" * 60)
    for outcome, count in summary['by_outcome'].items():
        print(f"{outcome:12s}: {count:4d} trades")
    print()
    
    print("💰 TOP PERFORMING SYMBOLS")
    print("=" * 60)
    # Sort symbols by win rate
    sorted_symbols = sorted(
        summary['by_symbol'].items(),
        key=lambda x: x[1]['win_rate'],
        reverse=True
    )
    
    for i, (symbol, stats) in enumerate(sorted_symbols[:5], 1):
        print(f"{i}. {symbol:12s}: {stats['win_rate']:5.1f}% win rate, "
              f"{stats['trades']:3d} trades, "
              f"{stats['avg_gain_pct']:+6.2f}% avg gain")
    print()
    
    # Save results to files
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    summary_file = os.path.join(output_dir, 'multi_backtest_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results['summary'], f, indent=2, ensure_ascii=False)
    print(f"📁 Summary saved to: {summary_file}")
    
    detailed_file = os.path.join(output_dir, 'multi_backtest_results.json')
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results['detailed_results'], f, indent=2, ensure_ascii=False)
    print(f"📁 Detailed results saved to: {detailed_file}")
    
    print()
    print("=" * 60)
    print("💡 NOTE: This backtest uses simulated data")
    print("   For production use, integrate with real exchange API")
    print("   (See boring_strategy_examples.py for integration guide)")
    print("=" * 60)


if __name__ == "__main__":
    main()
