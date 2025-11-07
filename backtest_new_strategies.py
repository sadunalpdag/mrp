#!/usr/bin/env python3
"""
Backtest for New Trading Strategies (LO_ORB, NYR, ICT_P3)
Tests all futures coins with 3 months of historical data
Does NOT modify ema.py

Usage:
  python3 backtest_new_strategies.py                    # Full backtest (all symbols, 3 months)
  python3 backtest_new_strategies.py --quick            # Quick test (top 20 symbols, 1 week)
  python3 backtest_new_strategies.py --symbols BTC ETH  # Specific symbols
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import strategy functions from ema.py
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import ema

BINANCE_FAPI = "https://fapi.binance.com"
RESULTS_FILE = "backtest_results_new_strategies.json"
TRADES_FILE = "backtest_trades_new_strategies.json"

# Configuration
RATE_LIMIT_DELAY = 0.1  # Delay between API calls to avoid rate limiting

def log(msg):
    """Print and log messages"""
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)

def get_all_futures_symbols():
    """Get all USDT futures symbols from Binance"""
    try:
        info = requests.get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo", timeout=10).json()
        symbols = [
            s["symbol"] for s in info["symbols"]
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        ]
        log(f"Found {len(symbols)} futures symbols")
        return sorted(symbols)
    except Exception as e:
        log(f"Error getting symbols: {e}")
        log("Using fallback symbol list...")
        # Fallback to common symbols if API is not accessible
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
            "XRPUSDT", "DOTUSDT", "UNIUSDT", "LINKUSDT", "MATICUSDT",
            "SOLUSDT", "AVAXUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT"
        ]

def get_historical_klines(symbol, interval="1h", days=90):
    """
    Get historical klines for a symbol
    Fetches N months of 1-hour data
    """
    try:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1500
            }
            
            response = requests.get(
                f"{BINANCE_FAPI}/fapi/v1/klines",
                params=params,
                timeout=15
            )
            
            if response.status_code != 200:
                log(f"  {symbol}: Error {response.status_code}")
                break
            
            klines = response.json()
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Move to next batch
            current_start = int(klines[-1][0]) + 1
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            if len(klines) < 1500:
                break
        
        log(f"  {symbol}: Retrieved {len(all_klines)} klines")
        return all_klines
        
    except Exception as e:
        log(f"  {symbol}: Error - {e}")
        return []

def simulate_trade(signal, klines, signal_index):
    """
    Simulate a trade from signal to TP/SL
    Returns trade result dict
    """
    entry = signal["entry"]
    tp = signal["tp"]
    sl = signal["sl"]
    direction = signal["dir"]
    
    # Look forward from signal_index to find TP or SL hit
    entry_time = int(klines[signal_index][0])
    
    for i in range(signal_index + 1, len(klines)):
        kline = klines[i]
        high = float(kline[2])
        low = float(kline[3])
        close = float(kline[4])
        current_time = int(kline[0])
        
        hit_tp = False
        hit_sl = False
        exit_price = close
        
        if direction == "UP":
            if high >= tp:
                hit_tp = True
                exit_price = tp
            elif low <= sl:
                hit_sl = True
                exit_price = sl
        else:  # DOWN
            if low <= tp:
                hit_tp = True
                exit_price = tp
            elif high >= sl:
                hit_sl = True
                exit_price = sl
        
        if hit_tp or hit_sl:
            # Calculate profit/loss
            if direction == "UP":
                pnl_pct = ((exit_price / entry) - 1.0) * 100
            else:
                pnl_pct = ((entry / exit_price) - 1.0) * 100
            
            duration_hours = (current_time - entry_time) / (1000 * 3600)
            
            return {
                "symbol": signal["symbol"],
                "strategy": signal["kind"],
                "direction": direction,
                "entry_time": datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc).isoformat(),
                "exit_time": datetime.fromtimestamp(current_time / 1000, tz=timezone.utc).isoformat(),
                "entry_price": entry,
                "exit_price": exit_price,
                "tp": tp,
                "sl": sl,
                "exit_reason": "TP" if hit_tp else "SL",
                "pnl_pct": round(pnl_pct, 3),
                "duration_hours": round(duration_hours, 2),
                "power": signal.get("power", 0),
                "rsi": signal.get("rsi", 50)
            }
    
    # Trade not completed within data range
    return None

def backtest_symbol(symbol, strategies_to_test, days=90):
    """
    Backtest a single symbol with new strategies
    Returns list of completed trades
    """
    klines = get_historical_klines(symbol, interval="1h", days=days)
    
    if len(klines) < 100:
        return []
    
    trades = []
    signals_found = 0
    
    # Scan through historical data
    for i in range(60, len(klines) - 10):  # Leave some bars at end for trade completion
        # Get klines up to current bar
        kl_subset = klines[:i+1]
        
        # Test each strategy
        for strategy_name, strategy_func in strategies_to_test.items():
            try:
                signal = strategy_func(symbol, kl_subset, i)
                
                if signal:
                    signals_found += 1
                    
                    # Simulate the trade
                    trade_result = simulate_trade(signal, klines, i)
                    
                    if trade_result:
                        trades.append(trade_result)
                        
            except Exception as e:
                # Strategy might fail on some data, skip quietly
                pass
    
    if signals_found > 0:
        log(f"  {symbol}: {signals_found} signals, {len(trades)} completed trades")
    
    return trades

def analyze_results(all_trades):
    """
    Analyze backtest results and generate statistics
    """
    if not all_trades:
        return {
            "error": "No trades found",
            "total_trades": 0
        }
    
    stats = {
        "total_trades": len(all_trades),
        "by_strategy": {},
        "by_direction": {"UP": [], "DOWN": []},
        "winners": [t for t in all_trades if t["exit_reason"] == "TP"],
        "losers": [t for t in all_trades if t["exit_reason"] == "SL"],
        "overall": {}
    }
    
    # Overall statistics
    win_count = len(stats["winners"])
    loss_count = len(stats["losers"])
    total = len(all_trades)
    
    stats["overall"]["win_rate"] = round((win_count / total * 100) if total > 0 else 0, 2)
    stats["overall"]["total_pnl"] = round(sum(t["pnl_pct"] for t in all_trades), 2)
    stats["overall"]["avg_pnl"] = round(sum(t["pnl_pct"] for t in all_trades) / total if total > 0 else 0, 3)
    stats["overall"]["avg_winner"] = round(sum(t["pnl_pct"] for t in stats["winners"]) / win_count if win_count > 0 else 0, 3)
    stats["overall"]["avg_loser"] = round(sum(t["pnl_pct"] for t in stats["losers"]) / loss_count if loss_count > 0 else 0, 3)
    stats["overall"]["avg_duration_hours"] = round(sum(t["duration_hours"] for t in all_trades) / total if total > 0 else 0, 2)
    
    # By strategy
    strategies = set(t["strategy"] for t in all_trades)
    for strategy in strategies:
        strategy_trades = [t for t in all_trades if t["strategy"] == strategy]
        strategy_winners = [t for t in strategy_trades if t["exit_reason"] == "TP"]
        strategy_losers = [t for t in strategy_trades if t["exit_reason"] == "SL"]
        
        st_total = len(strategy_trades)
        st_wins = len(strategy_winners)
        st_losses = len(strategy_losers)
        
        stats["by_strategy"][strategy] = {
            "total_trades": st_total,
            "wins": st_wins,
            "losses": st_losses,
            "win_rate": round((st_wins / st_total * 100) if st_total > 0 else 0, 2),
            "total_pnl": round(sum(t["pnl_pct"] for t in strategy_trades), 2),
            "avg_pnl": round(sum(t["pnl_pct"] for t in strategy_trades) / st_total if st_total > 0 else 0, 3),
            "avg_winner": round(sum(t["pnl_pct"] for t in strategy_winners) / st_wins if st_wins > 0 else 0, 3),
            "avg_loser": round(sum(t["pnl_pct"] for t in strategy_losers) / st_losses if st_losses > 0 else 0, 3),
            "avg_duration_hours": round(sum(t["duration_hours"] for t in strategy_trades) / st_total if st_total > 0 else 0, 2)
        }
    
    # By direction
    for direction in ["UP", "DOWN"]:
        dir_trades = [t for t in all_trades if t["direction"] == direction]
        dir_winners = [t for t in dir_trades if t["exit_reason"] == "TP"]
        
        if dir_trades:
            stats["by_direction"][direction] = {
                "total_trades": len(dir_trades),
                "wins": len(dir_winners),
                "win_rate": round((len(dir_winners) / len(dir_trades) * 100), 2),
                "total_pnl": round(sum(t["pnl_pct"] for t in dir_trades), 2),
                "avg_pnl": round(sum(t["pnl_pct"] for t in dir_trades) / len(dir_trades), 3)
            }
    
    return stats

def print_statistics(stats):
    """Print formatted statistics"""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS - NEW STRATEGIES (LO_ORB, NYR, ICT_P3)")
    print("=" * 80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Trades: {stats['overall'].get('total_trades', stats.get('total_trades', 0))}")
    
    if stats.get("error"):
        print(f"  {stats['error']}")
        return
    
    print(f"  Win Rate: {stats['overall']['win_rate']}%")
    print(f"  Total PnL: {stats['overall']['total_pnl']}%")
    print(f"  Average PnL per Trade: {stats['overall']['avg_pnl']}%")
    print(f"  Average Winner: {stats['overall']['avg_winner']}%")
    print(f"  Average Loser: {stats['overall']['avg_loser']}%")
    print(f"  Average Duration: {stats['overall']['avg_duration_hours']} hours")
    
    print(f"\nBy Strategy:")
    for strategy, st_stats in stats["by_strategy"].items():
        print(f"\n  {strategy}:")
        print(f"    Trades: {st_stats['total_trades']} (W:{st_stats['wins']}, L:{st_stats['losses']})")
        print(f"    Win Rate: {st_stats['win_rate']}%")
        print(f"    Total PnL: {st_stats['total_pnl']}%")
        print(f"    Avg PnL: {st_stats['avg_pnl']}%")
        print(f"    Avg Winner: {st_stats['avg_winner']}%")
        print(f"    Avg Loser: {st_stats['avg_loser']}%")
        print(f"    Avg Duration: {st_stats['avg_duration_hours']} hours")
    
    print("\n" + "=" * 80)

def main():
    """Main backtest execution"""
    parser = argparse.ArgumentParser(description='Backtest new trading strategies')
    parser.add_argument('--quick', action='store_true', help='Quick test (20 symbols, 7 days)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to test')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest (default: 90)')
    
    args = parser.parse_args()
    
    log("=" * 80)
    log("Starting Backtest for New Trading Strategies")
    log(f"Period: {args.days} days | Interval: 1 hour")
    log("=" * 80)
    
    # Define strategies to test
    strategies_to_test = {
        "LO_ORB": ema.build_lo_orb_signal,
        "NYR": ema.build_ny_reversal_signal,
        "ICT_P3": ema.build_ict_power3_signal
    }
    
    log(f"\nStrategies to test: {', '.join(strategies_to_test.keys())}")
    
    # Get symbols
    if args.symbols:
        symbols = [s.upper() if not s.endswith('USDT') else s.upper() for s in args.symbols]
        symbols = [s if s.endswith('USDT') else f"{s}USDT" for s in symbols]
    else:
        symbols = get_all_futures_symbols()
    
    if not symbols:
        log("No symbols found. Exiting.")
        return
    
    # Quick mode: limit symbols and days
    if args.quick:
        symbols = symbols[:20]
        args.days = 7
        log("Quick mode: Testing first 20 symbols with 7 days of data")
    
    log(f"\nBacktesting {len(symbols)} symbols...")
    log("This may take a while...\n")
    
    all_trades = []
    processed = 0
    
    # Process symbols in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(backtest_symbol, symbol, strategies_to_test, args.days): symbol
            for symbol in symbols
        }
        
        for future in as_completed(futures):
            symbol = futures[future]
            processed += 1
            
            try:
                trades = future.result()
                all_trades.extend(trades)
                
                if processed % 10 == 0:
                    log(f"Progress: {processed}/{len(symbols)} symbols processed, {len(all_trades)} trades so far")
                    
            except Exception as e:
                log(f"Error processing {symbol}: {e}")
    
    log(f"\nBacktest complete!")
    log(f"Processed {processed} symbols")
    log(f"Total trades found: {len(all_trades)}")
    
    # Save all trades
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_trades, f, indent=2, ensure_ascii=False)
    log(f"\nTrades saved to: {TRADES_FILE}")
    
    # Analyze results
    stats = analyze_results(all_trades)
    stats["backtest_info"] = {
        "start_time": (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "symbols_tested": len(symbols),
        "strategies": list(strategies_to_test.keys()),
        "days": args.days
    }
    
    # Save statistics
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log(f"Results saved to: {RESULTS_FILE}")
    
    # Print statistics
    print_statistics(stats)
    
    log("\nBacktest complete!")

if __name__ == "__main__":
    main()
