#!/usr/bin/env python3
"""
Backtest Framework for EMA Trading Strategies
Tests all strategies from ema.py against historical data
"""

import os
import json
import time
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import sys

# Import strategy functions and indicators from ema.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ema import (
    ema, rsi, macd, schaff_tc, atr_like, supertrend,
    build_utstc_signal, build_macd_trend_signal, build_fvg_break_signal,
    build_ema_pullback_signal, build_kivanc_confirm_signal, 
    build_cest_signal, build_ema_structure_signal
)

BINANCE_FAPI = "https://fapi.binance.com"

# Backtest configuration
INITIAL_CAPITAL = 10000.0  # Starting capital in USDT
TRADE_SIZE_USDT = 250.0    # Size per trade
COMMISSION_RATE = 0.0004   # 0.04% per trade (maker/taker combined)

class BacktestPosition:
    """Represents an open position in the backtest"""
    def __init__(self, symbol: str, direction: str, entry_price: float, 
                 tp: float, sl: float, size_usdt: float, entry_bar: int,
                 strategy: str, power: float):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.tp = tp
        self.sl = sl
        self.size_usdt = size_usdt
        self.entry_bar = entry_bar
        self.strategy = strategy
        self.power = power
        self.quantity = size_usdt / entry_price
        
    def check_exit(self, high: float, low: float, close: float, 
                   bar_index: int) -> Optional[Tuple[str, float, int]]:
        """Check if position should be exited. Returns (reason, price, bars_held)"""
        bars_held = bar_index - self.entry_bar
        
        if self.direction == "UP":
            # Check TP first (price went up)
            if high >= self.tp:
                return ("TP", self.tp, bars_held)
            # Then check SL (price went down)
            elif low <= self.sl:
                return ("SL", self.sl, bars_held)
        else:  # DOWN
            # Check TP first (price went down)
            if low <= self.tp:
                return ("TP", self.tp, bars_held)
            # Then check SL (price went up)
            elif high >= self.sl:
                return ("SL", self.sl, bars_held)
        
        return None
    
    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate profit/loss for this position"""
        if self.direction == "UP":
            # Long position
            pnl_pct = (exit_price / self.entry_price - 1.0)
        else:
            # Short position
            pnl_pct = (self.entry_price / exit_price - 1.0)
        
        # Apply commission (entry + exit)
        pnl_pct -= (2 * COMMISSION_RATE)
        
        return self.size_usdt * pnl_pct


class BacktestEngine:
    """Backtesting engine for strategy evaluation"""
    
    def __init__(self, symbol: str, interval: str = "1h", 
                 lookback_days: int = 90):
        self.symbol = symbol
        self.interval = interval
        self.lookback_days = lookback_days
        self.data = []
        self.results = []
        
    def fetch_historical_data(self, use_sample_data=True) -> bool:
        """Fetch historical kline data from Binance or sample data"""
        print(f"Fetching {self.lookback_days} days of {self.interval} data for {self.symbol}...")
        
        # Try to load from sample data file first
        if use_sample_data:
            sample_file = os.path.join(os.path.dirname(__file__), "sample_market_data.json")
            if os.path.exists(sample_file):
                try:
                    with open(sample_file, "r") as f:
                        all_data = json.load(f)
                    
                    if self.symbol in all_data:
                        self.data = all_data[self.symbol]
                        print(f"Loaded {len(self.data)} candles from sample data")
                        return len(self.data) > 0
                    else:
                        print(f"Symbol {self.symbol} not found in sample data")
                except Exception as e:
                    print(f"Error loading sample data: {e}")
        
        # Fall back to API if sample data not available
        # Calculate how many candles we need
        if self.interval == "1h":
            limit = min(1500, self.lookback_days * 24)
        elif self.interval == "4h":
            limit = min(1500, self.lookback_days * 6)
        elif self.interval == "1d":
            limit = min(1500, self.lookback_days)
        else:
            limit = 1000
        
        try:
            response = requests.get(
                f"{BINANCE_FAPI}/fapi/v1/klines",
                params={
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "limit": limit
                },
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code}")
                return False
            
            self.data = response.json()
            
            # Remove incomplete candle if present
            if self.data and int(self.data[-1][6]) > int(time.time() * 1000):
                self.data = self.data[:-1]
            
            print(f"Fetched {len(self.data)} candles")
            return len(self.data) > 0
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def run_strategy_backtest(self, strategy_name: str, 
                              strategy_func) -> Dict[str, Any]:
        """Run backtest for a specific strategy"""
        print(f"\nBacktesting {strategy_name}...")
        
        capital = INITIAL_CAPITAL
        positions = []
        closed_trades = []
        
        # Need enough data for indicators
        min_bars = 250
        
        if len(self.data) < min_bars:
            print(f"Not enough data for {strategy_name} (need {min_bars}, have {len(self.data)})")
            return self._empty_results(strategy_name)
        
        # Iterate through historical data
        for bar_index in range(min_bars, len(self.data)):
            # Get kline data up to current bar
            kl = self.data[:bar_index + 1]
            
            # Extract current bar OHLC
            current_bar = self.data[bar_index]
            curr_open = float(current_bar[1])
            curr_high = float(current_bar[2])
            curr_low = float(current_bar[3])
            curr_close = float(current_bar[4])
            
            # Check if any open positions should be closed
            still_open = []
            for pos in positions:
                exit_result = pos.check_exit(curr_high, curr_low, curr_close, bar_index)
                
                if exit_result:
                    reason, exit_price, bars_held = exit_result
                    pnl = pos.calculate_pnl(exit_price)
                    capital += pnl
                    
                    pnl_pct = (pnl / pos.size_usdt) * 100
                    
                    closed_trades.append({
                        "symbol": pos.symbol,
                        "strategy": pos.strategy,
                        "direction": pos.direction,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "exit_reason": reason,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "power": pos.power
                    })
                else:
                    still_open.append(pos)
            
            positions = still_open
            
            # Check for new signals (only if we have capital available)
            if capital > TRADE_SIZE_USDT:
                try:
                    signal = strategy_func(self.symbol, kl, bar_index)
                    
                    if signal:
                        # Adjust SL if it's using placeholder values (0.8 or 1.2 multipliers)
                        # These indicate "no real SL" in the original code
                        entry = signal["entry"]
                        tp = signal["tp"]
                        sl = signal["sl"]
                        
                        # Check if SL is a placeholder (too wide)
                        if signal["dir"] == "UP":
                            sl_pct = abs((sl / entry) - 1)
                            if sl_pct > 0.15:  # More than 15% = placeholder
                                # Use reasonable 2% SL
                                sl = entry * 0.98
                        else:  # DOWN
                            sl_pct = abs((sl / entry) - 1)
                            if sl_pct > 0.15:  # More than 15% = placeholder
                                # Use reasonable 2% SL
                                sl = entry * 1.02
                        
                        # Open new position
                        pos = BacktestPosition(
                            symbol=self.symbol,
                            direction=signal["dir"],
                            entry_price=entry,
                            tp=tp,
                            sl=sl,
                            size_usdt=TRADE_SIZE_USDT,
                            entry_bar=bar_index,
                            strategy=strategy_name,
                            power=signal.get("power", 0)
                        )
                        
                        positions.append(pos)
                        capital -= TRADE_SIZE_USDT
                        
                except Exception as e:
                    # Strategy might fail on some data, continue
                    pass
        
        # Close any remaining open positions at last price
        for pos in positions:
            exit_price = float(self.data[-1][4])
            pnl = pos.calculate_pnl(exit_price)
            capital += pnl
            
            pnl_pct = (pnl / pos.size_usdt) * 100
            bars_held = len(self.data) - 1 - pos.entry_bar
            
            closed_trades.append({
                "symbol": pos.symbol,
                "strategy": pos.strategy,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "exit_reason": "END",
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "bars_held": bars_held,
                "power": pos.power
            })
        
        # Calculate statistics
        return self._calculate_statistics(strategy_name, closed_trades, capital)
    
    def _calculate_statistics(self, strategy_name: str, 
                             trades: List[Dict], final_capital: float) -> Dict:
        """Calculate performance statistics from trades"""
        if not trades:
            return self._empty_results(strategy_name)
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        
        tp_trades = [t for t in trades if t["exit_reason"] == "TP"]
        sl_trades = [t for t in trades if t["exit_reason"] == "SL"]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        tp_rate = (len(tp_trades) / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t["pnl"] for t in trades)
        total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        avg_bars_held = np.mean([t["bars_held"] for t in trades]) if trades else 0
        avg_power = np.mean([t["power"] for t in trades]) if trades else 0
        
        # Direction breakdown
        long_trades = [t for t in trades if t["direction"] == "UP"]
        short_trades = [t for t in trades if t["direction"] == "DOWN"]
        
        long_wins = len([t for t in long_trades if t["pnl"] > 0])
        short_wins = len([t for t in short_trades if t["pnl"] > 0])
        
        long_win_rate = (long_wins / len(long_trades) * 100) if long_trades else 0
        short_win_rate = (short_wins / len(short_trades) * 100) if short_trades else 0
        
        return {
            "strategy": strategy_name,
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": win_rate,
            "tp_rate": tp_rate,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "final_capital": final_capital,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_bars_held": avg_bars_held,
            "avg_power": avg_power,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "tp_count": len(tp_trades),
            "sl_count": len(sl_trades)
        }
    
    def _empty_results(self, strategy_name: str) -> Dict:
        """Return empty results for failed strategies"""
        return {
            "strategy": strategy_name,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "tp_rate": 0,
            "total_pnl": 0,
            "total_return_pct": 0,
            "final_capital": INITIAL_CAPITAL,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_bars_held": 0,
            "avg_power": 0,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": 0,
            "short_win_rate": 0,
            "tp_count": 0,
            "sl_count": 0
        }


def run_comprehensive_backtest(symbols: List[str], lookback_days: int = 90):
    """Run backtest for all strategies on multiple symbols"""
    
    # Define all strategies to test
    strategies = [
        ("UT/STC", build_utstc_signal),
        ("MACD", build_macd_trend_signal),
        ("FVG", build_fvg_break_signal),
        ("EMA_PULLBACK", build_ema_pullback_signal),
        ("KIVANC_CONFIRM", build_kivanc_confirm_signal),
        ("CEST", build_cest_signal),
        ("EMA_STRUCTURE", build_ema_structure_signal)
    ]
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"BACKTESTING {symbol}")
        print(f"{'='*60}")
        
        # Create backtest engine for this symbol
        engine = BacktestEngine(symbol, interval="1h", lookback_days=lookback_days)
        
        # Fetch historical data
        if not engine.fetch_historical_data():
            print(f"Failed to fetch data for {symbol}, skipping...")
            continue
        
        # Test each strategy
        for strategy_name, strategy_func in strategies:
            try:
                result = engine.run_strategy_backtest(strategy_name, strategy_func)
                result["symbol"] = symbol
                all_results.append(result)
                
                # Print quick summary
                print(f"  {strategy_name:20} | Trades: {result['total_trades']:3} | "
                      f"Win%: {result['win_rate']:5.1f}% | "
                      f"Return: {result['total_return_pct']:+6.2f}% | "
                      f"PF: {result['profit_factor']:.2f}")
                
            except Exception as e:
                print(f"  {strategy_name:20} | ERROR: {e}")
    
    return all_results


def print_summary_report(results: List[Dict]):
    """Print comprehensive summary report"""
    
    print("\n" + "="*80)
    print("BACKTEST SUMMARY REPORT")
    print("="*80)
    
    # Aggregate by strategy across all symbols
    strategy_aggregates = {}
    
    for r in results:
        strategy = r["strategy"]
        if strategy not in strategy_aggregates:
            strategy_aggregates[strategy] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0,
                "tp_count": 0,
                "sl_count": 0,
                "symbols_tested": 0
            }
        
        agg = strategy_aggregates[strategy]
        agg["total_trades"] += r["total_trades"]
        agg["winning_trades"] += r["winning_trades"]
        agg["losing_trades"] += r["losing_trades"]
        agg["total_pnl"] += r["total_pnl"]
        agg["tp_count"] += r["tp_count"]
        agg["sl_count"] += r["sl_count"]
        agg["symbols_tested"] += 1
    
    # Calculate aggregate statistics
    print("\nSTRATEGY PERFORMANCE SUMMARY:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Trades':>8} {'Win%':>8} {'TP%':>8} {'PnL':>12} {'Symbols':>8}")
    print("-" * 80)
    
    sorted_strategies = sorted(
        strategy_aggregates.items(),
        key=lambda x: x[1]["total_pnl"],
        reverse=True
    )
    
    for strategy, agg in sorted_strategies:
        win_rate = (agg["winning_trades"] / agg["total_trades"] * 100) if agg["total_trades"] > 0 else 0
        tp_rate = (agg["tp_count"] / agg["total_trades"] * 100) if agg["total_trades"] > 0 else 0
        
        print(f"{strategy:<20} {agg['total_trades']:8} {win_rate:7.1f}% {tp_rate:7.1f}% "
              f"{agg['total_pnl']:+11.2f} {agg['symbols_tested']:8}")
    
    print("-" * 80)
    
    # Find best strategy
    if sorted_strategies:
        best_strategy = sorted_strategies[0]
        print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy[0]}")
        print(f"   Total PnL: ${best_strategy[1]['total_pnl']:.2f}")
        print(f"   Total Trades: {best_strategy[1]['total_trades']}")
        win_rate = (best_strategy[1]["winning_trades"] / best_strategy[1]["total_trades"] * 100) if best_strategy[1]["total_trades"] > 0 else 0
        print(f"   Win Rate: {win_rate:.2f}%")
    
    # Detailed per-symbol, per-strategy results
    print("\n\nDETAILED RESULTS BY SYMBOL AND STRATEGY:")
    print("="*80)
    
    # Group by symbol
    by_symbol = {}
    for r in results:
        symbol = r["symbol"]
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(r)
    
    for symbol in sorted(by_symbol.keys()):
        print(f"\n{symbol}:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Trades':>8} {'Win':>5} {'Loss':>5} {'Win%':>8} "
              f"{'Return%':>10} {'PF':>8} {'Avg Bars':>10}")
        print("-" * 80)
        
        symbol_results = sorted(by_symbol[symbol], 
                               key=lambda x: x["total_return_pct"], 
                               reverse=True)
        
        for r in symbol_results:
            print(f"{r['strategy']:<20} {r['total_trades']:8} {r['winning_trades']:5} "
                  f"{r['losing_trades']:5} {r['win_rate']:7.1f}% {r['total_return_pct']:+9.2f}% "
                  f"{r['profit_factor']:7.2f} {r['avg_bars_held']:9.1f}")


def save_results_to_json(results: List[Dict], filename: str = "backtest_results.json"):
    """Save results to JSON file"""
    output_path = os.path.join(os.path.dirname(__file__), filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")


def main():
    """Main backtest execution"""
    print("="*80)
    print("EMA STRATEGY BACKTEST FRAMEWORK")
    print("="*80)
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Trade Size: ${TRADE_SIZE_USDT:.2f}")
    print(f"Commission Rate: {COMMISSION_RATE*100:.2f}%")
    print("="*80)
    
    # Test on popular symbols
    test_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "ADAUSDT"
    ]
    
    # Run comprehensive backtest
    results = run_comprehensive_backtest(test_symbols, lookback_days=90)
    
    # Print summary
    print_summary_report(results)
    
    # Save results
    save_results_to_json(results)
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
