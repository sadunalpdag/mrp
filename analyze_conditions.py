#!/usr/bin/env python3
"""
Strategy Condition Analysis Tool

This script analyzes closed trades with their strategy-specific condition parameters
to identify patterns that correlate with successful (TP) vs unsuccessful (SL) closures.

Usage:
    python3 analyze_conditions.py [--data-dir DATA_DIR]

Example output:
    - Success rate by strategy
    - Correlation between condition parameters and TP/SL outcomes
    - Optimal parameter ranges for each strategy
"""

import json
import os
import sys
from collections import defaultdict
import pandas as pd

def load_closed_trades(data_dir="./data"):
    """Load closed trades from real_closed.json"""
    file_path = os.path.join(data_dir, "real_closed.json")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("ℹ️  This file will be created once the bot starts trading and positions close.")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []


def analyze_strategy_performance(trades):
    """Analyze success rates by strategy"""
    if not trades:
        return
    
    strategy_stats = defaultdict(lambda: {"total": 0, "tp": 0, "sl": 0, "other": 0})
    
    for trade in trades:
        strategy = trade.get("strategy", "UNKNOWN")
        
        # Determine outcome based on PnL or exit_reason
        pnl = trade.get("pnl_pct")
        exit_reason = trade.get("exit_reason", "")
        
        strategy_stats[strategy]["total"] += 1
        
        if exit_reason == "PROFIT_TARGET" or (pnl and pnl > 0):
            strategy_stats[strategy]["tp"] += 1
        elif pnl and pnl < 0:
            strategy_stats[strategy]["sl"] += 1
        else:
            strategy_stats[strategy]["other"] += 1
    
    print("\n" + "="*70)
    print("📊 STRATEGY PERFORMANCE ANALYSIS")
    print("="*70)
    
    for strategy, stats in sorted(strategy_stats.items()):
        total = stats["total"]
        tp_count = stats["tp"]
        success_rate = (tp_count / total * 100) if total > 0 else 0
        
        print(f"\n{strategy}:")
        print(f"  Total trades: {total}")
        print(f"  ✅ Successful (TP): {tp_count} ({success_rate:.1f}%)")
        print(f"  ❌ Failed (SL): {stats['sl']}")
        print(f"  ➖ Other: {stats['other']}")


def analyze_condition_parameters(trades):
    """Analyze condition parameters and their correlation with success"""
    if not trades:
        return
    
    print("\n" + "="*70)
    print("📈 CONDITION PARAMETERS ANALYSIS")
    print("="*70)
    
    # Group trades by strategy
    strategy_trades = defaultdict(list)
    for trade in trades:
        if "conditions" in trade and trade.get("conditions"):
            strategy = trade.get("strategy", "UNKNOWN")
            strategy_trades[strategy].append(trade)
    
    if not strategy_trades:
        print("\nℹ️  No condition parameters found in closed trades yet.")
        print("    Condition tracking was recently added and will be available")
        print("    for new trades as they close.")
        return
    
    for strategy, strat_trades in sorted(strategy_trades.items()):
        print(f"\n🔹 {strategy} ({len(strat_trades)} trades with conditions)")
        print("-" * 70)
        
        # Separate successful and failed trades
        successful = [t for t in strat_trades if (t.get("pnl_pct") or 0) > 0]
        failed = [t for t in strat_trades if (t.get("pnl_pct") or 0) < 0]
        
        if not successful and not failed:
            print("  ⚠️  Not enough data to analyze yet")
            continue
        
        # Get common condition parameters
        if successful:
            sample_conditions = successful[0].get("conditions", {})
            numeric_params = [k for k, v in sample_conditions.items() 
                            if isinstance(v, (int, float)) and not isinstance(v, bool)]
            
            print(f"  Successful: {len(successful)} | Failed: {len(failed)}")
            print(f"  Success rate: {len(successful)/(len(successful)+len(failed))*100:.1f}%")
            
            # Analyze numeric parameters
            if numeric_params:
                print(f"\n  Key condition parameters:")
                for param in numeric_params[:5]:  # Show top 5 numeric params
                    success_vals = [t["conditions"].get(param) for t in successful 
                                  if param in t.get("conditions", {})]
                    fail_vals = [t["conditions"].get(param) for t in failed 
                               if param in t.get("conditions", {})]
                    
                    success_vals = [v for v in success_vals if v is not None]
                    fail_vals = [v for v in fail_vals if v is not None]
                    
                    if success_vals and fail_vals:
                        avg_success = sum(success_vals) / len(success_vals)
                        avg_fail = sum(fail_vals) / len(fail_vals)
                        
                        print(f"    • {param}:")
                        print(f"      ✅ Avg (successful): {avg_success:.4f}")
                        print(f"      ❌ Avg (failed): {avg_fail:.4f}")
                        print(f"      📊 Difference: {abs(avg_success - avg_fail):.4f}")


def analyze_power_correlation(trades):
    """Analyze correlation between power score and success"""
    if not trades:
        return
    
    print("\n" + "="*70)
    print("⚡ POWER SCORE ANALYSIS")
    print("="*70)
    
    # Group trades by power bands
    power_bands = {
        "<60": [],
        "60-65": [],
        "65-70": [],
        "70-75": [],
        "75-80": [],
        ">80": []
    }
    
    for trade in trades:
        power = trade.get("power")
        if power is None:
            continue
        
        if power < 60:
            band = "<60"
        elif power < 65:
            band = "60-65"
        elif power < 70:
            band = "65-70"
        elif power < 75:
            band = "70-75"
        elif power < 80:
            band = "75-80"
        else:
            band = ">80"
        
        power_bands[band].append(trade)
    
    print("\nPower Band | Total | ✅ TP | ❌ SL | Success Rate")
    print("-" * 70)
    
    for band, band_trades in power_bands.items():
        if not band_trades:
            continue
        
        total = len(band_trades)
        tp_count = sum(1 for t in band_trades if (t.get("pnl_pct") or 0) > 0)
        sl_count = sum(1 for t in band_trades if (t.get("pnl_pct") or 0) < 0)
        success_rate = (tp_count / total * 100) if total > 0 else 0
        
        print(f"{band:10} | {total:5} | {tp_count:5} | {sl_count:5} | {success_rate:6.1f}%")


def export_to_csv(trades, output_file="closed_trades_analysis.csv"):
    """Export trades with condition parameters to CSV for external analysis"""
    if not trades:
        print("\n⚠️  No trades to export")
        return
    
    # Flatten the trades data
    flattened_trades = []
    for trade in trades:
        flat_trade = {
            "symbol": trade.get("symbol"),
            "strategy": trade.get("strategy"),
            "direction": trade.get("direction"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "pnl_pct": trade.get("pnl_pct"),
            "power": trade.get("power"),
            "exit_reason": trade.get("exit_reason"),
            "market_state": trade.get("market_state"),
            "open_time": trade.get("open_time"),
            "close_time": trade.get("close_time")
        }
        
        # Add condition parameters with prefix
        conditions = trade.get("conditions", {})
        for key, value in conditions.items():
            flat_trade[f"cond_{key}"] = value
        
        flattened_trades.append(flat_trade)
    
    try:
        df = pd.DataFrame(flattened_trades)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Exported {len(flattened_trades)} trades to {output_file}")
        print(f"   Use this CSV file for detailed analysis in Excel, Python pandas, etc.")
    except Exception as e:
        print(f"\n❌ Failed to export CSV: {e}")


def main():
    """Main analysis function"""
    # Parse command line arguments
    data_dir = "./data"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--data-dir" and len(sys.argv) > 2:
            data_dir = sys.argv[2]
    
    print("🔍 Loading closed trades data...")
    trades = load_closed_trades(data_dir)
    
    if not trades:
        print("\n⚠️  No closed trades found yet.")
        print("    This is normal if the bot hasn't closed any positions yet.")
        print("    Run this script again after some trades have closed.")
        return
    
    print(f"✅ Loaded {len(trades)} closed trades")
    
    # Run analyses
    analyze_strategy_performance(trades)
    analyze_power_correlation(trades)
    analyze_condition_parameters(trades)
    
    # Export to CSV
    print("\n" + "="*70)
    print("💾 DATA EXPORT")
    print("="*70)
    export_to_csv(trades)
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)
    print("\n📘 Tips for further analysis:")
    print("  • Use the CSV export for detailed statistical analysis")
    print("  • Track how condition parameters change over time")
    print("  • Identify optimal parameter ranges for each strategy")
    print("  • Compare market_state correlation with success rates")
    print("  • Analyze time-of-day patterns using open_time/close_time")


if __name__ == "__main__":
    main()
