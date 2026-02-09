#!/usr/bin/env python3
"""
Test script to demonstrate stop loss calculation functionality.
This script analyzes closed trades data and calculates recommended stop loss.
"""

import json
import os

def safe_float(value, default=0.0):
    """Safely convert any value to float"""
    try:
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default

def calculate_avg_max_loss(closed_trades):
    """
    Calculate the average max loss from all closed trades.
    
    Args:
        closed_trades: List of closed trade dictionaries
        
    Returns:
        tuple: (avg_max_loss, sample_size) where avg_max_loss is average (as negative dollar amount)
               and sample_size is number of trades with loss data
    """
    if not closed_trades:
        print("No closed trades data available")
        return 0.0, 0
    
    # Collect all max_loss values from closed trades
    max_losses = []
    for trade in closed_trades:
        max_loss = safe_float(trade.get("max_loss", 0.0))
        # Only include actual losses (negative values or zero)
        if max_loss <= 0:
            max_losses.append(max_loss)
    
    if not max_losses:
        print("No max loss data in closed trades")
        return 0.0, 0
    
    # Calculate average
    avg_max_loss = sum(max_losses) / len(max_losses)
    sample_size = len(max_losses)
    
    print(f"Analyzed {sample_size} trades with loss data")
    print(f"Average Max Loss: ${avg_max_loss:.2f}")
    
    return avg_max_loss, sample_size

def get_recommended_stop_loss(closed_trades, trade_size=500.0):
    """
    Calculate recommended stop loss based on historical max loss data.
    
    Args:
        closed_trades: List of closed trade dictionaries
        trade_size: Current trade size in USDT (default: 500.0)
        
    Returns:
        dict: Dictionary with stop loss recommendations
    """
    # Get average max loss from history (optimized to return both avg and count)
    avg_max_loss, sample_size = calculate_avg_max_loss(closed_trades)
    
    # Apply safety buffer (20% more conservative than average)
    buffer_pct = 20.0
    recommended_sl_usd = abs(avg_max_loss) * (1 + buffer_pct / 100)
    
    # Calculate as percentage of trade size
    recommended_sl_pct = (recommended_sl_usd / trade_size) * 100 if trade_size > 0 else 0.0
    
    return {
        "avg_max_loss_usd": avg_max_loss,
        "recommended_sl_usd": recommended_sl_usd,
        "recommended_sl_pct": recommended_sl_pct,
        "trade_size_usdt": trade_size,
        "buffer_pct": buffer_pct,
        "sample_size": sample_size
    }

def main():
    print("=" * 60)
    print("STOP LOSS CALCULATION TEST")
    print("=" * 60)
    print()
    
    # Try to load real_closed.json from data directory first
    data_dir = os.getenv("DATA_DIR", "./data")
    real_closed_file = os.path.join(data_dir, "real_closed.json")
    
    # If real_closed.json doesn't exist, use sim_closed.json from root
    if not os.path.exists(real_closed_file):
        print(f"real_closed.json not found at {real_closed_file}")
        print("Trying sim_closed.json from repository root for demonstration...")
        real_closed_file = "sim_closed.json"
    
    # Load closed trades data
    try:
        with open(real_closed_file, "r", encoding="utf-8") as f:
            closed_trades = json.load(f)
        print(f"Loaded {len(closed_trades)} closed trades from {real_closed_file}")
        print()
    except FileNotFoundError:
        print(f"Error: Could not find {real_closed_file}")
        print("Please ensure you have closed trades data to analyze.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {real_closed_file}: {e}")
        return
    
    # Calculate stop loss recommendation
    print("-" * 60)
    print("CALCULATING STOP LOSS RECOMMENDATION...")
    print("-" * 60)
    print()
    
    result = get_recommended_stop_loss(closed_trades, trade_size=500.0)
    
    print()
    print("=" * 60)
    print("STOP LOSS RECOMMENDATION")
    print("=" * 60)
    print(f"📊 Analysis based on: {result['sample_size']} trades with loss data")
    print(f"💰 Trade Size: ${result['trade_size_usdt']:.0f}")
    print(f"📉 Average Max Loss: ${abs(result['avg_max_loss_usd']):.2f}")
    print(f"🛡️  Safety Buffer: {result['buffer_pct']:.0f}%")
    print()
    print("✅ RECOMMENDED STOP LOSS:")
    print(f"   💵 ${result['recommended_sl_usd']:.2f} per trade")
    print(f"   📊 {result['recommended_sl_pct']:.2f}% of trade size")
    print()
    print("ℹ️  This recommendation is based on the average maximum loss")
    print(f"   experienced across all trades, plus a {result['buffer_pct']:.0f}% safety buffer.")
    print("=" * 60)

if __name__ == "__main__":
    main()
