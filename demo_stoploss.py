#!/usr/bin/env python3
"""
Demonstration of stop loss calculation with sample data.
Shows how the feature works with realistic max_loss values.
"""

import json

def safe_float(value, default=0.0):
    """Safely convert any value to float"""
    try:
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default

def calculate_avg_max_loss(closed_trades):
    """Calculate the average max loss from all closed trades."""
    if not closed_trades:
        return 0.0
    
    max_losses = []
    for trade in closed_trades:
        max_loss = safe_float(trade.get("max_loss", 0.0))
        if max_loss <= 0:  # Only include losses
            max_losses.append(max_loss)
    
    if not max_losses:
        return 0.0
    
    avg_max_loss = sum(max_losses) / len(max_losses)
    return avg_max_loss

def get_recommended_stop_loss(closed_trades, trade_size=500.0):
    """Calculate recommended stop loss based on historical data."""
    avg_max_loss = calculate_avg_max_loss(closed_trades)
    buffer_pct = 20.0
    recommended_sl_usd = abs(avg_max_loss) * (1 + buffer_pct / 100)
    recommended_sl_pct = (recommended_sl_usd / trade_size) * 100 if trade_size > 0 else 0.0
    
    sample_size = len([t for t in closed_trades if safe_float(t.get("max_loss", 0.0)) <= 0])
    
    return {
        "avg_max_loss_usd": avg_max_loss,
        "recommended_sl_usd": recommended_sl_usd,
        "recommended_sl_pct": recommended_sl_pct,
        "trade_size_usdt": trade_size,
        "buffer_pct": buffer_pct,
        "sample_size": sample_size
    }

def main():
    print("=" * 70)
    print("STOP LOSS CALCULATION DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create sample data with realistic max_loss values
    # Simulating scenarios where trades went negative before closing
    sample_trades = [
        # Winning trade that went negative first (max_loss = -12 as per problem statement)
        {"symbol": "BTCUSDT", "exit_reason": "TP", "pnl_pct": 1.2, "max_profit": 15.0, "max_loss": -12.0},
        # Another winning trade with smaller drawdown
        {"symbol": "ETHUSDT", "exit_reason": "TP", "pnl_pct": 0.8, "max_profit": 10.0, "max_loss": -5.0},
        # Winning trade with larger drawdown
        {"symbol": "BNBUSDT", "exit_reason": "TP", "pnl_pct": 2.0, "max_profit": 25.0, "max_loss": -18.0},
        # Losing trade
        {"symbol": "ADAUSDT", "exit_reason": "SL", "pnl_pct": -3.5, "max_profit": 2.0, "max_loss": -35.0},
        # Winning trade with minimal drawdown
        {"symbol": "SOLUSDT", "exit_reason": "TP", "pnl_pct": 1.5, "max_profit": 18.0, "max_loss": -3.0},
        # Losing trade with large loss
        {"symbol": "DOTUSDT", "exit_reason": "SL", "pnl_pct": -2.8, "max_profit": 5.0, "max_loss": -28.0},
        # Winning trade
        {"symbol": "LINKUSDT", "exit_reason": "TP", "pnl_pct": 0.9, "max_profit": 12.0, "max_loss": -8.0},
        # Winning trade with moderate drawdown
        {"symbol": "MATICUSDT", "exit_reason": "TP", "pnl_pct": 1.1, "max_profit": 14.0, "max_loss": -10.0},
        # Losing trade
        {"symbol": "AVAXUSDT", "exit_reason": "SL", "pnl_pct": -4.0, "max_profit": 3.0, "max_loss": -40.0},
        # Winning trade
        {"symbol": "ATOMUSDT", "exit_reason": "TP", "pnl_pct": 1.3, "max_profit": 16.0, "max_loss": -7.0},
    ]
    
    print(f"Sample data: {len(sample_trades)} closed trades")
    print()
    print("Trade Summary:")
    print("-" * 70)
    for i, trade in enumerate(sample_trades, 1):
        result = "✅ WIN" if trade["exit_reason"] == "TP" else "❌ LOSS"
        print(f"{i:2d}. {trade['symbol']:12s} | {result} | "
              f"PnL: {trade['pnl_pct']:+6.2f}% | "
              f"Max Profit: ${trade['max_profit']:5.1f} | "
              f"Max Loss: ${trade['max_loss']:6.1f}")
    
    print()
    print("=" * 70)
    
    # Calculate stop loss recommendation
    result = get_recommended_stop_loss(sample_trades, trade_size=500.0)
    
    print()
    print("STOP LOSS ANALYSIS RESULTS")
    print("=" * 70)
    print(f"📊 Trades analyzed: {result['sample_size']} (all have max_loss data)")
    print(f"💰 Trade Size: ${result['trade_size_usdt']:.0f}")
    print(f"📉 Average Max Loss: ${abs(result['avg_max_loss_usd']):.2f}")
    print(f"🛡️  Safety Buffer: {result['buffer_pct']:.0f}%")
    print()
    print("✅ RECOMMENDED STOP LOSS:")
    print(f"   💵 ${result['recommended_sl_usd']:.2f} per trade")
    print(f"   📊 {result['recommended_sl_pct']:.2f}% of ${result['trade_size_usdt']:.0f} trade size")
    print()
    print("📝 EXPLANATION:")
    print(f"   • Your trades historically went down to an average of ${abs(result['avg_max_loss_usd']):.2f}")
    print(f"     before either recovering to profit or hitting stop loss.")
    print(f"   • Adding a {result['buffer_pct']:.0f}% safety buffer gives: ${result['recommended_sl_usd']:.2f}")
    print(f"   • This represents {result['recommended_sl_pct']:.2f}% of your ${result['trade_size_usdt']:.0f} trade size.")
    print(f"   • Setting your stop loss at this level would allow normal market")
    print(f"     fluctuations while protecting against excessive losses.")
    print()
    print("🎯 USAGE:")
    print("   Use this recommended stop loss percentage when configuring your")
    print("   trading parameters to better manage risk based on your actual")
    print("   trading history.")
    print("=" * 70)
    
    # Show distribution
    print()
    print("MAX LOSS DISTRIBUTION:")
    print("-" * 70)
    losses = sorted([abs(t["max_loss"]) for t in sample_trades])
    print(f"Minimum: ${losses[0]:.2f}")
    print(f"Maximum: ${losses[-1]:.2f}")
    print(f"Median:  ${losses[len(losses)//2]:.2f}")
    print(f"Average: ${abs(result['avg_max_loss_usd']):.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
