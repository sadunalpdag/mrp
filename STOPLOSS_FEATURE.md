# Stop Loss Calculation Feature

## Overview

This feature tracks the maximum loss for every order and calculates a recommended stop loss point based on historical trading data. This helps traders set more informed stop loss levels based on their actual trading patterns rather than arbitrary percentages.

## Problem Statement

The original requirement was:
> To calculate stop loss store the max loss for every order and average of their return point for example open an order it goes - 12 Store this value than this is closed so We can define a stop loss point general be careful for calculation string or integer error

## Implementation

### 1. Max Loss Tracking

For every open position, the system now tracks:
- **max_profit**: The highest unrealized profit reached (existing feature)
- **max_loss**: The lowest unrealized PnL reached (NEW)

This is tracked in real-time via the `update_max_profit_tracking()` function, which has been enhanced to also track max_loss.

### 2. Data Storage

When a position closes, the following information is saved to `real_closed.json`:
```json
{
  "symbol": "BTCUSDT",
  "direction": "UP",
  "strategy": "MACD",
  "entry_price": 50000.0,
  "exit_price": 50600.0,
  "pnl_pct": 1.2,
  "max_profit": 15.0,
  "max_loss": -12.0,  // NEW: Tracks the worst drawdown during the trade
  ...
}
```

### 3. Stop Loss Calculation

The system analyzes all closed trades to calculate:
- **Average Max Loss**: The average of all maximum losses across historical trades
- **Recommended Stop Loss**: Average max loss + 20% safety buffer
- **Stop Loss Percentage**: Recommended stop loss as a percentage of trade size

#### Calculation Formula:
```
Average Max Loss = Sum of all max_loss values / Number of trades
Recommended SL (USD) = |Average Max Loss| × 1.20  (20% buffer)
Recommended SL (%) = (Recommended SL USD / Trade Size) × 100
```

### 4. Type Safety

All calculations use the `safe_float()` function to prevent string/integer conversion errors:
```python
def safe_float(value, default=0.0):
    """Safely convert any value to float, preventing type errors"""
    try:
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default
```

## Usage

### Telegram Commands

#### View Stop Loss Recommendation
```
/stoploss
```

This command displays:
- Number of trades analyzed
- Current trade size
- Average max loss from history
- Recommended stop loss in USD and percentage
- Safety buffer applied

Example output:
```
📊 STOP LOSS RECOMMENDATION
━━━━━━━━━━━━━━━━
📈 Analysis based on 50 closed trades

💰 Trade Size: $500
📉 Avg Max Loss: $16.60
🛡️ Safety Buffer: 20%

✅ RECOMMENDED STOP LOSS:
   💵 $19.92
   📊 3.98% of trade size

ℹ️ This recommendation is based on the average maximum loss
experienced across all your closed trades, plus a 20% safety buffer.
```

### Automatic Logging

The system automatically:
1. **Tracks max_loss** for all open positions (every 30 seconds)
2. **Logs max_loss** when positions close
3. **Calculates and logs recommendations** in the system log as needed
4. **Sends Telegram notifications** with recommendations once per 24 hours

### Log Messages

When a position closes:
```
[REAL CLOSED] BTCUSDT UP Strategy:MACD PnL:1.20% Exit:50600.0 MaxProfit:$15.00 MaxLoss:$-12.00
```

When calculating recommendations:
```
[STOP LOSS CALC] Analyzed 50 trades, Avg Max Loss: $-16.60
[STOP LOSS RECOMMENDATION] Based on 50 trades: Avg Max Loss: $-16.60, Recommended SL: $19.92 (3.98% of trade size)
```

## Testing

Two test scripts are provided:

### 1. test_stoploss.py
Tests the feature with real historical data from `sim_closed.json` or `real_closed.json`:
```bash
python3 test_stoploss.py
```

### 2. demo_stoploss.py
Demonstrates the feature with realistic sample data:
```bash
python3 demo_stoploss.py
```

Example output:
```
STOP LOSS ANALYSIS RESULTS
======================================================================
📊 Trades analyzed: 10 (all have max_loss data)
💰 Trade Size: $500
📉 Average Max Loss: $16.60
🛡️  Safety Buffer: 20%

✅ RECOMMENDED STOP LOSS:
   💵 $19.92 per trade
   📊 3.98% of $500 trade size
```

## Code Changes

### Main Changes in ema.py:

1. **Enhanced `update_max_profit_tracking()` function** (lines ~3131-3195)
   - Now also tracks max_loss (minimum unrealized PnL)
   - Logs both max profit and max loss averages

2. **Updated position tracking initialization** (lines ~4048-4060, ~5433-5445)
   - Added `max_loss: 0.0` field to all new positions

3. **Enhanced closed trade logging** (lines ~3090-3119)
   - Added `max_loss` to closed trade data
   - Included max_loss in log messages

4. **New function: `calculate_avg_max_loss_from_history()`** (lines ~3562-3593)
   - Analyzes all closed trades
   - Calculates average max loss

5. **New function: `get_recommended_stop_loss()`** (lines ~3595-3649)
   - Calculates recommended stop loss with safety buffer
   - Returns comprehensive recommendation data

6. **New Telegram command: `/stoploss`** (lines ~5207-5236)
   - Displays stop loss recommendation on demand

7. **Periodic recommendation logging** (lines ~5691-5699)
   - Logs recommendations every 100 bar cycles

## Benefits

1. **Data-Driven Decisions**: Stop loss recommendations based on actual trading history, not arbitrary percentages
2. **Risk Management**: Helps prevent setting stop losses too tight (causing premature exits) or too loose (causing excessive losses)
3. **Adaptive**: Recommendations automatically improve as more trading data is collected
4. **Type Safety**: Prevents calculation errors from string/integer type mismatches
5. **Visibility**: Easy to monitor via Telegram commands and logs

## Example Scenario

```
Trader opens a position at $100
Price drops to $88 (max_loss = -$12)
Price recovers to $101 (max_profit = $1)
Position closes at $101

Stored data:
- pnl_pct: 1.0%
- max_profit: $1.00
- max_loss: -$12.00

After 50 similar trades with average max_loss of -$16.60:
- Recommended SL: $19.92 (with 20% buffer)
- This is 3.98% of $500 trade size
```

This means setting stop loss at ~4% would accommodate normal market fluctuations while protecting against excessive losses, based on actual trading patterns.

## Future Enhancements

Potential improvements:
1. Strategy-specific stop loss recommendations (different for MACD vs FVG, etc.)
2. Time-of-day based recommendations (different for Asian vs NY sessions)
3. Volatility-adjusted recommendations
4. Configurable safety buffer percentage via Telegram
5. Historical tracking of how recommendations improve over time
