# Strategy Condition Parameters Tracking

## Overview

This feature enables comprehensive tracking of strategy-specific condition parameters when trades are opened. This data is captured in the `real_closed.json` file and can be analyzed later to identify which conditions correlate with successful vs unsuccessful trade outcomes.

## What's Tracked

Each closed trade now includes a `conditions` dictionary containing strategy-specific parameters that were present when the trade was opened:

### CEST Strategy
- `ma50`: Moving Average 50 value
- `swing_low` / `swing_high`: Pattern swing levels
- `body_ratio`: Candle body to range ratio
- `has_fvg`: Whether Fair Value Gap was present
- `trend_4h`: 4H timeframe trend direction
- `tolerance`: Pattern matching tolerance
- `rr_ratio`: Risk/Reward ratio

### EMA_PULLBACK Strategy
- `ema9`, `ema30`, `ema200`: EMA indicator values
- `swing_high`, `swing_low`: Swing levels
- `uptrend` / `downtrend`: Trend direction flags

### MACD Strategy
- `ema20`, `ema200`: EMA indicator values
- `macd_line`, `macd_signal`: MACD indicator values
- `macd_prev`, `signal_prev`: Previous values for crossover detection
- `ema_spread`: Distance between EMA20 and EMA200

### FVG Strategy
- `gap_size`: Absolute gap size
- `gap_size_pct`: Gap size as percentage of price
- `h1`, `h2`, `h3`: High values of last 3 candles
- `l1`, `l2`, `l3`: Low values of last 3 candles
- `up_gap` / `dn_gap`: Gap direction flags

### ORB_FVG Strategy
- `or_high`, `or_low`: Opening range high/low
- `or_range`, `or_range_pct`: Opening range size
- `fvg_gap_size`: Fair Value Gap size
- `broke_high` / `broke_low`: Breakout flags
- `risk`, `rr_ratio`: Risk and risk/reward ratio

### LONDON_BO Strategy
- `lo_range_high`, `lo_range_low`: London session range
- `lo_range`, `lo_range_pct`: Range size
- `ema20`: EMA20 value
- `above_ema20`: Price position relative to EMA20
- `risk`, `rr_ratio`: Risk and risk/reward ratio

### NY_REVERSAL Strategy
- `sweep_level`: Liquidity sweep level
- `sweep_direction`: Direction of the sweep
- `risk`, `rr_ratio`: Risk and risk/reward ratio

### REENTRY_4H_5M Strategy
- `zone_high`, `zone_low`: 4H zone boundaries
- `zone_range`, `zone_range_pct`: Zone size
- `trend_4h`: 4H trend direction
- `in_kill_zone`: Whether entry occurred in kill zone
- `current_hour`: UTC hour of entry
- `risk`, `rr_ratio`: Risk and risk/reward ratio

### FVG_MSS_ENTRY Strategy
- `mss_level`: Market Structure Shift level
- `mss_direction`: MSS direction
- `ob_high`, `ob_low`: Order Block boundaries
- `ob_range`: Order Block size
- `fvg_gap_size`: Fair Value Gap size
- `fvg_zone`: FVG zone type (bullish/bearish)
- `has_bullish_fvg` / `has_bearish_fvg`: FVG type flags
- `risk`, `rr_ratio`: Risk and risk/reward ratio

## Data Structure

### Closed Trade Example
```json
{
  "symbol": "BTCUSDT",
  "direction": "UP",
  "strategy": "CEST",
  "tag": "🧩 C.E.S.T. BUY",
  "entry_price": 45000.0,
  "exit_price": 45300.0,
  "pnl_pct": 0.67,
  "power": 68.5,
  "open_time": "2024-11-16T10:00:00",
  "close_time": "2024-11-16T12:30:00",
  "exit_reason": "TP",
  "market_state": "PULLBACK",
  "conditions": {
    "ma50": 44800.0,
    "swing_low": 44500.0,
    "body_ratio": 0.72,
    "has_fvg": true,
    "trend_4h": "UP",
    "tolerance": 0.015,
    "rr_ratio": 1.4
  }
}
```

## Analysis Tool

Use the provided analysis script to examine condition parameters:

```bash
# Basic usage
python3 analyze_conditions.py

# Specify custom data directory
python3 analyze_conditions.py --data-dir /path/to/data
```

### Analysis Features

1. **Strategy Performance Analysis**
   - Success rates by strategy
   - TP vs SL breakdown

2. **Power Score Correlation**
   - Success rates by power bands
   - Identifies optimal power score ranges

3. **Condition Parameters Analysis**
   - Average parameter values for successful vs failed trades
   - Identifies which conditions correlate with success

4. **CSV Export**
   - Exports flattened data to `closed_trades_analysis.csv`
   - All condition parameters included as separate columns
   - Ready for analysis in Excel, Python pandas, R, etc.

### Example Output

```
======================================================================
📊 STRATEGY PERFORMANCE ANALYSIS
======================================================================

CEST:
  Total trades: 15
  ✅ Successful (TP): 12 (80.0%)
  ❌ Failed (SL): 3
  ➖ Other: 0

======================================================================
📈 CONDITION PARAMETERS ANALYSIS
======================================================================

🔹 CEST (15 trades with conditions)
----------------------------------------------------------------------
  Successful: 12 | Failed: 3
  Success rate: 80.0%

  Key condition parameters:
    • body_ratio:
      ✅ Avg (successful): 0.7250
      ❌ Avg (failed): 0.6100
      📊 Difference: 0.1150
    • rr_ratio:
      ✅ Avg (successful): 1.4500
      ❌ Avg (failed): 1.3800
      📊 Difference: 0.0700
```

## Use Cases

### 1. Optimize Strategy Parameters
Analyze which parameter ranges lead to higher success rates:
- Identify optimal body_ratio thresholds for CEST
- Find best RR ratios for each strategy
- Determine which market_state works best for pullback entries

### 2. Strategy Filtering
Use insights to add filters to existing strategies:
```python
# Example: Only trade CEST when body_ratio > 0.70
if strategy == "CEST" and body_ratio < 0.70:
    skip_trade()
```

### 3. Machine Learning
Export CSV data for ML model training:
- Predict trade outcome based on conditions
- Train models per strategy
- Identify non-obvious parameter correlations

### 4. Backtesting Improvements
Use historical condition data to:
- Test new parameter ranges
- Validate strategy modifications
- Compare before/after changes

### 5. Performance Monitoring
Track how conditions evolve over time:
- Identify degrading patterns
- Detect market regime changes
- Adjust strategies based on market conditions

## Integration with Trading Bot

The tracking is automatic and transparent:

1. **Signal Generation**: Each strategy adds `conditions` dict to signal
2. **Position Opening**: Conditions stored in `REAL_POSITIONS_TRACKER`
3. **Position Closing**: Conditions included in `real_closed.json`

No manual intervention required. Just run the bot and analyze results later.

## Files Modified

- `ema.py`: All strategy signal builders updated to include conditions
- `execute_real_trade()`: Stores conditions in position tracker
- `check_and_log_real_closed_trades()`: Includes conditions in closed trades log
- `close_all_positions_at_market()`: Includes conditions for profit target closures

## Best Practices

1. **Regular Analysis**: Run analysis weekly to identify trends
2. **Parameter Tuning**: Use insights to adjust strategy parameters
3. **Strategy Selection**: Focus on strategies with best condition patterns
4. **Market Adaptation**: Adjust based on changing market conditions
5. **Documentation**: Keep notes on parameter changes and their effects

## Future Enhancements

Potential improvements:
- Real-time condition monitoring dashboard
- Automated parameter optimization
- Alert system for degrading condition patterns
- Integration with ML prediction models
- Comparative analysis across time periods

## Support

For issues or questions:
1. Check `real_closed.json` format matches expected structure
2. Verify `conditions` dict is present in closed trades
3. Ensure all required packages are installed (pandas)
4. Review analysis script output for warnings

---

**Note**: This feature was added to enable comprehensive post-trade analysis. Historical trades (before this update) won't have condition parameters, but all new trades will include them automatically.
