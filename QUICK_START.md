# Quick Start Guide: Strategy Condition Analysis

## What's New?

Your trading bot now automatically tracks **strategy-specific condition parameters** for every trade. This lets you analyze later which conditions lead to successful vs unsuccessful trades.

## No Setup Required! 🎉

The tracking happens automatically. Just run your bot as usual:
```bash
python3 ema.py
```

## After Trades Close

Run the analysis tool to see insights:

```bash
python3 analyze_conditions.py
```

## What You'll See

### 1. Strategy Performance
Which strategies are winning?
```
CEST: 15 trades, 80% success rate ✅
FVG_MSS_ENTRY: 12 trades, 75% success rate ✅
MACD: 8 trades, 50% success rate ⚠️
```

### 2. Power Score Analysis
Which power scores work best?
```
Power 70-75: 85% success (best!)
Power 65-70: 70% success
Power 60-65: 55% success
```

### 3. Condition Analysis
Which conditions predict success?
```
CEST Strategy:
  body_ratio: Higher = Better
    Successful trades avg: 0.72
    Failed trades avg: 0.61
    
  FVG present: 85% success
  No FVG: 60% success
```

### 4. CSV Export
Get all data for Excel/Python analysis:
```
✅ Exported to: closed_trades_analysis.csv
```

## What Gets Tracked?

Different parameters for each strategy:

**CEST**: MA50 level, swing points, candle body quality, FVG presence, 4H trend
**FVG_MSS**: Market structure shift level, order blocks, FVG gaps  
**REENTRY**: 4H trend, zone boundaries, kill zone timing
**MACD**: EMA values, MACD crossover details
**And more...** (see STRATEGY_CONDITIONS_README.md for full list)

## Example: Using Insights

### Before Analysis
```python
# Trading any CEST signal
if cest_signal:
    open_trade()
```

### After Analysis (if you find body_ratio matters)
```python
# Only trade high-quality CEST setups
if cest_signal and body_ratio > 0.70:
    open_trade()  # Better win rate!
```

## When to Run Analysis

- **Weekly**: Check for patterns and trends
- **After 20+ trades**: Need enough data for insights
- **After market changes**: See if conditions shifted
- **Before parameter tuning**: Make data-driven decisions

## Files to Check

**For analysis:**
- `data/real_closed.json` - Raw trade data with conditions

**For results:**
- `closed_trades_analysis.csv` - Exported data for detailed analysis

## Three Documents Available

1. **This file (QUICK_START.md)** - You are here! 📍
2. **STRATEGY_CONDITIONS_README.md** - Complete reference guide
3. **IMPLEMENTATION_SUMMARY.md** - Technical implementation details

## Common Questions

**Q: Will this slow down my bot?**
A: No! Minimal performance impact (< 0.1%)

**Q: Do I need to change my bot settings?**
A: No! It works automatically with your current setup

**Q: What if I don't have many closed trades yet?**
A: That's normal! Just wait for more trades to close, then run analysis

**Q: Can I use Excel for analysis?**
A: Yes! The CSV export works with Excel, Google Sheets, etc.

**Q: How much data do I need for good insights?**
A: Minimum 20 closed trades per strategy for basic patterns. 50+ is better.

## Pro Tips

1. **Save analysis results** - Compare weekly to see trends
2. **Focus on volume** - Strategies with more trades give better insights
3. **Test one change at a time** - Easier to measure impact
4. **Document changes** - Keep notes on what you adjust and why
5. **Be patient** - Need time to collect meaningful data

## Need Help?

Check the detailed documentation:
```bash
cat STRATEGY_CONDITIONS_README.md  # Full parameter reference
cat IMPLEMENTATION_SUMMARY.md      # Technical details
```

---

**Remember**: This is a **powerful analysis tool**, not a magic bullet. Use insights to make informed decisions about strategy parameters, but always test changes carefully!

Happy trading and analyzing! 📊✨
