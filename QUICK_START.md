# Stop Loss Calculator - Quick Start Guide

## What It Does
Calculates recommended stop loss levels based on your actual trading history.

## How to Use

### 1. View Current Recommendation
Send this command in Telegram:
```
/stoploss
```

### 2. What You'll See
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
```

### 3. What It Means
- Your trades historically went down $16.60 on average before closing
- Recommended stop loss: $19.92 (includes 20% safety buffer)
- This is 3.98% of your $500 trade size
- Use this to set your SCALP_SL_PCT parameter

### 4. Automatic Updates
- System tracks max loss for all trades automatically
- Telegram notification sent once per 24 hours
- Recommendations improve as more data is collected

## How It Works

1. **While Trade is Open:**
   - System tracks lowest point (max loss) every 30 seconds
   - Example: Entry at $100, drops to $88 = max_loss: -$12

2. **When Trade Closes:**
   - Stores max_loss in trade history
   - Example: Trade recovers to $101, closes with profit
   - Saved: pnl_pct: +1%, max_loss: -$12

3. **Calculate Recommendation:**
   - Analyzes all closed trades
   - Average max loss = $16.60 (from all trades)
   - Add 20% buffer = $19.92
   - Convert to % = 3.98% of trade size

## Testing

Run demonstration:
```bash
python3 demo_stoploss.py
```

## Files

- **STOPLOSS_FEATURE.md** - Complete documentation
- **IMPLEMENTATION_SUMMARY.md** - Technical details
- **demo_stoploss.py** - Try it with sample data
- **test_stoploss.py** - Test with your data

## Benefits

✅ Data-driven decisions (not arbitrary percentages)  
✅ Prevents setting stops too tight (premature exits)  
✅ Prevents setting stops too loose (excessive losses)  
✅ Adapts to your trading style automatically  
✅ Improves as you collect more data  

## Example Scenario

**Without This Feature:**
- Set stop loss at 5% (arbitrary)
- Get stopped out frequently on normal dips
- OR set at 10% and lose too much

**With This Feature:**
- Your data shows trades typically dip to -$16.60
- Recommended: 3.98% (accommodates your pattern)
- Result: Fewer false stops, better risk management
