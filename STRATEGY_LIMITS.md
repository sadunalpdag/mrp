# Position Limits for New Strategies

## Overview
Added position limits for the 3 new strategies (Bollinger Bands, Stochastic RSI, and Fibonacci Retracement) to control risk and prevent over-exposure to any single strategy.

## Limits Configuration

Each new strategy has its own position limits (in addition to the global limits):

| Strategy | Long Limit | Short Limit |
|----------|------------|-------------|
| Bollinger Bands | 3 | 3 |
| Stochastic RSI | 3 | 3 |
| Fibonacci Retracement | 3 | 3 |

## How It Works

### 1. Hierarchy of Limits

The bot enforces limits at two levels:

1. **Global Limits** (apply to all strategies combined):
   - `MAX_BUY`: 45 (total long positions)
   - `MAX_SELL`: 45 (total short positions)

2. **Strategy-Specific Limits** (sub-limits within global):
   - CEST: 15 long / 15 short
   - Bollinger Bands: 3 long / 3 short
   - Stochastic RSI: 3 long / 3 short
   - Fibonacci: 3 long / 3 short

### 2. Limit Enforcement

Before opening a new position:
1. Check global limits first
2. If passed, check strategy-specific limit
3. Only open position if both checks pass

Example:
```
Total positions: 20 long (within 45 global limit) ✓
BB positions: 2 long (within 3 BB limit) ✓
→ Can open new BB long position
```

```
Total positions: 20 long (within 45 global limit) ✓
BB positions: 3 long (at 3 BB limit) ✗
→ Cannot open new BB long position
```

## Telegram Commands

### View Strategy Limits

Use `/strategies` to see all strategy limits:

```
📊 STRATEGY STATUS
━━━━━━━━━━━━━━━━
✅ MACD Trend
...
✅ Bollinger Bands
✅ Stochastic RSI
✅ Fibonacci Retracement

📌 Global Limits (All Strategies):
Long: 45
Short: 45

🧩 CEST Sub-Limits (within global):
Long: 15
Short: 15

📊 BB Sub-Limits (within global):
Long: 3
Short: 3

🔄 Stoch RSI Sub-Limits (within global):
Long: 3
Short: 3

📐 Fib Sub-Limits (within global):
Long: 3
Short: 3
```

### Change Strategy Limits

Use `/setlimits <type> <value>` to change limits:

**Examples:**

```
/setlimits bb_buy 5         → Set BB long limit to 5
/setlimits bb_sell 5        → Set BB short limit to 5
/setlimits stoch_rsi_buy 4  → Set Stoch RSI long limit to 4
/setlimits fib_buy 2        → Set Fib long limit to 2
```

**Available limit types:**
- `bb_buy`, `bb_sell` - Bollinger Bands
- `stoch_rsi_buy`, `stoch_rsi_sell` - Stochastic RSI
- `fib_buy`, `fib_sell` - Fibonacci Retracement
- `cest_buy`, `cest_sell` - CEST strategy
- `buy`, `sell` - Global limits

## Position Tracking

The bot tracks open positions by strategy type in real-time:

### Status Command (`/status`)
Shows current position counts:
```
General long:20/45 short:15/45
CEST long:5/15 short:3/15
BB long:2/3 short:1/3
Stoch RSI long:1/3 short:0/3
Fib long:3/3 short:2/3
```

### Hourly Margin Updates
Includes new strategy counts:
```
⏰ HOURLY MARGIN UPDATE
━━━━━━━━━━━━━━━━
💰 Current Profit: $50.00
...
📌 Open Positions: 25
🧩 CEST Long: 5/15
🧩 CEST Short: 3/15
📊 BB Long: 2/3
📊 BB Short: 1/3
🔄 Stoch RSI Long: 1/3
🔄 Stoch RSI Short: 0/3
📐 Fib Long: 3/3
📐 Fib Short: 2/3
```

## Implementation Details

### State Variables

New blocked flags added to `STATE_DEFAULT`:
```python
"bb_long_blocked": False
"bb_short_blocked": False
"stoch_rsi_long_blocked": False
"stoch_rsi_short_blocked": False
"fib_long_blocked": False
"fib_short_blocked": False
```

### Counting Logic

Position counts are updated in real-time by checking the strategy `kind`:
- `BOLLINGER_BANDS` → bb_long_count / bb_short_count
- `STOCHASTIC_RSI` → stoch_rsi_long_count / stoch_rsi_short_count
- `FIBONACCI_RETRACEMENT` → fib_long_count / fib_short_count

### Limit Checks

The `_can_direction()` function checks limits before allowing trades:
```python
if kind == "BOLLINGER_BANDS":
    if direction=="UP" and STATE.get("bb_long_blocked",False):
        log(f"[BB LIMIT] Bollinger Bands long positions blocked (max: 3)")
        return False
    # ... similar for short
```

## Why These Limits?

The 3 position limit per strategy:

1. **Risk Management**: Prevents over-concentration in any single strategy
2. **Diversification**: Ensures balanced exposure across all 13 strategies
3. **Performance Testing**: Allows each strategy to prove itself without dominating the portfolio
4. **Flexibility**: Can be adjusted via Telegram as strategies prove themselves

## Adjusting Limits

You can adjust limits based on:

1. **Strategy Performance**: Increase limits for high-performing strategies
2. **Market Conditions**: Reduce limits during high volatility
3. **Capital**: Scale limits based on account size
4. **Risk Tolerance**: Conservative = lower limits, Aggressive = higher limits

**Recommended approach:**
- Start with default (3 each)
- Monitor performance for 1-2 weeks
- Increase best performers to 5-7
- Keep or reduce underperformers

## Notes

- Strategy limits are checked **in addition to** global limits
- If global limit is reached, no strategy can open new positions
- Limits apply independently to long and short positions
- Limits are enforced at trade execution time
- Position close frees up the limit immediately
