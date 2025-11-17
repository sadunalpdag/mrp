# Hourly Performance Analysis & Trade Scheduling

## Overview

This system tracks trading performance by hour of day (UTC) and automatically blocks poor-performing hours after a 2-week data collection period.

## Features

### 1. Automatic Data Collection
- Tracks all closed trades by opening hour (0-23 UTC)
- Records win rate, average PnL, and total trades per hour
- Tracks strategy-specific performance per hour
- Requires 2 weeks of data before activating hour-based filtering

### 2. Intelligent Hour Blocking
After 2 weeks, the system automatically:
- Analyzes performance for each hour
- Blocks hours that don't meet performance thresholds
- Prevents opening new trades during blocked hours
- Continues tracking to adjust blocked hours over time

### 3. Enhanced Status Reporting
The `/status` command now shows:
- Strategy-specific position counts
- Top 5 positions closest to TP target
- Hourly analysis status
- Data collection progress
- Currently blocked hours

## Telegram Commands

### View Hourly Statistics
```
/hourlystats
```
Shows detailed hourly performance including:
- Status (active or collecting data)
- Start date of data collection
- List of blocked hours
- Top 10 most active hours with their win rates and average PnL

### Block/Unblock Specific Hours
```
/blockhour <hour> [block|unblock]
```
Manually block or unblock specific hours.

Examples:
- `/blockhour 3 block` - Block hour 3 UTC
- `/blockhour 14 unblock` - Unblock hour 14 UTC

### Reset Statistics
```
/resethourlystats
```
Resets all hourly statistics and restarts the 2-week data collection period.

### Force Activate Analysis
```
/forcehourlyanalysis
```
Bypasses the 2-week wait and immediately activates hourly analysis based on current data.

## Configuration Parameters

You can adjust these parameters via `/set` command:

### HOURLY_MIN_TRADES (default: 20)
Minimum number of trades required in an hour before considering it for blocking.
```
/set HOURLY_MIN_TRADES 30
```

### HOURLY_MIN_WIN_RATE (default: 40.0)
Minimum win rate percentage. Hours below this threshold will be blocked.
```
/set HOURLY_MIN_WIN_RATE 45.0
```

### HOURLY_MIN_AVG_PNL (default: -0.5)
Minimum average PnL percentage. Hours below this threshold will be blocked.
```
/set HOURLY_MIN_AVG_PNL 0.0
```

## How It Works

### Phase 1: Data Collection (First 2 Weeks)
1. System starts tracking all trades by hour
2. No hours are blocked during this phase
3. All strategies operate normally
4. Progress shown in `/status` command

### Phase 2: Analysis Active (After 2 Weeks)
1. System automatically activates hourly analysis
2. Calculates performance metrics for each hour
3. Blocks hours that don't meet thresholds:
   - Less than minimum trades → ignored (not enough data)
   - Win rate < threshold OR avg PnL < threshold → blocked
4. Sends notification via Telegram when activated

### Phase 3: Ongoing Optimization
1. System continues tracking all trades
2. Periodically re-evaluates blocked hours
3. Can adapt to changing market conditions
4. Manual overrides available via commands

## Example Usage

### Check Status
```
/status
```
Shows:
```
📊 STATUS bar:1234 auto:✅
━━━━━━━━━━━━━━━━
General long:12/30 short:8/30
CEST long:5/15 short:3/15
Re-entry long:2/5 short:1/5
Closed trades:150
Unrealized PnL: $45.30

━━━━━ STRATEGY BREAKDOWN ━━━━━
MACD_LONG: 5
FVG_SHORT: 3
CEST_LONG: 4
...

━━━ CLOSEST TO TP (Top 5) ━━━
1. BTCUSDT (MACD LONG)
   Current: 0.45%, Target: 0.60%
   Distance: 0.15%
...

━━━━━ HOURLY ANALYSIS ━━━━━
✅ Active (Current hour: 13 UTC)
🚫 Blocked hours: [3, 8, 16, 22]
```

### View Detailed Statistics
```
/hourlystats
```
Shows:
```
📊 HOURLY PERFORMANCE STATS
━━━━━━━━━━━━━━━━
Status: ✅ Active
Start date: 2025-11-03T10:00:00+00:00
🚫 Blocked hours: [3, 8, 16, 22]

━━━━━━━━━━━━━━━━
Top 10 active hours:

✅ Hour 12: 45 trades, WR 65.2%, Avg 0.32%
✅ Hour 14: 42 trades, WR 58.3%, Avg 0.28%
🚫 Hour 08: 38 trades, WR 35.1%, Avg -0.15%
✅ Hour 16: 35 trades, WR 51.4%, Avg 0.12%
...
```

### Manually Block an Hour
```
/blockhour 2 block
```
Response:
```
✅ Hour 2 blocked for trading
```

### Reset and Start Over
```
/resethourlystats
```
Response:
```
✅ Hourly statistics reset!
Data collection will restart on next trade.
Analysis will activate after 2 weeks.
```

## Best Practices

1. **Let it collect data**: Wait for the full 2 weeks before evaluating performance
2. **Monitor `/hourlystats`**: Check regularly to understand your hour-by-hour performance
3. **Adjust thresholds**: Fine-tune `HOURLY_MIN_WIN_RATE` and `HOURLY_MIN_AVG_PNL` based on your strategy
4. **Manual overrides**: Use `/blockhour` to quickly block problem hours without waiting
5. **Reset if needed**: Use `/resethourlystats` if you make major strategy changes

## Technical Details

### Data Storage
- Statistics stored in `data/hourly_stats.json`
- Automatically backed up with regular data exports
- Survives bot restarts

### Performance Impact
- Minimal overhead (updates only on trade close)
- Hour check happens before trade execution (fast)
- Periodic activation check (every loop iteration)

### Time Zone
- All hours are in UTC (0-23)
- Ensure you understand your local time → UTC conversion
- Most crypto markets are 24/7, but volatility varies by hour

## Troubleshooting

### Analysis not activating after 2 weeks?
- Check `/status` to see current date and start date
- Use `/forcehourlyanalysis` to activate manually
- Verify data collection is working via `/hourlystats`

### Too many hours blocked?
- Lower `HOURLY_MIN_WIN_RATE` threshold
- Increase `HOURLY_MIN_AVG_PNL` threshold
- Use `/blockhour <hour> unblock` for specific hours

### Want to disable hour blocking?
- Use `/resethourlystats` and don't wait 2 weeks
- Or manually unblock all hours via `/blockhour <hour> unblock`
- Set `HOURLY_MIN_WIN_RATE` to 0.0 (blocks nothing)

## Integration with Strategies

The hourly blocking system works with ALL strategies:
- MACD
- FVG
- CEST
- EMA Pullback
- ORB+FVG
- London Breakout
- NY Reversal
- ICT Power of 3
- Asian Breakout
- FVG+Breaker
- Re-entry
- FVG+MSS

Each strategy's performance is tracked separately per hour, allowing you to see which strategies work best at which times.
