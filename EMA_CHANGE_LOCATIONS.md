# EMA.PY Change Locations - Quick Reference

This document provides exact line numbers and locations for all changes made to ema.py.

## Table of Contents
1. [Recent Additions (Feb 2026)](#recent-additions)
2. [Modified Functions](#modified-functions)
3. [Configuration Changes](#configuration-changes)
4. [Line-by-Line Change Map](#line-by-line-change-map)

---

## Recent Additions (Feb 2026)

### New Indicator Functions
| Function | Lines | Description |
|----------|-------|-------------|
| `bollinger_bands()` | 260-289 | Calculate Bollinger Bands (middle, upper, lower, bandwidth) |
| `stochastic_rsi()` | 291-331 | Calculate Stochastic RSI (K and D lines) |
| `fibonacci_levels()` | 333-347 | Calculate Fibonacci retracement levels |
| `calculate_signal_power()` | 348-422 | Multi-factor power scoring algorithm |
| `calculate_adaptive_position_size()` | 424-489 | ATR-based adaptive position sizing |

### New Strategy Functions
| Function | Lines | Description |
|----------|-------|-------------|
| `build_bollinger_bands_signal()` | 2643-2844 | Bollinger Bands mean reversion & squeeze breakout |
| `build_stochastic_rsi_signal()` | 2846-2938 | Stochastic RSI overbought/oversold strategy |
| `build_fibonacci_retracement_signal()` | 2940-3060 | Fibonacci trend continuation strategy |

---

## Modified Functions

### Position Management
| Function | Lines | What Changed |
|----------|-------|--------------|
| `update_directional_limits()` | 3806-3854 | Added tracking for BB, STOCH_RSI, FIB position counts |
| `_can_direction()` | 5385-5432 | Added limit checks for new strategies |

### Strategy Scanning
| Function | Lines | What Changed |
|----------|-------|--------------|
| `scan_symbol()` | 3062-3104 | Added s_bb, s_stoch_rsi, s_fib strategy generation |

### Telegram Commands
| Function | Lines | What Changed |
|----------|-------|--------------|
| `_cmd_strategies()` | 4990-5033 | Added new strategies to display |
| `_cmd_setlimits()` | 5037-5117 | Added bb_buy, bb_sell, stoch_rsi_buy/sell, fib_buy/sell |
| `_cmd_enable()` | 4860-4885 | Added BB, STOCH_RSI, FIB to valid strategies |
| `_cmd_disable()` | 4887-4912 | Added BB, STOCH_RSI, FIB to valid strategies |

---

## Configuration Changes

### STATE_DEFAULT (Lines 3536-3543)
```python
# Added new blocked flags:
"bb_long_blocked": False
"bb_short_blocked": False
"stoch_rsi_long_blocked": False
"stoch_rsi_short_blocked": False
"fib_long_blocked": False
"fib_short_blocked": False
```

### PARAM_DEFAULT (Lines 3545-3570)
```python
# Added new limit parameters:
"MAX_BB_BUY": 3
"MAX_BB_SELL": 3
"MAX_STOCH_RSI_BUY": 3
"MAX_STOCH_RSI_SELL": 3
"MAX_FIB_BUY": 3
"MAX_FIB_SELL": 3

# Added strategy enable flags:
"ENABLE_BB": True
"ENABLE_STOCH_RSI": True
"ENABLE_FIB": True
```

---

## Line-by-Line Change Map

### Header Section (Lines 1-59)
- **Line 8:** Version updated from "v15.9.71" to "v15.10.0"
- **Lines 10-13:** Strategy count updated from 10 to 13
- **Lines 21-23:** Added new strategy descriptions (BB, STOCH_RSI, FIB)

### Indicators Section (Lines 155-500)
- **Lines 260-289:** NEW - `bollinger_bands()` function
- **Lines 275-287:** FIXED - Bandwidth calculation formula
- **Lines 291-331:** NEW - `stochastic_rsi()` function
- **Lines 333-347:** NEW - `fibonacci_levels()` function
- **Lines 348-422:** NEW - `calculate_signal_power()` function
- **Lines 424-489:** NEW - `calculate_adaptive_position_size()` function

### Strategy Implementation (Lines 2600-3100)
- **Lines 2643-2844:** NEW - `build_bollinger_bands_signal()`
  - Lines 2723-2728: FIXED - Added prev values for temporal alignment
  - Lines 2741-2768: FIXED - Updated comparisons to use prev values
- **Lines 2846-2938:** NEW - `build_stochastic_rsi_signal()`
  - Lines 2890-2903: FIXED - Simplified crossover conditions
- **Lines 2940-3060:** NEW - `build_fibonacci_retracement_signal()`
  - Lines 3007-3020: FIXED - Corrected downtrend level calculation

### Strategy Scanning (Lines 3062-3104)
- **Lines 3097:** Added `s_bb` strategy generation
- **Lines 3098:** Added `s_stoch_rsi` strategy generation
- **Lines 3099:** Added `s_fib` strategy generation
- **Lines 3101-3103:** Updated result collection to include new strategies

### Configuration (Lines 3536-3570)
- **Lines 3539-3541:** Added bb_long_blocked, bb_short_blocked
- **Lines 3542-3543:** Added stoch_rsi_long_blocked, stoch_rsi_short_blocked
- **Lines 3544-3545:** Added fib_long_blocked, fib_short_blocked
- **Lines 3548-3551:** Added MAX_BB_BUY/SELL, MAX_STOCH_RSI_BUY/SELL, MAX_FIB_BUY/SELL
- **Lines 3567-3569:** Added ENABLE_BB, ENABLE_STOCH_RSI, ENABLE_FIB

### Position Management (Lines 3800-3900)
- **Line 3806:** Updated live dict initialization with new count variables
- **Lines 3807-3809:** Added bb_long_count, bb_short_count, stoch_rsi counts, fib counts
- **Lines 3816-3825:** Added kind checking for BOLLINGER_BANDS (long positions)
- **Lines 3829-3838:** Added kind checking for STOCHASTIC_RSI (long positions)
- **Lines 3842-3851:** Added kind checking for FIBONACCI_RETRACEMENT (long positions)
- **Lines 3826-3840:** Added corresponding checks for short positions
- **Lines 3847-3852:** Added STATE updates for new blocked flags

### Limit Checking (Lines 5385-5432)
- **Lines 5404-5410:** Added BB limit checks
- **Lines 5412-5418:** Added Stoch RSI limit checks
- **Lines 5420-5426:** Added Fibonacci limit checks

### Hourly Margin Updates (Lines 4415-4540)
- **Lines 4420-4427:** Added count variables for new strategies
- **Lines 4428-4461:** Added position counting by strategy kind
- **Lines 4464-4469:** Added exception handling for new count variables
- **Lines 4475-4481:** Added new counts to balance_record
- **Lines 4534-4540:** Added new strategy counts to display message

### Telegram Commands (Lines 4850-5150)
- **Lines 4860-4876:** Updated /enable valid_strategies list
- **Lines 4887-4903:** Updated /disable valid_strategies list
- **Lines 5006-5008:** Added BB, STOCH_RSI, FIB to strategies list
- **Lines 5024-5031:** Added new strategy sub-limits to display
- **Lines 5040-5045:** Updated /setlimits usage message
- **Lines 5086-5113:** Added bb_buy, bb_sell, stoch_rsi_buy/sell, fib_buy/sell handlers

### Status Display (Lines 4778-4788)
- **Lines 4784-4787:** Added new strategy counts to status message

---

## Change Summary by Commit

### Commit 978b62c (Position Limits)
**Modified Sections:**
- Lines 3545-3551 (PARAM_DEFAULT)
- Lines 3536-3543 (STATE_DEFAULT)
- Lines 3806-3854 (update_directional_limits)
- Lines 4420-4481 (hourly margin tracking)
- Lines 4784-4787 (status display)
- Lines 5385-5432 (_can_direction)
- Lines 5006-5031 (_cmd_strategies)
- Lines 5037-5117 (_cmd_setlimits)

### Commit 0fa2da1 (Bug Fixes)
**Modified Sections:**
- Lines 275-290 (BB bandwidth formula)
- Lines 2723-2768 (BB temporal alignment)
- Lines 2890-2903 (Stoch RSI conditions)
- Lines 3007-3020 (Fibonacci downtrend)

### Commit a2898c2 (Advanced Algorithms)
**Added Sections:**
- Lines 348-422 (calculate_signal_power)
- Lines 424-489 (calculate_adaptive_position_size)

### Commit 39b8a1a (New Strategies)
**Added Sections:**
- Lines 260-289 (bollinger_bands indicator)
- Lines 291-331 (stochastic_rsi indicator)
- Lines 333-347 (fibonacci_levels indicator)
- Lines 2643-2844 (BB strategy)
- Lines 2846-2938 (Stoch RSI strategy)
- Lines 2940-3060 (Fibonacci strategy)
- Lines 3567-3569 (enable flags)
- Lines 3097-3103 (scan_symbol integration)

---

## Git Commands for Specific Changes

### View changes in a specific section:
```bash
# View indicator additions
git show 39b8a1a -- ema.py | grep -A 20 "def bollinger_bands"

# View position limit additions
git show 978b62c -- ema.py | grep -A 10 "MAX_BB_BUY"

# View bug fixes
git show 0fa2da1 -- ema.py
```

### Compare versions:
```bash
# Compare current with version before new strategies
git diff bca8961..HEAD -- ema.py

# See just the stat summary
git diff --stat bca8961..HEAD -- ema.py
```

---

## File Statistics

### Current ema.py (v15.10.0)
- **Total Lines:** ~5,764
- **Functions:** ~150+
- **Strategies:** 13
- **Indicators:** 20+

### Changes Since v15.9.71
- **Lines Added:** +862
- **Lines Removed:** -26
- **Net Change:** +836 lines
- **New Functions:** +6
- **Modified Functions:** ~15

---

Last Updated: February 5, 2026
