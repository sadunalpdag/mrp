# EMA Ultra v15.10.0 - Enhanced Trading Strategies

## Overview
This update adds 3 new advanced technical trading strategies to improve overall trading performance and diversification.

## New Strategies Added

### 1. 📊 Bollinger Bands Strategy

**Description:** Mean reversion and volatility breakout strategy using Bollinger Bands

**Entry Rules:**
- **Long (Mean Reversion):**
  - Price touches or breaks below lower band (oversold)
  - RSI < 35 confirms oversold condition
  - Entry when price closes back inside bands
  
- **Short (Mean Reversion):**
  - Price touches or breaks above upper band (overbought)
  - RSI > 65 confirms overbought condition
  - Entry when price closes back inside bands

- **Squeeze Breakout:**
  - Low bandwidth indicates consolidation
  - High volume breakout from squeeze
  - Entry on breakout in either direction

**TP/SL:** 
- Mean Reversion: Target middle band, 1:1.5 minimum RR
- Squeeze Breakout: 1:2.5 RR

**Power Score Factors:**
- Base: 60
- ATR/Price ratio contribution
- RSI divergence from neutral
- Volume spike bonus: +5
- Squeeze breakout bonus: +8

### 2. 🔄 Stochastic RSI Strategy

**Description:** Overbought/Oversold detection with trend filter

**Entry Rules:**
- **Long:**
  - Stoch RSI crosses above 20 from oversold
  - K line crosses above D line (bullish crossover)
  - Price above EMA 50 (trend filter)
  
- **Short:**
  - Stoch RSI crosses below 80 from overbought
  - K line crosses below D line (bearish crossover)
  - Price below EMA 50 (trend filter)

**TP/SL:** 1:2 Risk/Reward

**Power Score Factors:**
- Base: 62
- ATR/Price ratio contribution
- Crossover strength bonus
- Typical range: 65-80

### 3. 📐 Fibonacci Retracement Strategy

**Description:** Trend continuation entries at key Fibonacci levels

**Entry Rules:**
- **Long (in uptrend):**
  - Identify swing high and swing low
  - Price retraces to 0.618 or 0.786 Fibonacci level
  - Bullish reversal candle at Fib level
  - RSI > 40 (momentum confirmation)
  - Entry on bounce from Fib level
  
- **Short (in downtrend):**
  - Identify swing low and swing high
  - Price retraces to 0.618 or 0.786 Fibonacci level
  - Bearish reversal candle at Fib level
  - RSI < 60 (momentum confirmation)
  - Entry on rejection from Fib level

**TP/SL:** 
- Target previous swing high/low
- Minimum 1:2 RR
- Stop below/above Fib level + 1.5 ATR

**Power Score Factors:**
- Base: 65
- ATR/Price ratio contribution
- 0.786 level bonus: +5 (stronger than 0.618)
- Typical range: 68-85

## New Helper Functions

### 1. `calculate_signal_power()`

Advanced power scoring algorithm with multi-factor weighting:

**Inputs:**
- `base_power`: Base score (50-70)
- `atr_factor`: ATR/price ratio (0-150)
- `rsi_value`: RSI momentum (0-100)
- `volume_spike`: Boolean (+5 power)
- `trend_alignment`: Higher TF alignment (+6 power)
- `multi_timeframe_confirm`: Multi-TF confirmation (+7 power)
- `structure_quality`: Market structure quality 0-10 (+1.5x power)
- `risk_reward_ratio`: RR ratio (≥2.0: +8, ≥1.5: +4, ≥1.2: +2)
- `volatility_normalized`: Volatility score 0-1 (>0.7: -5, <0.3: +3)

**Output:** Power score capped at 50-100 range

### 2. `calculate_adaptive_position_size()`

ATR-based adaptive position sizing:

**Volatility Adjustment:**
- Low volatility (<1.5%): +20% size
- Medium volatility (1.5-2.5%): Base size
- High volatility (2.5-4%): -20% size
- Very high volatility (>4%): -40% size

**Power Score Adjustment:**
- 90+ power: +15% size
- 80-90 power: +10% size
- 70-80 power: +5% size
- 60-70 power: Base size
- <60 power: -10% size

**Bounds:** 0.5x to 2.0x of base size

## Configuration

All new strategies can be enabled/disabled via Telegram:

```
/strategies - List all strategies with status
/enable BB - Enable Bollinger Bands
/enable STOCH_RSI - Enable Stochastic RSI
/enable FIB - Enable Fibonacci Retracement
/disable BB - Disable Bollinger Bands
```

## Default Settings

All strategies are **enabled by default** in `PARAM_DEFAULT`:
- `"ENABLE_BB": True`
- `"ENABLE_STOCH_RSI": True`
- `"ENABLE_FIB": True`

## Testing

All new indicator functions and helper functions have been tested:
- ✅ Bollinger Bands calculation
- ✅ Stochastic RSI calculation
- ✅ Fibonacci levels calculation
- ✅ Signal power calculation
- ✅ Adaptive position sizing

Run tests with:
```bash
python3 /tmp/test_new_strategies.py
```

## Strategy Summary

Total Active Strategies: **13**

1. 📈 MACD Trend
2. 🟩 FVG Break
3. 📘 EMA Pullback
4. 🧩 C.E.S.T.
5. 🔥 ORB+FVG
6. 🔄 NY Reversal
7. ⚡ ICT Power of 3
8. 🧱 FVG+Breaker
9. 🔄 Re-entry 4H+5m
10. ⭐ FVG+MSS
11. 📊 Bollinger Bands (NEW)
12. 🔄 Stochastic RSI (NEW)
13. 📐 Fibonacci Retracement (NEW)

*Asian Session and London Breakout remain disabled per user request*

## Benefits of New Strategies

1. **Diversification:** Multiple technical approaches reduce correlation risk
2. **Volatility Adaptation:** BB and Stoch RSI excel in different volatility regimes
3. **Trend Capture:** Fibonacci helps catch trend continuations at optimal levels
4. **Risk Management:** Adaptive position sizing adjusts to market conditions
5. **Power Scoring:** More sophisticated signal quality assessment

## Version History

- **v15.10.0** - Added 3 new strategies + power scoring + adaptive sizing
- v15.9.71 - Previous version with 10 strategies
