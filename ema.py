You are an expert crypto trading system developer.

I have an existing trading bot (ema.py) that already includes:

* state machine logic (BREAKOUT_PENDING → RETEST_PENDING → CONFIRMED_LONG / FAKE_BREAKOUT)
* support bounce pattern detection
* EMA20 / EMA50 trend filters
* ATR-based overextension filter
* scoring system (VALID / WATCHLIST)

Your task is to IMPROVE and REFINE the LONG ENTRY STRATEGY based on real market behavior.

---

🎯 CORE STRATEGY:
Impulse → Pullback → Support Hold → Bounce → Long Entry

---

🔹 1. IMPULSE DETECTION

* Detect a bullish impulse move
* Minimum move: configurable (default 6–8%)
* Must be a clean move (not choppy candles)
* Optional: consecutive bullish candles or strong range expansion

---

🔹 2. PULLBACK VALIDATION

* Measure retracement from swing high to swing low
* Ideal zone: Fibonacci 0.382 – 0.618
* Acceptable max: 0.70
* Reject if retracement > 0.75 (structure broken)

---

🔹 3. SUPPORT CONFIRMATION
At pullback zone, confirm at least ONE of:

* Multiple candle closes above support
* Long lower wick (liquidity sweep)
* Range compression / consolidation

---

🔹 4. BOUNCE CONFIRMATION (VERY IMPORTANT)
Do NOT trigger entry without confirmation.

At least one of:

* Strong bullish candle (close > previous close)
* Two consecutive higher closes
* Bullish engulfing or momentum shift

---

🔹 5. TREND FILTER (EMA)

* Price should be above EMA20
* EMA20 > EMA50 preferred
* If below EMA20 → downgrade signal (WATCHLIST only)

---

🔹 6. VOLUME CONFIRMATION (BONUS, NOT MANDATORY)

* Volume spike above recent average increases score

---

🔹 7. OVEREXTENSION FILTER (CRITICAL)

* Do NOT enter if price is too far from support
* Use:

  * ATR-based distance OR
  * % distance (default ~3%)
* If overextended → mark as OVEREXTENDED_NO_ENTRY

---

🔹 8. BREAKOUT + RETEST LOGIC (IMPORTANT)

* If resistance is broken:
  → DO NOT immediately long
  → Wait for retest of breakout level
* Valid long only if:

  * price holds above level
  * bullish confirmation appears

---

🔹 9. STATE MACHINE (MUST KEEP)
Use or improve:

* BREAKOUT_PENDING
* RETEST_PENDING
* CONFIRMED_LONG
* FAKE_BREAKOUT_LONG
* OVEREXTENDED_NO_ENTRY

Prevent duplicate signals:

* Once signal is triggered, do not re-trigger for same structure
* Track coin for at least 15 minutes after breakout

---

🔹 10. SCORING SYSTEM
Return structured output:

* VALID (strong signal)
* WATCHLIST (medium confidence)
* REJECT (invalid)

Score factors:

* impulse strength
* retracement quality
* bounce strength
* EMA alignment
* volume confirmation

---

🔹 11. OUTPUT FORMAT
Return clean structured object:

{
"symbol": "...",
"pattern": "IMPULSE → PULLBACK → BOUNCE",
"state": "CONFIRMED_LONG / WATCHLIST / FAKE_BREAKOUT",
"entry": price,
"support": level,
"invalid": level,
"score": 0-100,
"reason": "short explanation"
}

---

🔹 12. IMPLEMENTATION RULES

* DO NOT remove any existing logic from ema.py
* ONLY extend or improve current functions
* Keep modular structure
* Write clean, production-ready Python code
* Make thresholds configurable

---

🎯 GOAL:
Reduce fake breakouts, avoid chasing pumps, and enter only after confirmed support + bounce behavior.

Focus on high-probability LONG setups only.
