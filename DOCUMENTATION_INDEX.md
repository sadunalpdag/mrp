# Documentation Index

This repository contains comprehensive documentation about the EMA trading bot and recent changes.

## 📚 Main Documentation Files

### Changes & Updates
1. **[EMA_CHANGES.md](EMA_CHANGES.md)** - Complete chronological summary of all ema.py changes
   - What changed, when, and why
   - Commit-by-commit breakdown
   - Statistics and impact analysis
   - Git commands for exploring changes

2. **[EMA_CHANGE_LOCATIONS.md](EMA_CHANGE_LOCATIONS.md)** - Exact line numbers for every change
   - Line-by-line change map
   - Function-by-function breakdown
   - Quick reference for specific changes
   - Section-by-section modifications

3. **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - Overall improvement summary
   - New features and strategies
   - Performance improvements
   - Position limits implementation
   - Getting started guide

### Strategy Documentation
4. **[STRATEGY_UPDATE.md](STRATEGY_UPDATE.md)** - Detailed strategy rules
   - Bollinger Bands strategy
   - Stochastic RSI strategy
   - Fibonacci Retracement strategy
   - Entry/exit rules and parameters

5. **[STRATEGY_LIMITS.md](STRATEGY_LIMITS.md)** - Position limit system
   - How limits work
   - Telegram commands
   - Configuration details
   - Usage examples

## 🔍 Quick Navigation

### "Where is ema.py changed?"
→ See **[EMA_CHANGES.md](EMA_CHANGES.md)** for complete answer

### "What are the exact line numbers?"
→ See **[EMA_CHANGE_LOCATIONS.md](EMA_CHANGE_LOCATIONS.md)**

### "What new features were added?"
→ See **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)**

### "How do the new strategies work?"
→ See **[STRATEGY_UPDATE.md](STRATEGY_UPDATE.md)**

### "How do I set position limits?"
→ See **[STRATEGY_LIMITS.md](STRATEGY_LIMITS.md)**

## 📊 Quick Stats

### Recent Changes (Feb 2026)
- **Lines Added:** 862
- **Lines Removed:** 26
- **New Functions:** 6
- **New Strategies:** 3
- **Version:** v15.9.71 → v15.10.0

### What Changed
1. Added 3 new trading strategies (Bollinger Bands, Stochastic RSI, Fibonacci)
2. Added 2 advanced algorithms (power scoring, adaptive position sizing)
3. Implemented position limits (3 long/3 short per new strategy)
4. Fixed 4 critical bugs
5. Enhanced Telegram integration

## 🗂️ File Structure

```
/home/runner/work/mrp/mrp/
├── ema.py                      # Main trading bot (5,764 lines)
├── EMA_CHANGES.md              # What changed (this answers "where")
├── EMA_CHANGE_LOCATIONS.md     # Exact line numbers
├── IMPROVEMENTS_SUMMARY.md     # Feature summary
├── STRATEGY_UPDATE.md          # Strategy details
├── STRATEGY_LIMITS.md          # Position limits guide
├── README.md                   # Basic readme
├── requirements.txt            # Python dependencies
└── ... (other files)
```

## 🔧 Git Commands

### View all changes to ema.py
```bash
git log --follow --oneline ema.py
```

### View specific commit
```bash
git show 39b8a1a -- ema.py    # New strategies
git show 978b62c -- ema.py    # Position limits
git show a2898c2 -- ema.py    # Power scoring
git show 0fa2da1 -- ema.py    # Bug fixes
```

### View diff statistics
```bash
git diff --stat 39b8a1a^..HEAD -- ema.py
```

### Search for specific changes
```bash
git log -p -S "bollinger_bands" -- ema.py
```

## 📖 Reading Order

For someone new to the changes:
1. Start with **IMPROVEMENTS_SUMMARY.md** for high-level overview
2. Read **EMA_CHANGES.md** for detailed chronological changes
3. Refer to **EMA_CHANGE_LOCATIONS.md** for specific line numbers
4. Check **STRATEGY_UPDATE.md** for strategy implementation details
5. See **STRATEGY_LIMITS.md** for position limit configuration

For developers:
1. **EMA_CHANGE_LOCATIONS.md** - Exact locations
2. **EMA_CHANGES.md** - Commit details
3. Use git commands to view actual diffs

## 📞 Support

For questions about:
- **Changes:** See EMA_CHANGES.md
- **Line numbers:** See EMA_CHANGE_LOCATIONS.md
- **Strategies:** See STRATEGY_UPDATE.md
- **Limits:** See STRATEGY_LIMITS.md
- **Features:** See IMPROVEMENTS_SUMMARY.md

---

Last Updated: February 5, 2026
