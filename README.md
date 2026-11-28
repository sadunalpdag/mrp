# mrp
pyton ile mrp yazımı

## Profit Target Configuration

### Where to Change Profit Target

There are **two ways** to change the profit target:

#### Method 1: Edit the Configuration File (Requires Bot Restart)
- **File**: `ema.py`
- **Line Number**: **Line 3396**
- **Current Default Value**: `20.0` USD

```python
"PROFIT_TARGET_USD":20.0,
```

To change it:
1. Open `ema.py`
2. Go to line 3396
3. Change the value from `20.0` to your desired profit target (e.g., `50.0`, `100.0`, etc.)
4. Save the file
5. Restart the bot for changes to take effect

**Note**: This sets the initial default value when the bot starts. Once running, the value is stored in `data/params.json`.

#### Method 2: Change at Runtime via Telegram (No Restart Required)
You can change the profit target while the bot is running using the Telegram command:

```
/settarget <amount>
```

Example:
- `/settarget 100` - Sets profit target to $100
- `/settarget 50.5` - Sets profit target to $50.50

This method:
- ✅ Takes effect immediately
- ✅ No bot restart required
- ✅ Persists across bot restarts (saved to `params.json`)

### How Profit Target Works
- When your total profit reaches the target amount, the bot will automatically close all positions
- The bot sends a notification when the profit target is reached
- You can check current progress with `/balance` command
