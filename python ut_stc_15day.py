# ============================================================
# 📘 UT BOT + STC Backtest (15 Günlük, Binance 5m)
# ============================================================

import os, json, time, requests, hmac, hashlib, threading, math
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime

# ------------------------------------------------------------
# Binance verisi çekme (15 gün)
# ------------------------------------------------------------
def fetch_binance(symbol="BTC/USDT", timeframe="5m", days=15):
    exchange = ccxt.binance()
    limit = 1500
    all_data = []
    since = exchange.parse8601((datetime.utcnow() - pd.Timedelta(days=days)).isoformat())

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        df_chunk = pd.DataFrame(ohlcv, columns=["Timestamp","Open","High","Low","Close","Volume"])
        all_data.append(df_chunk)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break

    df = pd.concat(all_data)
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Date", inplace=True)
    return df

# ------------------------------------------------------------
# UT Bot
# ------------------------------------------------------------
def ut_bot(df, key_value=2, atr_period=1):
    df = df.copy()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], atr_period).average_true_range()
    df["upperband"] = df["Close"] - (atr * key_value)
    df["lowerband"] = df["Close"] + (atr * key_value)

    buy, sell = [], []
    trend = 1
    for i in range(len(df)):
        if df["Close"].iloc[i] > df["lowerband"].iloc[i]:
            signal = 1
        elif df["Close"].iloc[i] < df["upperband"].iloc[i]:
            signal = -1
        else:
            signal = trend
        buy.append(1 if signal == 1 and trend != 1 else 0)
        sell.append(1 if signal == -1 and trend != -1 else 0)
        trend = signal
    df["UT_Buy"] = buy
    df["UT_Sell"] = sell
    return df

# ------------------------------------------------------------
# STC (Momentum filtre)
# ------------------------------------------------------------
def stc(df, length=80, fast=227):
    macd = ta.trend.MACD(df["Close"], window_slow=length, window_fast=int(length/2))
    stc_line = macd.macd_diff()
    df["STC"] = ta.trend.ema_indicator(stc_line, window=int(fast/50))
    return df

# ------------------------------------------------------------
# Backtest
# ------------------------------------------------------------
def backtest_ut_stc(symbol="BTC/USDT", timeframe="5m"):
    df = fetch_binance(symbol, timeframe, days=15)
    df = ut_bot(df, key_value=2, atr_period=1)
    df = ut_bot(df, key_value=2, atr_period=300)
    df = stc(df)

    df["buy_signal"] = df["UT_Buy"] & (df["STC"] > 0)
    df["sell_signal"] = df["UT_Sell"] & (df["STC"] < 0)

    position = None
    trades = []

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        if position is None:
            if df["buy_signal"].iloc[i]:
                entry = price
                sl = df["Low"].iloc[i-1]
                tp = entry + 2 * (entry - sl)
                position = {"side": "long", "entry": entry, "sl": sl, "tp": tp, "entry_time": df.index[i]}
            elif df["sell_signal"].iloc[i]:
                entry = price
                sl = df["High"].iloc[i-1]
                tp = entry - 2 * (sl - entry)
                position = {"side": "short", "entry": entry, "sl": sl, "tp": tp, "entry_time": df.index[i]}
        else:
            if position["side"] == "long":
                if price <= position["sl"]:
                    trades.append({**position, "exit": price, "exit_time": df.index[i], "result": -1})
                    position = None
                elif price >= position["tp"]:
                    trades.append({**position, "exit": price, "exit_time": df.index[i], "result": 2})
                    position = None
            elif position["side"] == "short":
                if price >= position["sl"]:
                    trades.append({**position, "exit": price, "exit_time": df.index[i], "result": -1})
                    position = None
                elif price <= position["tp"]:
                    trades.append({**position, "exit": price, "exit_time": df.index[i], "result": 2})
                    position = None

    results = pd.DataFrame(trades)
    if results.empty:
        print("❌ No trades found.")
        return

    results["PnL_%"] = np.where(
        results["side"]=="long",
        (results["exit"]-results["entry"])/results["entry"]*100,
        (results["entry"]-results["exit"])/results["entry"]*100
    )

    win_rate = (results["PnL_%"] > 0).mean()*100
    avg_pnl = results["PnL_%"].mean()
    total_trades = len(results)
    gross_pnl = results["PnL_%"].sum()

    print(f"\n📊 {symbol} — UT BOT + STC Backtest (5m, 15 Gün)")
    print(f"✅ İşlem Sayısı: {total_trades}")
    print(f"🏆 Kazanma Oranı: {win_rate:.1f}%")
    print(f"💰 Ortalama Kâr/Zarar: {avg_pnl:.2f}%")
    print(f"📈 Toplam Getiri: {gross_pnl:.2f}%\n")

    return results

# ------------------------------------------------------------
# Çalıştır
# ------------------------------------------------------------
if __name__ == "__main__":
    backtest_ut_stc("BTC/USDT", "5m")
