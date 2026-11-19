"""BTC AI scalp: independent 15-minute paper-only research, Google Sheets output.

Run as a dedicated Render cron job at 2,17,32,47 minutes past each hour (UTC).
Never imports/exposes broker order APIs. Does not send Telegram messages.
Requires COINBEACON_API_KEY, GEMINI_API_KEY, GOOGLE_SERVICE_ACCOUNT_JSON,
and GOOGLE_SHEETS_ID (or GOOGLE_SHEET_ID). Times use Europe/Paris.
"""
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

LOG = logging.getLogger("btc_ai_scalp_sheets")
PARIS = ZoneInfo("Europe/Paris")
FIFTEEN = 900_000
SIGNALS = "BTC_AI_SINYALLER"
TRADES = "BTC_AI_SANAL"
DAILY = "BTC_AI_GUNLUK"
SETTINGS = "BTC_AI_AYARLAR"
ANALYSIS = "BTC_AI_ANALIZ"
SIGNAL_HEADER = ["Kapanis UTC", "Paris saati", "BTC kapanis", "Karar", "Giris referans",
                 "TP", "SL", "Gemini gerekce", "Model", "RSI14", "EMA9", "EMA21",
                 "ATR14", "Durum"]
TRADE_HEADER = ["Sinyal UTC", "Yon", "Sinyal referans", "Giris UTC", "Giris",
                "TP", "SL", "Miktar BTC", "Durum", "Cikis UTC", "Cikis",
                "Net PnL USDT", "Not"]
DAILY_HEADER = ["Gun (Paris)", "Gun baslangic sermayesi", "Gerceklesen net PnL",
                "Acik PnL", "Gunluk hedef USDT", "Gunluk zarar limiti USDT",
                "Durum", "Kapanan islem", "Toplam sermaye (realize)"]
SETTING_DEFAULTS = [
    ["setting", "value", "description"],
    ["starting_capital_usdt", "1000", "Sanal baslangic sermayesi"],
    ["daily_target_pct", "5", "Gun baslangic sermayesinin yuzdesi, GARANTI DEGIL"],
    ["daily_loss_pct", "2", "Gunluk net zarar esigi yuzdesi"],
    ["position_fraction_pct", "20", "Tek pozisyon icin sermaye payi"],
    ["leverage", "1", "Sanal kaldirac; baslangicta 1x"],
    ["fee_roundtrip_pct", "0.10", "Varsayilan toplam giris+cikis komisyon %"],
    ["slippage_roundtrip_pct", "0.04", "Varsayilan toplam slippage %"],
    ["enabled", "TRUE", "FALSE ise yeni sinyal uretilmez"],
]
ANALYSIS_HEADER = ["Paris saati", "Son kapanis UTC", "Kapanan", "TP", "SL",
                   "Net PnL USDT", "TP oranı %", "Acik pozisyon",
                   "Aciklama"]


def numeric(value, fallback):
    try:
        n = float(str(value).replace(",", "."))
        return n if math.isfinite(n) else fallback
    except (ValueError, TypeError):
        return fallback


def connect():
    import gspread
    from google.oauth2.service_account import Credentials
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sheet_id = (os.getenv("GOOGLE_SHEETS_ID") or os.getenv("GOOGLE_SHEET_ID") or "").strip()
    if not raw or not sheet_id:
        raise RuntimeError("SHEETS_CONFIGURATION_MISSING")
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = (Credentials.from_service_account_info(json.loads(raw), scopes=scope)
             if raw.startswith("{") else Credentials.from_service_account_file(raw, scopes=scope))
    return gspread.authorize(creds).open_by_key(sheet_id)


def tab(book, name, header):
    import gspread
    try:
        ws = book.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=name, rows=1000, cols=max(15, len(header)))
    if not ws.row_values(1):
        ws.append_row(header, value_input_option="RAW")
    return ws


def ema(values, length):
    alpha = 2 / (length + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def indicators(bars):
    closes = [float(b[4]) for b in bars]
    if len(closes) < 60:
        raise ValueError("HISTORY_TOO_SHORT")
    diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gain = sum(max(0, d) for d in diffs[:14]) / 14
    loss = sum(max(0, -d) for d in diffs[:14]) / 14
    for d in diffs[14:]:
        gain = (gain * 13 + max(0, d)) / 14
        loss = (loss * 13 + max(0, -d)) / 14
    rsi = 100 - 100 / (1 + gain / loss) if loss else (100 if gain else 50)
    ranges = [max(float(b[2])-float(b[3]), abs(float(b[2])-float(bars[i-1][4])),
                  abs(float(b[3])-float(bars[i-1][4]))) for i, b in enumerate(bars) if i]
    atr = sum(ranges[:14]) / 14
    for tr in ranges[14:]:
        atr = (atr * 13 + tr) / 14
    return {"rsi14": round(rsi, 2), "ema9": round(ema(closes, 9), 2),
            "ema21": round(ema(closes, 21), 2), "atr14": round(atr, 2)}


def gemini_decision(bars, ind):
    from google import genai
    from google.genai import types
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY_MISSING")
    model = os.getenv("BTC_SCALP_GEMINI_MODEL", "gemini-2.5-flash").strip()
    payload = {"symbol": "BTCUSDT", "timeframe": "15m", "indicators": ind,
               "closed_ohlcv": [{"open_ms": int(b[0]), "open": b[1], "high": b[2],
                                 "low": b[3], "close": b[4], "volume": b[5]}
                                for b in bars[-48:]]}
    prompt = (
        "Yalnizca verilen kapanmis OHLCV verisini analiz et. Scalp icin LONG, SHORT "
        "veya BEKLE sec; islem zorunlu degil. Hedef gunluk %5, garanti degil; "
        "hedefi yakalamak icin zorlama sinyal verme. Gelecek fiyatini uydurma. "
        "JSON alanlari: action (LONG/SHORT/BEKLE), entry (sayi), tp (sayi), "
        "sl (sayi), reason (Turkce kisa cumle). BEKLE icin seviyeler 0 olabilir. "
        "Gercek islem veya emir uretme. VERI: " + json.dumps(payload, ensure_ascii=False)
    )
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1, response_mime_type="application/json"))
        decision = json.loads(response.text or "{}")
        return decision, model
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def validated_decision(decision, close, atr):
    action = str(decision.get("action", "BEKLE")).upper()
    if action not in ("LONG", "SHORT", "BEKLE"):
        action = "BEKLE"
    entry, tp, sl = [numeric(decision.get(k), 0) for k in ("entry", "tp", "sl")]
    reason = str(decision.get("reason", ""))[:400]
    # A frozen close reference is the quote at signal creation, not a guaranteed fill.
    if action != "BEKLE":
        valid = (entry > 0 and tp > 0 and sl > 0 and abs(entry / close - 1) <= 0.002
                 and atr > 0)
        valid &= (sl < entry < tp if action == "LONG" else tp < entry < sl)
        reward = abs(tp - entry)
        risk = abs(entry - sl)
        valid &= risk > 0 and reward / risk >= 1.2 and risk / entry <= 0.015
        if not valid:
            reason = "RISK_VALIDATION_FAILED: " + reason
            action = "BEKLE"
    return action, entry if action != "BEKLE" else 0, tp if action != "BEKLE" else 0, sl if action != "BEKLE" else 0, reason


def settings(ws):
    rows = ws.get_all_values()
    if len(rows) <= 1:
        ws.update(range_name="A1:C9", values=SETTING_DEFAULTS, value_input_option="RAW")
        rows = SETTING_DEFAULTS
    cfg = {row[0]: row[1] for row in rows[1:] if len(row) >= 2}
    capital = max(1, numeric(cfg.get("starting_capital_usdt"), 1000))
    target = min(20, max(0.1, numeric(cfg.get("daily_target_pct"), 5)))
    loss = min(20, max(0.1, numeric(cfg.get("daily_loss_pct"), 2)))
    fraction = min(100, max(0.1, numeric(cfg.get("position_fraction_pct"), 20)))
    leverage = min(5, max(1, numeric(cfg.get("leverage"), 1)))
    fee = max(0, numeric(cfg.get("fee_roundtrip_pct"), 0.10))
    slip = max(0, numeric(cfg.get("slippage_roundtrip_pct"), 0.04))
    return capital, target, loss, fraction, leverage, fee + slip, cfg.get("enabled", "TRUE").upper() == "TRUE"


def run():
    now_ms = int(time.time() * 1000)
    end_ms = now_ms // FIFTEEN * FIFTEEN
    if now_ms - end_ms < 60_000:
        raise RuntimeError("WAIT_60_SECONDS_AFTER_CLOSE")
    book = connect()
    sw = tab(book, SIGNALS, SIGNAL_HEADER)
    tw = tab(book, TRADES, TRADE_HEADER)
    dw = tab(book, DAILY, DAILY_HEADER)
    cw = tab(book, SETTINGS, SETTING_DEFAULTS[0])
    aw = tab(book, ANALYSIS, ANALYSIS_HEADER)
    initial, target_pct, loss_pct, fraction, leverage, cost_pct, enabled = settings(cw)
    sig = sw.get_all_records()
    trades = tw.get_all_values()[1:]
    days = dw.get_all_values()[1:]
    # Data source is the same quota-aware CoinBeacon client as the BTC observer.
    from btc_coinbeacon_ohlcv import fetch_closed_15m
    bars = fetch_closed_15m(end_ms, 128)
    last_close_ms = end_ms
    stamp = datetime.fromtimestamp(last_close_ms / 1000, timezone.utc).isoformat()
    paris_now = datetime.fromtimestamp(last_close_ms / 1000, PARIS).isoformat()
    today = paris_now[:10]
    # Stop duplicate signals after cron retries; all state lives in Sheets.
    signal_exists = any(str(r.get("Kapanis UTC", "")) == stamp for r in sig)

    # Only candles after a frozen signal may create fills. A pending signal
    # fills at the open of the NEXT fully closed 15m bar (never at old close).
    updates = []
    for idx, row in enumerate(trades, start=2):
        r = (row + [""] * len(TRADE_HEADER))[:len(TRADE_HEADER)]
        status = r[8]
        if status not in ("PENDING", "OPEN"):
            continue
        signal_ms = int(datetime.fromisoformat(r[0]).timestamp() * 1000)
        entry_ms = (int(datetime.fromisoformat(r[3]).timestamp() * 1000)
                    if r[3] else 0)
        side, tp, sl = r[1], float(r[5]), float(r[6])
        for bar in bars:
            open_ms = int(bar[0])
            if open_ms < signal_ms or (status == "OPEN" and open_ms < entry_ms):
                continue
            if status == "PENDING":
                # The candle opening at signal timestamp only becomes
                # observable after its 15-minute close.
                entry_ms, price = open_ms, float(bar[1])
                qty = initial * fraction / 100 * leverage / price
                r[3], r[4], r[7], r[8] = (
                    datetime.fromtimestamp(entry_ms/1000, timezone.utc).isoformat(),
                    str(price), str(qty), "OPEN")
                status = "OPEN"
                # Do not evaluate TP/SL on the same entry candle:
                # fill and intrabar ordering cannot be proven.
                continue
            hi, lo = float(bar[2]), float(bar[3])
            hit_tp = (hi >= tp if side == "LONG" else lo <= tp)
            hit_sl = (lo <= sl if side == "LONG" else hi >= sl)
            if not hit_tp and not hit_sl:
                continue
            # If both touched in a candle, assume SL first.
            exit_price = sl if hit_sl else tp
            qty = float(r[7])
            gross = (exit_price - float(r[4])) * qty * (1 if side == "LONG" else -1)
            fees = float(r[4]) * qty * cost_pct / 100
            r[8], r[9], r[10], r[11], r[12] = (
                "SL" if hit_sl else "TP",
                datetime.fromtimestamp((open_ms+FIFTEEN)/1000, timezone.utc).isoformat(),
                str(exit_price), str(round(gross-fees, 4)),
                "Slippage+fee assumed; same-bar tie->SL")
            status = r[8]
            break
        updates.append((idx, r))
    for row_idx, record in updates:
        tw.update(range_name=f"A{row_idx}:M{row_idx}",
                  values=[record], value_input_option="RAW")
    trades = tw.get_all_values()[1:]

    def pnl_for_day(day):
        return sum(numeric(r[11], 0) for r in trades
                   if len(r) > 11 and r[9] and
                   datetime.fromisoformat(r[9]).astimezone(PARIS).date().isoformat() == day)
    # Capital remains anchored to the day-opening realized equity, not
    # current unrealized price or a changing intraday high watermark.
    previous_pnl = sum(numeric(r[11], 0) for r in trades
                       if len(r) > 11 and r[9] and
                       datetime.fromisoformat(r[9]).astimezone(PARIS).date().isoformat() < today)
    day_start = initial + previous_pnl
    realized = pnl_for_day(today)
    open_pnl = 0.0
    live = []
    for r in trades:
        if len(r) > 8 and r[8] in ("PENDING", "OPEN"):
            live.append(r)
            if r[8] == "OPEN":
                mark = float(bars[-1][4])
                qty = numeric(r[7], 0)
                open_pnl += (mark-numeric(r[4], mark))*qty*(1 if r[1]=="LONG" else -1)
                open_pnl -= numeric(r[4], mark)*qty*cost_pct/100
    target = max(0, day_start) * target_pct / 100
    loss_limit = max(0, day_start) * loss_pct / 100
    risk_equity = realized + open_pnl
    halt = risk_equity >= target or risk_equity <= -loss_limit or day_start <= 0
    # A pending/open trade is not duplicated; no live orders or Telegram.
    if not signal_exists:
        close = float(bars[-1][4])
        ind = indicators(bars)
        if enabled and not halt and not live:
            try:
                decision, model = gemini_decision(bars, ind)
                action, entry, tp, sl, reason = validated_decision(decision, close, ind["atr14"])
            except Exception as exc:
                LOG.exception("Gemini failed; no artificial signal")
                action, entry, tp, sl, model = "BEKLE", 0, 0, 0, "ERROR"
                reason = "GEMINI_UNAVAILABLE: " + type(exc).__name__
        else:
            action, entry, tp, sl, model = "BEKLE", 0, 0, 0, "SKIPPED"
            reason = ("DISABLED" if not enabled else
                      "DAILY_LIMIT_REACHED" if halt else "POSITION_ALREADY_ACTIVE")
        sw.append_row([stamp, paris_now, close, action, entry, tp, sl, reason,
                       model, ind["rsi14"], ind["ema9"], ind["ema21"], ind["atr14"],
                       "PAPER_ONLY"], value_input_option="RAW")
        if action in ("LONG", "SHORT"):
            tw.append_row([stamp, action, entry, "", "", tp, sl, "",
                           "PENDING", "", "", "", "NEXT_CLOSED_CANDLE_FILL"],
                          value_input_option="RAW")
    closed_today = [r for r in trades if len(r) > 11 and r[9] and
                    datetime.fromisoformat(r[9]).astimezone(PARIS).date().isoformat() == today]
    state = ("HEDEF" if risk_equity >= target else "ZARAR_LIMITI"
             if risk_equity <= -loss_limit else "KAPALI" if not enabled else "TAKIP")
    daily_row = [today, round(day_start, 4), round(realized, 4), round(open_pnl, 4),
                 round(target, 4), round(loss_limit, 4), state, len(closed_today),
                 round(day_start+realized, 4)]
    existing = next((i for i, row in enumerate(days, 2) if row and row[0] == today), None)
    if existing:
        dw.update(range_name=f"A{existing}:I{existing}",
                  values=[daily_row], value_input_option="RAW")
    else:
        dw.append_row(daily_row, value_input_option="RAW")
    tp_count = sum(1 for r in closed_today if r[8] == "TP")
    sl_count = sum(1 for r in closed_today if r[8] == "SL")
    analysis_row = [paris_now, stamp, len(closed_today), tp_count, sl_count,
                    round(realized,4), round(tp_count*100/len(closed_today),2)
                    if closed_today else "", len(live),
                    "Sanal; fill next candle open, costs estimated, target not guaranteed"]
    aw.append_row(analysis_row, value_input_option="RAW")
    LOG.info("BTC AI SCALP sheet=OK date=%s state=%s pnl=%.4f active=%s",
             today, state, realized, len(live))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
