"""Hourly BTC direction observer: no orders, no account access, no trading locks.

Runs in its own daemon thread. Forecasts are frozen at creation and evaluated only
against later *closed* 1h candles. Sheets and R2 writes are best-effort.
"""
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

LOG = logging.getLogger("btc_hourly_direction")
PARIS = ZoneInfo("Europe/Paris")
HORIZONS = (1, 3, 6, 12, 24)
HEADER = ["Saat (Paris)", "UTC saat", "BTC fiyat", "Tahmin", "Puan (-100..100)",
          "RSI 1H", "ADX 1H", "EMA20 1H", "EMA50 1H", "MACD hist 1H",
          "RVOL 1H", "4H yon", "Funding", "OI", "Spot CVD",
          "1H %", "1H dogru", "3H %", "3H dogru",
          "6H %", "6H dogru", "12H %", "12H dogru",
          "24H %", "24H dogru", "Degerlendirme", "Yatay rejim", "Destek", "Direnc", "Sanal durum", "Sanal net %"]
_STARTED = False
_LOCK = threading.Lock()


def _data_path():
    base = Path(os.getenv("DATA_DIR") or (Path(__file__).parent / "data"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "btc_hourly_direction.json"


def _ema(values, period):
    alpha = 2.0 / (period + 1)
    out = float(values[0])
    result = []
    for value in values:
        out = float(value) * alpha + out * (1 - alpha)
        result.append(out)
    return result


def _rsi(values, period=14):
    if len(values) <= period:
        return None
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    avg_gain = sum(max(x, 0) for x in diffs[:period]) / period
    avg_loss = sum(max(-x, 0) for x in diffs[:period]) / period
    for delta in diffs[period:]:
        avg_gain = (avg_gain * (period - 1) + max(delta, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0)) / period
    if not avg_loss:
        return 100.0 if avg_gain else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1 + rs)


def _adx(candles, n=14):
    if len(candles) < 2 * n + 2:
        return None, None, None
    tr, pdm, mdm = [], [], []
    for previous, current in zip(candles, candles[1:]):
        up = current[2] - previous[2]
        down = previous[3] - current[3]
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
        tr.append(max(current[2] - current[3],
                      abs(current[2] - previous[4]),
                      abs(current[3] - previous[4])))
    def smooth(series):
        value = sum(series[:n])
        result = [value]
        for item in series[n:]:
            value = value - value / n + item
            result.append(value)
        return result
    ats, ps, ms = smooth(tr), smooth(pdm), smooth(mdm)
    dx = []
    last_p = last_m = 0.0
    for a, p, m in zip(ats, ps, ms):
        last_p, last_m = 100 * p / a if a else 0, 100 * m / a if a else 0
        denom = last_p + last_m
        dx.append(100 * abs(last_p - last_m) / denom if denom else 0.0)
    if len(dx) < n:
        return None, last_p, last_m
    value = sum(dx[:n]) / n
    for item in dx[n:]:
        value = (value * (n - 1) + item) / n
    return value, last_p, last_m


def _fetch_closed_bars(interval, limit, last_open_ms, priority_checker):
    # Public MARKET ONLY. Explicitly yield to live Crash BUY/TP workload.
    if priority_checker():
        raise RuntimeError("LIVE_CRASH_PRIORITY")
    from crash_rest_budget import BUDGET, estimated_weight
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    path = "/fapi/v1/klines"
    BUDGET.reserve("research", estimated_weight(path, params))
    response = None
    try:
        response = requests.get("https://fapi.binance.com" + path,
                                params=params, timeout=12)
        BUDGET.observe(response.headers, response.status_code,
                       response.headers.get("Retry-After", 0))
        response.raise_for_status()
        data = response.json()
    except Exception:
        if response is None:
            LOG.warning("[BTC DIRECTION] Binance data request failed (no response)")
        raise
    # Keep only completely closed candles; never use an in-progress future candle.
    return [[int(b[0]), float(b[1]), float(b[2]), float(b[3]),
             float(b[4]), float(b[5])] for b in data
            if int(b[0]) + (3600000 if interval == "1h" else 14400000) <= last_open_ms]


def _predict(bars, bars4, hour_ms):
    closes = [b[4] for b in bars]
    volumes = [b[5] for b in bars]
    if len(bars) < 65 or len(bars4) < 55 or bars[-1][0] != hour_ms - 3600000:
        raise RuntimeError("BTC_CLOSED_HISTORY_NOT_READY")
    ema20 = _ema(closes, 20)[-1]
    ema50 = _ema(closes, 50)[-1]
    rsi = _rsi(closes)
    macd_line = [a - b for a, b in zip(_ema(closes, 12), _ema(closes, 26))]
    hist = macd_line[-1] - _ema(macd_line, 9)[-1]
    adx, plus_di, minus_di = _adx(bars)
    rvol = volumes[-1] / (sum(volumes[-21:-1]) / 20) if sum(volumes[-21:-1]) else 0.0
    close = closes[-1]
    close4 = bars4[-1][4]
    ema4 = _ema([b[4] for b in bars4], 20)[-1]
    # Positive/negative scores are a research heuristic, not calibrated probabilities.
    components = {
        "ema_1h": 2 if close > ema20 > ema50 else (-2 if close < ema20 < ema50 else 0),
        "ema_4h": 2 if close4 > ema4 else (-2 if close4 < ema4 else 0),
        "macd": 1 if hist > 0 else -1,
        "rsi": 1 if rsi >= 55 else (-1 if rsi <= 45 else 0),
        "dmi_adx": (2 if plus_di > minus_di else -2) if adx is not None and adx >= 20 else 0,
        "rvol": (1 if closes[-1] > closes[-2] else -1) if rvol >= 1.5 else 0,
    }
    score = 10 * sum(components.values())
    label = "YUKSELIS" if score >= 30 else ("DUSUS" if score <= -30 else "YATAY")
    return {
        "timestamp_utc": datetime.fromtimestamp(hour_ms / 1000, timezone.utc).isoformat(),
        "timestamp_paris": datetime.fromtimestamp(hour_ms / 1000, PARIS).isoformat(),
        "hour_ms": hour_ms,
        "entry_price": close, "direction": label, "score": score,
        "indicators": {"rsi_1h": round(rsi, 3), "adx_1h": round(adx, 3) if adx is not None else None,
                       "plus_di": round(plus_di, 3) if plus_di is not None else None,
                       "minus_di": round(minus_di, 3) if minus_di is not None else None,
                       "ema20_1h": round(ema20, 3), "ema50_1h": round(ema50, 3),
                       "macd_hist_1h": round(hist, 5), "rvol_1h": round(rvol, 3),
                       "ema20_4h": round(ema4, 3), "trend_4h": "UP" if close4 > ema4 else "DOWN",
                       "funding": None, "open_interest": None, "spot_cvd": None,
                       "components": components},
        "results": {}, "model": "btc_direction_v1", "telegram_sent": False,
    }


def _evaluate(records, bars):
    candles = {b[0]: b for b in bars}
    changed = set()
    for key, record in records.items():
        for h in HORIZONS:
            hkey = str(h)
            if hkey in record.get("results", {}):
                continue
            # Observation at hour T uses closed candle [T-1h,T).
            # H-hour outcome = last closed candle ending T+H.
            future = candles.get(record["hour_ms"] + (h - 1) * 3600000)
            if future is None:
                continue
            ret = (future[4] / record["entry_price"] - 1) * 100
            direction = record["direction"]
            # YATAY counts as correct only when absolute movement < 0.2%.
            correct = (ret > 0 if direction == "YUKSELIS"
                       else ret < 0 if direction == "DUSUS"
                       else abs(ret) < 0.2)
            record.setdefault("results", {})[hkey] = {
                "return_pct": round(ret, 4), "correct": bool(correct),
                "settled_utc": datetime.fromtimestamp((record["hour_ms"] + h * 3600000) / 1000,
                                                        timezone.utc).isoformat()}
            changed.add(datetime.fromisoformat(record["timestamp_paris"]).date().isoformat())
    return changed


def _read_local():
    try:
        data = json.loads(_data_path().read_text(encoding="utf-8"))
        return {r["timestamp_utc"]: r for r in data if isinstance(r, dict)}
    except (FileNotFoundError, ValueError, OSError, TypeError, KeyError):
        return {}


def _write_local(records):
    path = _data_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(records.values(), key=lambda r: r["hour_ms"]),
                              ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _r2_client():
    from crash_report_storage import R2_ENV_NAMES, r2_configured
    if not r2_configured():
        return None, None
    import boto3
    values = [os.environ[n].strip() for n in R2_ENV_NAMES]
    account, bucket, access, secret = values
    return boto3.client("s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
                        aws_access_key_id=access, aws_secret_access_key=secret,
                        region_name="auto"), bucket


def _r2_key(day):
    return "btc-direction/" + day.replace("-", "/") + "/btc_direction_" + day + ".json"


def _restore_r2(records):
    try:
        client, bucket = _r2_client()
        if not client:
            return
        for days_ago in (2, 1, 0):
            day = (datetime.now(PARIS).date() - timedelta(days=days_ago)).isoformat()
            try:
                obj = client.get_object(Bucket=bucket, Key=_r2_key(day))
                items = json.loads(obj["Body"].read())
                for rec in items:
                    if isinstance(rec, dict) and "timestamp_utc" in rec:
                        records.setdefault(rec["timestamp_utc"], rec)
            except Exception as exc:
                # Missing object is normal on first deployment.
                if getattr(exc, "response", {}).get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                    LOG.warning("[BTC DIRECTION R2 RESTORE] %s", type(exc).__name__)
    except Exception as exc:
        LOG.warning("[BTC DIRECTION R2 INIT] %s", type(exc).__name__)


def _archive_r2(records, days):
    client, bucket = _r2_client()
    if not client:
        return
    for day in days:
        rows = [r for r in records.values() if r["timestamp_paris"][:10] == day]
        rows.sort(key=lambda r: r["hour_ms"])
        client.put_object(Bucket=bucket, Key=_r2_key(day),
                          Body=json.dumps(rows, ensure_ascii=False).encode("utf-8"),
                          ContentType="application/json")


def _sync_sheet(records):
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sheet_id = (os.getenv("GOOGLE_SHEETS_ID") or os.getenv("GOOGLE_SHEET_ID") or "").strip()
    if not raw or not sheet_id:
        return
    import gspread
    from google.oauth2.service_account import Credentials
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = (Credentials.from_service_account_info(json.loads(raw), scopes=scope)
                   if raw.startswith("{") else Credentials.from_service_account_file(raw, scopes=scope))
    doc = gspread.authorize(credentials).open_by_key(sheet_id)
    try:
        ws = doc.worksheet("BTC_YON_ANALIZ")
    except gspread.WorksheetNotFound:
        ws = doc.add_worksheet(title="BTC_YON_ANALIZ", rows=3000, cols=len(HEADER))
    # Persisted R2 is source of truth; write sorted records as a single range.
    # max 35 days prevents growing Sheets writes unbounded.
    cutoff_ms = (time.time() - 35 * 86400) * 1000
    rows = [r for r in records.values() if r["hour_ms"] >= cutoff_ms]
    rows.sort(key=lambda r: r["hour_ms"])
    values = [HEADER]
    for r in rows:
        i = r["indicators"]
        row = [r["timestamp_paris"], r["timestamp_utc"], r["entry_price"],
               r["direction"], r["score"], i["rsi_1h"], i["adx_1h"],
               i["ema20_1h"], i["ema50_1h"], i["macd_hist_1h"],
               i["rvol_1h"], i["trend_4h"],
               i.get("funding") or "", i.get("open_interest") or "",
               i.get("spot_cvd") or ""]
        for h in HORIZONS:
            outcome = r.get("results", {}).get(str(h))
            row += ([outcome["return_pct"], "DOGRU" if outcome["correct"] else "YANLIS"]
                    if outcome else ["", "BEKLIYOR"])
        row.append("TAMAMLANDI" if len(r.get("results", {})) == len(HORIZONS) else "IZLENIYOR")
        observation = r.get("range_observation") or {}
        plan = r.get("range_paper") or {}
        row.extend([observation.get("regime", ""),
                    observation.get("support", ""), observation.get("resistance", ""),
                    (plan.get("status", "") + (" / " + plan.get("result", "") if plan.get("result") else "")),
                    plan.get("net_pct", "")])
        values.append(row)
    if len(values) > ws.row_count:
        ws.add_rows(len(values) - ws.row_count + 100)
    if ws.col_count < len(HEADER):
        ws.add_cols(len(HEADER) - ws.col_count)
    ws.update(range_name=f"A1:AE{len(values)}", values=values, value_input_option="RAW")


def _telegram(record):
    token = os.getenv("BOT_TOKEN", "").strip()
    chat_id = os.getenv("CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    i = record["indicators"]
    text = (f"BTC SAATLIK YON | {record['timestamp_paris']}\n"
            f"BTC: {record['entry_price']:,.2f} USDT\n"
            f"Tahmin: {record['direction']} | Yon skoru: {record['score']:+d}/100\n"
            f"RSI 1H: {i['rsi_1h']} | ADX: {i['adx_1h']} | RVOL: {i['rvol_1h']}\n"
            f"4H: {i['trend_4h']} | MACD hist: {i['macd_hist_1h']}\n"
            "1/3/6/12/24 saatlik sonuc Sheets ve R2'de izleniyor.\n"
            "SADECE BILGI - Emir veya strateji degisikligi YOK.")
    plan = record.get("range_paper")
    if plan:
        observation = record.get("range_observation") or {}
        text += (f"\nSANAL AL ADAYI (emir verilmez) | DESTEK {observation.get('support')}"
                 f" | DIRENC {observation.get('resistance')}\n"
                 f"Sanal limit alis: {plan['entry']:,.2f} | Sanal hedef satis: {plan['target']:,.2f}"
                 f" | Iptal seviyesi: {plan['invalidation']:,.2f} USDT\n"
                 "Sinyal performansi Sheets / R2 uzerinden izlenecek.")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": text}, timeout=12)
    r.raise_for_status()
    return bool(r.json().get("ok"))


def _run(priority_checker):
    records = _read_local()
    _restore_r2(records)
    _write_local(records)
    while True:
        try:
            now = datetime.now(timezone.utc)
            # At hh:05 allow exchange to finalize the hh:00 close.
            if now.minute < 5 or priority_checker():
                time.sleep(35)
                continue
            hour_ms = int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
            key = datetime.fromtimestamp(hour_ms / 1000, timezone.utc).isoformat()
            # 1h history also settles older pending predictions.
            bars = _fetch_closed_bars("1h", 100, hour_ms, priority_checker)
            changed_days = _evaluate(records, bars)
            if key not in records:
                bars4 = _fetch_closed_bars("4h", 100, hour_ms, priority_checker)
                records[key] = _predict(bars, bars4, hour_ms)
                changed_days.add(records[key]["timestamp_paris"][:10])
            # Fully isolated paper observer. Failures cannot prevent the base
            # direction signal, real trading, or its ordinary R2/Sheets updates.
            if os.getenv("BTC_RANGE_PAPER_ENABLED", "0") == "1":
                try:
                    from btc_range_paper import step as btc_range_paper_step
                    changed_days.update(btc_range_paper_step(
                        records, hour_ms, priority_checker, records[key]))
                except Exception as exc:
                    LOG.warning("[BTC RANGE PAPER] %s: %s",
                                type(exc).__name__, str(exc)[:160])
            # Always save locally before network writes. Failure must not affect trading.
            if changed_days:
                _write_local(records)
            if not records[key].get("telegram_sent"):
                try:
                    if _telegram(records[key]):
                        records[key]["telegram_sent"] = True
                        _write_local(records)
                except Exception as exc:
                    LOG.warning("[BTC DIRECTION TG] %s", type(exc).__name__)
            if changed_days:
                try:
                    _archive_r2(records, changed_days)
                except Exception as exc:
                    LOG.warning("[BTC DIRECTION R2] %s", type(exc).__name__)
                try:
                    _sync_sheet(records)
                except Exception as exc:
                    LOG.warning("[BTC DIRECTION SHEETS] %s", type(exc).__name__)
            # No second Binance request in the same wall-clock hour.
            time.sleep(3600 - (time.time() % 3600))
        except Exception as exc:
            LOG.warning("[BTC DIRECTION] %s: %s", type(exc).__name__, str(exc)[:200])
            time.sleep(75)


def start(priority_checker=None):
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        # Default: disabled until operator opts in. No effect on existing bot.
        if os.getenv("BTC_DIRECTION_OBSERVER_ENABLED", "0") != "1":
            return
        checker = priority_checker or (lambda: False)
        threading.Thread(target=_run, args=(checker,), daemon=True,
                         name="btc-hourly-direction").start()
        _STARTED = True
