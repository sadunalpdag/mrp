"""CoinBeacon 4h bullish breakout paper tracking; no real orders or extra Binance REST.

Read-only CoinBeacon Smart Setups (one request per closed four-hour bar),
existing Binance Futures ticker WebSocket for hypothetical 1% TP, existing
cached 15m candles for 4h RSI14/ATR14/volume, daily private R2 JSON + ZIP.
"""
import datetime as dt
import json
import math
import os
import threading
import time
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

FOUR_H_MS = 4 * 3600000
FIFTEEN_M_MS = 900000
MAX_AGE_SECONDS = 72 * 3600
PAPER_TP_PCT = 1.0
MODEL = "coinbeacon_4h_long_break_paper_v1"
TAB = "4H Break Long"
HEADERS = ["Coin", "4s kırılım kapanışı UTC", "Sanal giriş Paris", "Kırılım",
           "SQS", "Sanal giriş", "TP %1", "Son fiyat", "Durum",
           "RSI14 (4s)", "ATR14 % (4s)", "Hacim 4s / ort.", "Beacon RVOL",
           "En kötü %", "En iyi %", "24s %", "48s %", "72s %",
           "Son gözlem Paris", "Veri kalitesi", "Kaynak"]
PARIS = ZoneInfo("Europe/Paris")
API = "https://api.coinbeacon.io/public/setups"


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def time_text(value, zone=PARIS):
    x = finite(value)
    return dt.datetime.fromtimestamp(x, zone).strftime("%d.%m.%Y %H:%M") if x is not None else "—"


def _load(path, default):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else default
    except (ValueError, OSError):
        return default


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def last_four_hour_close(now):
    return int(now * 1000) // FOUR_H_MS * FOUR_H_MS - 1


def _four_hour_bars(directory, symbol, ws_rows, now):
    """Aggregate ONLY 16 real contiguous closed 15m slots per UTC 4h candle."""
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isalnum() or not symbol.endswith("USDT"):
        return []
    path = Path(directory) / "crash_research" / "history_cache" / (symbol + ".json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = []
    end = int(now * 1000) // FIFTEEN_M_MS * FIFTEEN_M_MS
    start = end - 960 * FIFTEEN_M_MS
    slots = {}
    for r in list(raw if isinstance(raw, list) else []) + list(ws_rows if isinstance(ws_rows, list) else []):
        try:
            t = int(r[0])
            ct = int(r[6])
            op, high, low, close, volume = map(float, (r[1],r[2],r[3],r[4],r[5]))
            if (t % FIFTEEN_M_MS or not start <= t < end or
                    ct != t + FIFTEEN_M_MS - 1 or ct >= now * 1000 or
                    min(op, low, close, volume) < 0 or op <= 0 or
                    high < max(op, close, low) or low > min(op, close) or
                    not all(math.isfinite(x) for x in (op, high, low, close, volume))):
                continue
            slots[t] = (op, high, low, close, volume)
        except (ValueError, TypeError, IndexError, OverflowError):
            continue
    bars = []
    current_four_start = int(now * 1000) // FOUR_H_MS * FOUR_H_MS
    for first in sorted({t // FOUR_H_MS * FOUR_H_MS for t in slots}):
        if first + FOUR_H_MS > current_four_start:
            continue
        times = range(first, first + FOUR_H_MS, FIFTEEN_M_MS)
        if not all(t in slots for t in times):
            continue
        group = [slots[t] for t in times]
        bars.append([first, group[0][0], max(x[1] for x in group),
                     min(x[2] for x in group), group[-1][3],
                     sum(x[4] for x in group), first + FOUR_H_MS - 1])
    return bars


def indicators(bars):
    """Wilder RSI14; simple ATR14 true ranges, 4h volume ratio vs prior 20."""
    if len(bars) < 22:
        return {"rsi14_4h": None, "atr14_pct_4h": None, "volume_ratio_4h": None,
                "bars_4h": len(bars)}
    closes = [b[4] for b in bars]
    differences = [closes[i] - closes[i-1] for i in range(1,len(closes))]
    gains = [max(x, 0) for x in differences]
    losses = [max(-x, 0) for x in differences]
    avg_g = sum(gains[:14]) / 14
    avg_l = sum(losses[:14]) / 14
    for g, l in zip(gains[14:], losses[14:]):
        avg_g, avg_l = (13 * avg_g + g) / 14, (13 * avg_l + l) / 14
    rsi = 100 if avg_l == 0 and avg_g > 0 else 50 if avg_l == avg_g == 0 else 100-100/(1+avg_g/avg_l)
    tr = [max(bars[i][2] - bars[i][3],
              abs(bars[i][2] - bars[i-1][4]), abs(bars[i][3] - bars[i-1][4]))
          for i in range(1,len(bars))]
    atr = sum(tr[-14:]) / 14 / closes[-1] * 100 if closes[-1] else None
    baseline = sum(b[5] for b in bars[-21:-1]) / 20
    volume_ratio = bars[-1][5] / baseline if baseline > 0 else None
    return {"rsi14_4h": round(rsi, 3), "atr14_pct_4h": round(atr, 3) if atr is not None else None,
            "volume_ratio_4h": round(volume_ratio, 3) if volume_ratio is not None else None,
            "bars_4h": len(bars)}


def parse_setups(payload, close_ms, prices, now, directory, candle_get, market):
    """Strictly new bullish 4h breakout events on actual Binance-USDT perpetuals."""
    if not isinstance(payload, dict) or not isinstance(payload.get("setups"), list):
        raise ValueError("COINBEACON_INVALID_RESPONSE")
    result = []
    for item in payload["setups"]:
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol")
        arch = item.get("archetype") or {}
        if (not isinstance(sym, str) or not sym.endswith("USDT") or not sym.isascii()
                or not sym.isalnum() or sym not in market or not isinstance(arch, dict)
                or item.get("tf") != "4h" or item.get("bias") != "bullish"
                or arch.get("direction") != "long"
                or arch.get("eventKind") not in ("trendline_break", "pattern_breakout")
                or finite(arch.get("barsSinceTrigger")) != 0):
            continue
        event_at = finite(item.get("computedAt"))
        # Do not invent a live breakout from an old board snapshot.
        if event_at is None or abs(event_at - close_ms) > 1000:
            continue
        entry = finite(prices.get(sym))
        if entry is None or entry <= 0:
            continue
        bars = _four_hour_bars(directory, sym, candle_get(sym), now)
        metric = indicators(bars)
        participation = item.get("participation") or {}
        rvol = finite(participation.get("rvol")) if isinstance(participation, dict) else None
        rid = f"{MODEL}:{sym}:{int(event_at)}:{arch.get('eventKind')}"
        result.append({
            "id": rid, "symbol": sym, "status": "OPEN", "created_at": now,
            "breakout_close_ms": int(event_at), "breakout": arch.get("eventKind"),
            "breakout_label": str(arch.get("label", ""))[:120],
            "sqs": finite(item.get("sqs")), "entry": entry,
            "tp": round(entry * 1.01, 12), "last_price": entry,
            "last_at": now, "highest": entry, "lowest": entry,
            "max_favorable_pct": 0.0, "max_adverse_pct": 0.0,
            "coinbeacon_rvol": rvol,
            **metric, "metric_source": "local_closed_15m_aggregated_4h",
            "metric_status": ("COMPLETE" if all(metric.get(k) is not None for k in
                             ("rsi14_4h", "atr14_pct_4h", "volume_ratio_4h"))
                             else "PARTIAL"),
            "source": "coinbeacon_setup_4h_binance_futures + binance_ws_ticker",
            "returns": {}, "hourly_path": [], "observations": 1,
            "tp_seen_at": None, "finished_at": None,
        })
    return result


def update_prices(state, prices, now):
    changed = False
    for r in state["signals"]:
        if r.get("status") not in ("OPEN", "TP") or now < r["created_at"]:
            continue
        # TP records continue tracking for all 72 hours to measure post-TP paths.
        if now - r["created_at"] > MAX_AGE_SECONDS:
            if r["status"] == "OPEN":
                r["status"] = "TIMEOUT_72H"
            r["finished_at"] = r.get("finished_at") or now
            changed = True
            continue
        value = finite(prices.get(r["symbol"]))
        if value is None or value <= 0:
            continue
        r["last_price"], r["last_at"] = value, now
        r["observations"] = r.get("observations", 0) + 1
        r["highest"] = max(r.get("highest", r["entry"]), value)
        r["lowest"] = min(r.get("lowest", r["entry"]), value)
        ret = (value / r["entry"] - 1) * 100
        r["max_favorable_pct"] = round((r["highest"] / r["entry"] - 1) * 100, 4)
        r["max_adverse_pct"] = round((r["lowest"] / r["entry"] - 1) * 100, 4)
        # Observed ticker price crossed +1%; NOT an executed Binance TP.
        if r["status"] == "OPEN" and value >= r["tp"] and now > r["created_at"]:
            r["status"], r["tp_seen_at"] = "TP", now
        elapsed = now - r["created_at"]
        for hours in (24, 48, 72):
            if elapsed >= hours * 3600 and str(hours) not in r["returns"]:
                r["returns"][str(hours)] = round(ret, 4)
        path = r["hourly_path"]
        if not path or now - path[-1]["at"] >= 3600:
            path.append({"at": now, "return_pct": round(ret, 4), "price": value})
        changed = True
    return changed


def export_payload(state, now):
    after = now - 7 * 86400
    rows = [r for r in state.get("signals", []) if r.get("created_at", 0) >= after]
    rows.sort(key=lambda r: (r["created_at"], r["id"]))
    return {
        "schema": MODEL, "generated_at": now, "window_days": 7,
        "strategy": "PAPER_ONLY_4H_BULLISH_COINBEACON_BREAK",
        "tp_pct": PAPER_TP_PCT, "stop_loss": None,
        "caveat": "Ticker observations only, no exchange fills, missing ticks can miss TP/drawdown. 24/48/72 returns are nearest post-horizon observation.",
        "signals": rows,
        "summary": {"n": len(rows), "open": sum(r["status"] == "OPEN" for r in rows),
                    "tp": sum(r["tp_seen_at"] is not None for r in rows),
                    "timeout": sum(r["status"] == "TIMEOUT_72H" for r in rows)},
    }


def upload_pair(directory, payload, now, env=None):
    """Store a private 7d rolling JSON snapshot AND exact matching ZIP in R2."""
    from crash_report_storage import R2_ENV_NAMES, r2_configured
    env = os.environ if env is None else env
    if not r2_configured(env):
        raise ValueError("R2_NOT_CONFIGURED")
    import boto3
    account, bucket, access, secret = [env[n].strip() for n in R2_ENV_NAMES]
    prefix = str(env.get("CRASH_REPORT_R2_PREFIX") or "crash-bounce").strip().strip("/")
    if not prefix or ".." in prefix or "\\" in prefix:
        raise ValueError("INVALID_R2_PREFIX")
    day = dt.datetime.fromtimestamp(now, PARIS).strftime("%Y-%m-%d")
    key_prefix = f"{prefix}/4h-break/{day[:4]}/{day[5:7]}/{day[8:10]}"
    filename = "4h_break_" + day
    work = Path(directory) / "4h_break_exports"
    work.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path_json, path_zip = work / (filename+".json"), work / (filename+".zip")
    try:
        path_json.write_bytes(raw)
        with zipfile.ZipFile(path_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(filename+".json", raw)
        client = boto3.client("s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
                             aws_access_key_id=access, aws_secret_access_key=secret, region_name="auto")
        keys = {}
        for path, mime in ((path_json, "application/json"), (path_zip, "application/zip")):
            key = key_prefix+"/"+path.name
            client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": mime})
            actual = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
            if actual != path.stat().st_size:
                raise ValueError("R2_UPLOAD_SIZE_MISMATCH")
            keys[path.suffix.lstrip(".")] = key
        return keys
    finally:
        for path in (path_json, path_zip):
            try:
                path.unlink()
            except OSError:
                pass


def publish_sheet(state, now, log=None):
    """One dedicated tab in the existing EMA Filter workbook."""
    from crash_sheets_dashboard import _credentials, _name
    client, key = _credentials()
    if not client:
        return False
    book = client.open_by_key(key)
    if "emafilter" not in _name(book.title):
        raise ValueError("WRONG_SHEETS_WORKBOOK")
    tabs = {t.title: t for t in book.worksheets()}
    tab = tabs.get(TAB)
    if tab is None:
        tab = book.add_worksheet(title=TAB, rows=82, cols=len(HEADERS))
    elif getattr(tab, "col_count", len(HEADERS)) < len(HEADERS):
        tab.resize(cols=len(HEADERS))
    view = sorted(state["signals"], key=lambda r:r["created_at"], reverse=True)[:79]
    def fmt(x):
        return round(x, 4) if isinstance(x,(float,int)) and math.isfinite(x) else "—"
    rows = [HEADERS]
    for r in view:
        returns = r.get("returns") or {}
        rows.append([r["symbol"], time_text(r["breakout_close_ms"]/1000, dt.timezone.utc),
            time_text(r["created_at"]), r["breakout_label"], fmt(r.get("sqs")),
            fmt(r["entry"]), fmt(r["tp"]), fmt(r.get("last_price")), r["status"],
            fmt(r.get("rsi14_4h")), fmt(r.get("atr14_pct_4h")),
            fmt(r.get("volume_ratio_4h")), fmt(r.get("coinbeacon_rvol")),
            fmt(r.get("max_adverse_pct")), fmt(r.get("max_favorable_pct")),
            fmt(returns.get("24")), fmt(returns.get("48")), fmt(returns.get("72")),
            time_text(r.get("last_at")), r.get("metric_status"), r.get("source")])
    rows += [[""] * len(HEADERS) for _ in range(max(0,82-len(rows)))]
    book.values_batch_update({"valueInputOption": "RAW", "data": [
        {"range": "'" + TAB + "'!A1:U82", "values": rows}]})
    return True


class Breakout4HShadow:
    """Independent daemon-side research worker, no broker or live order access."""
    def __init__(self, directory, logger, clock=time.time, http_get=None,
                 r2_upload=upload_pair, sheet_writer=publish_sheet):
        self.directory, self.logger, self.clock = directory, logger, clock
        self.http_get = http_get
        self.r2_upload, self.sheet_writer = r2_upload, sheet_writer
        self.path = Path(directory) / "4h_break_shadow_state.json"
        self.state = _load(self.path, {"signals": [], "last_scan_slot": 0, "upload_day": ""})
        self.state.setdefault("signals", [])
        self.state.setdefault("last_scan_slot", 0)
        self.state.setdefault("upload_day", "")
        self.next_tick = 0
        self.next_attempt = 0
        self.next_sheet = 0
        self.lock = threading.Lock()

    def _fetch(self):
        key = (os.getenv("hbreak") or os.getenv("HBREAK") or
               os.getenv("COINBEACON_API_KEY") or "").strip()
        if not key:
            raise ValueError("HBREAK_KEY_NOT_CONFIGURED")
        if self.http_get is None:
            import requests
            result = requests.get(API, headers={"X-API-Key": key},
                                  params={"exchange": "binance-futures", "timeframe": "4h"},
                                  timeout=25)
        else:
            result = self.http_get(API, headers={"X-API-Key": key},
                                   params={"exchange": "binance-futures", "timeframe": "4h"},
                                   timeout=25)
        result.raise_for_status()
        return result.json()

    def _work(self, prices, market, candle_get, now):
        try:
            changed = update_prices(self.state, prices, now)
            slot = last_four_hour_close(now)
            if slot > self.state["last_scan_slot"] and now >= self.next_attempt:
                self.next_attempt = now + 15*60
                try:
                    data = self._fetch()
                    new = parse_setups(data, slot, prices, now, self.directory, candle_get, market)
                    ids = {r["id"] for r in self.state["signals"]}
                    active_symbols = {r["symbol"] for r in self.state["signals"] if r["status"] == "OPEN"}
                    appended = 0
                    for rec in new:
                        if rec["id"] not in ids and rec["symbol"] not in active_symbols:
                            self.state["signals"].append(rec)
                            appended += 1
                    self.state["last_scan_slot"] = slot
                    changed = True
                    self.logger(f"[4H BREAK BEACON] candidates={len(new)}; new_paper_longs={appended}; source=coinbeacon; real_orders=0")
                except Exception as error:
                    # Never print the URL, request headers, API key or HTTP error body.
                    self.logger(f"[4H BREAK BEACON] scan deferred: {type(error).__name__}; retry_15m=1")
            today = dt.datetime.fromtimestamp(now, PARIS).strftime("%Y-%m-%d")
            is_export_time = dt.datetime.fromtimestamp(now, PARIS).hour >= 21
            self.state["signals"] = [r for r in self.state["signals"]
                                     if now - r.get("created_at",0) <= 10 * 86400]
            if changed:
                _save(self.path, self.state)
            if now >= self.next_sheet:
                self.next_sheet = now + 300
                try:
                    self.sheet_writer(self.state, now)
                except Exception as error:
                    self.logger(f"[4H BREAK SHEETS] skipped: {type(error).__name__}")
            if is_export_time and self.state["upload_day"] != today and now >= self.next_attempt - 15*60:
                try:
                    keys = self.r2_upload(self.directory, export_payload(self.state, now), now)
                    self.state["upload_day"] = today
                    _save(self.path, self.state)
                    self.logger(f"[4H BREAK R2] day={today}; json=ok; zip=ok; signals={len(self.state['signals'])}; prefix=4h-break")
                except Exception as error:
                    self.logger(f"[4H BREAK R2] deferred: {type(error).__name__}; no_trade_impact=1")
        except Exception as error:
            self.logger(f"[4H BREAK SHADOW] skipped: {type(error).__name__}; no_trade_impact=1")
        finally:
            self.lock.release()

    def tick(self, tickers, market, candle_get):
        now = self.clock()
        if now < self.next_tick or not self.lock.acquire(False):
            return
        self.next_tick = now + 60
        prices = {t["symbol"]: finite(t.get("lastPrice"))
                  for t in (tickers if isinstance(tickers,list) else [])
                  if isinstance(t,dict) and isinstance(t.get("symbol"),str)}
        # Pass only a ticker snapshot; scan + Sheets + R2 in separate worker thread.
        threading.Thread(target=self._work, args=(prices, dict(market or {}), candle_get, now),
                         name="4h-break-shadow", daemon=True).start()
