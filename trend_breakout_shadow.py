"""Read-only trend-breakout research with optional R2 JSON+ZIP exports.

No Binance REST calls, extra websocket subscriptions, trading decisions,
Google Sheets edits, or changes to existing Crash Bounce orders.
"""
import json
import math
import os
import re
import threading
import time
import zipfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
SCHEMA = "trend_breakout_shadow_v1"
HORIZONS = (24, 48, 72)
FOLLOWUP_SECONDS = 72 * 3600
KEEP_SECONDS = 10 * 86400
MAX_SYMBOLS = 20
MAX_EVENTS = 500
STEP_MS = 900000
SYMBOL = re.compile(r"^[A-Z0-9]{2,24}USDT$")


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _date(ts):
    return datetime.fromtimestamp(ts, PARIS).strftime("%Y-%m-%d")


def _joined_rows(directory, symbol, now, candle_get):
    from crash_levels_5d import load_cached
    rows = load_cached(directory, symbol, candle_get(symbol), now)
    # Latest CLOSED bars only. Existing cache merging validates every row.
    return rows


def _closed_bar(row):
    try:
        opened, closed = int(row[0]), int(row[6])
        hi, lo, price = map(float, (row[2], row[3], row[4]))
        if (closed - opened != STEP_MS - 1 or opened % STEP_MS or
                any(not math.isfinite(v) or v <= 0 for v in (hi, lo, price))
                or not lo <= price <= hi):
            return None
        return (closed + 1) / 1000, hi, lo, price
    except (IndexError, ValueError, TypeError, OverflowError):
        return None


def _match_signal(symbol, at, history):
    candidates = [r for r in history if isinstance(r, dict)
                  and r.get("symbol") == symbol and
                  _number(r.get("created_at")) is not None and
                  0 <= at - float(r["created_at"]) <= 1800]
    if not candidates:
        return {}
    r = max(candidates, key=lambda x: x["created_at"])
    shadow = r.get("failure_shadow") or {}
    return {
        "signal_id": r.get("signal_id"),
        "v2_status": r.get("v2_status"),
        "risk_flags": {k: shadow.get("flags", {}).get(k)
                       for k in ("weak_volume", "falling_rsi_ema",
                                 "repeat_lower_entry", "low_atr_15m")},
        "risk_known": shadow.get("known_flags"),
        "risk_count": shadow.get("flag_count"),
        "risk_sizing_version": shadow.get("sizing_policy_version"),
        "signal_entry": _number(r.get("entry")),
        "signal_tp": _number(r.get("tp")),
    }


def _add_bar(event, close_at, high, low, close):
    """Use post-observation closed-bar extrema; intra-bar ordering is UNKNOWN."""
    initial = event["reference_price"]
    until = event["observed_at"] + FOLLOWUP_SECONDS
    if not event["observed_at"] < close_at <= until:
        return False
    if close_at <= event.get("last_bar_at", event["observed_at"]):
        return False
    up = round(100 * (high / initial - 1), 5)
    down = round(100 * (low / initial - 1), 5)
    end = round(100 * (close / initial - 1), 5)
    event["max_up_pct"] = max(event["max_up_pct"], up)
    event["max_down_pct"] = min(event["max_down_pct"], down)
    if event.get("first_plus_1pct_at") is None and up >= 1:
        event["first_plus_1pct_at"] = close_at
    if event.get("first_minus_1pct_at") is None and down <= -1:
        event["first_minus_1pct_at"] = close_at
    if event.get("last_bar_at") and close_at - event["last_bar_at"] > 900 + 60:
        event["coverage_gaps"] += 1
    event["last_bar_at"] = close_at
    event["last_close_pct"] = end
    event["bars_observed"] += 1
    for hours in HORIZONS:
        key = str(hours) + "h"
        target = event["observed_at"] + hours * 3600
        if event["horizons"][key] is None and close_at >= target:
            if close_at - target <= 30 * 60:
                event["horizons"][key] = {
                    "close_at": close_at, "close_pct": end,
                    "max_up_pct": event["max_up_pct"],
                    "max_down_pct": event["max_down_pct"],
                    "bars_observed": event["bars_observed"],
                    "coverage_gaps": event["coverage_gaps"],
                }
            else:
                event["horizons"][key] = {"status": "GAP_AT_HORIZON"}
    if close_at >= until - 60:
        event["phase"] = "MATURED"
    return True


def _r2_upload_files(json_path, zip_path, day, env=None):
    """Upload and HEAD-verify both private objects; never create public links."""
    from crash_report_storage import R2_ENV_NAMES, r2_configured
    env = os.environ if env is None else env
    if not r2_configured(env):
        raise ValueError("R2_NOT_CONFIGURED")
    import boto3
    from boto3.s3.transfer import TransferConfig
    bucket = str(env[R2_ENV_NAMES[1]]).strip()
    account = str(env[R2_ENV_NAMES[0]]).strip()
    prefix = str(env.get("CRASH_REPORT_R2_PREFIX") or "crash-bounce").strip().strip("/")
    if not account.isascii() or not account.isalnum() or not 16 <= len(account) <= 64:
        raise ValueError("INVALID_R2_ACCOUNT_ID")
    if not prefix or ".." in prefix or "\\" in prefix:
        raise ValueError("INVALID_R2_PREFIX")
    client = boto3.client(
        "s3", endpoint_url="https://" + account + ".r2.cloudflarestorage.com",
        aws_access_key_id=str(env[R2_ENV_NAMES[2]]).strip(),
        aws_secret_access_key=str(env[R2_ENV_NAMES[3]]).strip(),
        region_name="auto",
    )
    y, m, d = day.split("-")
    root = "/".join((prefix, "trend-breakout", y, m, d))
    targets = []
    for path, media in ((json_path, "application/json"), (zip_path, "application/zip")):
        key = root + "/" + os.path.basename(path)
        client.upload_file(
            path, bucket, key,
            ExtraArgs={"ContentType": media},
            Config=TransferConfig(multipart_threshold=8 * 1024 * 1024,
                                  multipart_chunksize=8 * 1024 * 1024,
                                  max_concurrency=1))
        actual = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        if actual != os.path.getsize(path):
            raise ValueError("R2_UPLOAD_SIZE_MISMATCH")
        targets.append(key)
    return targets


class TrendBreakoutShadow:
    """Prospective observations of <=20 EXISTING Crash Bounce watched coins only.

    API operations run on daemon thread. Only code named in this class controls
    its own state; exception handling never changes the trading executor.
    """

    def __init__(self, data_dir, log=print, now_fn=time.time, uploader=_r2_upload_files):
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "trend_breakout_shadow")
        os.makedirs(self.path, exist_ok=True)
        self.state_file = os.path.join(self.path, "state.json")
        self.log, self.now_fn, self.uploader = log, now_fn, uploader
        self.lock = threading.Lock()
        self.running = False
        self.next_scan = 0
        try:
            with open(self.state_file, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or data.get("schema_version") != SCHEMA:
                raise ValueError("STATE_VERSION_INVALID")
            self.state = data
            if not isinstance(self.state.get("events"), list):
                raise ValueError("STATE_EVENTS_INVALID")
        except (OSError, ValueError, TypeError):
            self.state = {"schema_version": SCHEMA, "events": [],
                          "last_uploaded_day": None, "retry_after": 0}
        self.log("[TREND SHADOW] READY; research_only=1; no_extra_rest=1; no_orders=1")

    def _save(self):
        _atomic(self.state_file, self.state)

    def _scan(self, tickers, candle_get, history, now, watched_symbols):
        from crash_levels_5d import analyze
        # CrashBounceWebSocketCache.reconcile() already selects these symbols.
        # Research is strictly a CONSUMER of existing kline subscriptions,
        # never a second scanner for another 20 or 50 coins.
        allowed = {s for s in watched_symbols if isinstance(s, str)
                   and SYMBOL.fullmatch(s)}
        eligible = []
        for ticker in tickers:
            if not isinstance(ticker, dict):
                continue
            symbol = ticker.get("symbol", "")
            drop = _number(ticker.get("priceChangePercent"))
            if symbol in allowed and drop is not None:
                eligible.append((drop, symbol))
        ranked = [s for _, s in sorted(eligible)[:MAX_SYMBOLS]]
        active = sorted(
            (e for e in self.state["events"]
             if e.get("phase") == "TRACKING"
             and now < e["observed_at"] + FOLLOWUP_SECONDS + 3600),
            key=lambda e: e["observed_at"], reverse=True)
        # At most 20 symbols total, INCLUDING existing 72h follow-ups.
        # Prioritize outstanding follow-ups over discovering new signals.
        active_symbols = list(dict.fromkeys(e["symbol"] for e in active))
        # Do not add websocket subscriptions even for 72h follow-ups.
        # If a coin leaves Crash's watchlist, any follow-up gaps remain explicit.
        symbols = [s for s in dict.fromkeys(active_symbols + ranked)
                   if s in allowed][:MAX_SYMBOLS]
        events_seen = {(e["symbol"], e["direction"], e["confirmed_at"])
                       for e in self.state["events"]}
        new_count = 0
        completed_bars = 0
        for symbol in symbols:
            try:
                rows = _joined_rows(self.data_dir, symbol, now, candle_get)
                valid = [(closed, high, low, close) for r in rows
                         if (parsed := _closed_bar(r)) is not None
                         for closed, high, low, close in (parsed,) if closed <= now]
                if not valid:
                    continue
                valid.sort()
                relevant = [e for e in self.state["events"] if e["symbol"] == symbol
                            and e.get("phase") == "TRACKING"]
                for e in relevant:
                    for closed, high, low, price in valid:
                        if _add_bar(e, closed, high, low, price):
                            completed_bars += 1
                    if now >= e["observed_at"] + FOLLOWUP_SECONDS + 3600:
                        e["phase"] = "MATURED" if (
                            e.get("last_bar_at", 0) >=
                            e["observed_at"] + FOLLOWUP_SECONDS - 60) else "INCOMPLETE"
                if symbol not in ranked:
                    continue
                result = analyze(rows, now)
                if result["status"] != "KIRDI" or result["direction"] not in ("YUKARI", "AŞAĞI"):
                    continue
                confirmed = _number(result["confirmed_at"])
                if confirmed is None or not 0 <= now - confirmed <= 45 * 60:
                    continue
                event_key = (symbol, result["direction"], confirmed)
                if event_key in events_seen:
                    continue
                # Do not register multiple breakouts on one unchanged symbol
                # and direction within 12h as independent research samples.
                if any(e["symbol"] == symbol and e["direction"] == result["direction"]
                       and now - e["observed_at"] < 12 * 3600
                       for e in self.state["events"]):
                    continue
                baseline_at, _, _, baseline = valid[-1]
                if not 0 < now - baseline_at <= 30 * 60:
                    continue
                row = {
                    "event_id": symbol + "-" + result["direction"] + "-" + str(int(confirmed)),
                    "symbol": symbol, "direction": result["direction"],
                    "confirmed_at": confirmed, "observed_at": baseline_at,
                    "reference_price": baseline, "status_at_observation": "KIRDI",
                    "distance_to_upper_pct": result["distance_pct"],
                    "coverage": result["coverage"],
                    "support": result["support"], "resistance": result["resistance"],
                    "fib_band": result["fib_band"],
                    "rank_in_crash_watch": ranked.index(symbol) + 1,
                    "signal_context": _match_signal(symbol, baseline_at, history),
                    "phase": "TRACKING", "last_bar_at": baseline_at,
                    "last_close_pct": 0.0, "max_up_pct": 0.0,
                    "max_down_pct": 0.0, "bars_observed": 0,
                    "coverage_gaps": 0, "first_plus_1pct_at": None,
                    "first_minus_1pct_at": None,
                    "horizons": {str(h) + "h": None for h in HORIZONS},
                    "intrabar_order": "UNKNOWN",
                    "price_basis": "CLOSED_15M_HIGH_LOW",
                }
                self.state["events"].append(row)
                events_seen.add(event_key)
                new_count += 1
            except Exception as error:
                self.log(f"[TREND SHADOW SCAN SKIP] {symbol}; {type(error).__name__}")
        # Only old completed observations can be pruned; never prune an active
        # three-day follow-up event. 10 days of disk state remain.
        self.state["events"] = [
            e for e in self.state["events"]
            if now - e.get("observed_at", 0) < KEEP_SECONDS
            or e.get("phase") == "TRACKING"][-MAX_EVENTS:]
        self._save()
        self.log(f"[TREND SHADOW] symbols={len(symbols)}/20; new_breakouts={new_count}; "
                 f"events={len(self.state['events'])}; followup_bars={completed_bars}; "
                 "execution=OFF; no_extra_rest=1")

    def _export(self, day, now):
        """One AI-friendly JSON plus an identical JSON inside a private backup ZIP."""
        recent = [e for e in self.state["events"]
                  if now - e.get("observed_at", 0) <= 7 * 86400]
        recent.sort(key=lambda x: (x["observed_at"], x["event_id"]))
        exported = {
            "schema_version": SCHEMA,
            "research_only": True,
            "trading_actions": False,
            "export_day_paris": day,
            "exported_at": now,
            "time_zone": "Europe/Paris",
            "sampling": "max_20_of_existing_crash_bounce_watched_symbols",
            "no_extra_websocket_subscriptions": True,
            "breakout_definition": "5d channel; 2 consecutive closed 15m bars; 0.15% buffer",
            "followup_hours": list(HORIZONS),
            "price_path_disclaimer": "Candle OHLC does not reveal intrabar order; no fill or profit claims.",
            "event_count": len(recent),
            "events": recent,
        }
        stem = "trend_breakout_" + day
        json_path = os.path.join(self.path, stem + ".json")
        zip_path = os.path.join(self.path, stem + ".zip")
        _atomic(json_path, exported)
        temp_zip = zip_path + ".tmp"
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            archive.write(json_path, arcname=stem + ".json")
        os.replace(temp_zip, zip_path)
        return json_path, zip_path, exported["event_count"]

    def _upload(self, now):
        day = _date(now)
        if self.state.get("last_uploaded_day") == day:
            return
        retry_after = _number(self.state.get("retry_after")) or 0
        if now < retry_after:
            return
        local = datetime.fromtimestamp(now, PARIS)
        if (local.hour, local.minute) < (21, 15):
            return
        paths = []
        try:
            jp, zp, count = self._export(day, now)
            paths = [jp, zp]
            result = self.uploader(jp, zp, day)
            if not isinstance(result, list) or len(result) != 2:
                raise ValueError("R2_EXPORT_NOT_VERIFIED")
            self.state["last_uploaded_day"] = day
            self.state["retry_after"] = 0
            self.state["uploaded_keys"] = result
            self._save()
            for path in paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.log(f"[TREND R2] JSON_AND_ZIP_OK; date={day}; events={count}; "
                     "objects=2; private=1")
        except Exception as error:
            self.state["retry_after"] = now + 3600
            self._save()
            self.log(f"[TREND R2] retry_60m; reason={type(error).__name__}; "
                     "orders_untouched=1")

    def tick(self, tickers, candle_get, history=None, watched_symbols=None):
        """Never block the caller: full scan/upload is in a research daemon."""
        now = self.now_fn()
        if self.running or not self.lock.acquire(blocking=False):
            return
        try:
            self.running = True
            want_scan = now >= self.next_scan
            self.next_scan = max(self.next_scan, (int(now) // 900 + 1) * 900)
            # Main worker calls tick() frequently. A no-op must be truly cheap:
            # no threads/files until next closed 15m scan or scheduled R2 upload.
            local = datetime.fromtimestamp(now, PARIS)
            upload_due = (
                (local.hour, local.minute) >= (21, 15)
                and self.state.get("last_uploaded_day") != _date(now)
                and now >= (_number(self.state.get("retry_after")) or 0)
            )
            if not want_scan and not upload_due:
                self.running = False
                self.lock.release()
                return
            cloned = [dict(t) for t in tickers or [] if isinstance(t, dict)]
            records = [dict(r) for r in (history or []) if isinstance(r, dict)]
            allowed = set(watched_symbols or ())
            def worker():
                try:
                    if want_scan:
                        self._scan(cloned, candle_get, records, now, allowed)
                    self._upload(now)
                except Exception as error:
                    self.log("[TREND SHADOW ERROR] " + type(error).__name__)
                finally:
                    self.running = False
                    self.lock.release()
            threading.Thread(target=worker, name="trend-breakout-shadow",
                             daemon=True).start()
        except Exception:
            self.running = False
            self.lock.release()
            raise
