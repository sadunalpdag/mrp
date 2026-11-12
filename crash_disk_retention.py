"""Bounded, conservative Render-disk cleanup for Crash Bounce research.

Only explicitly named, non-live research artifacts can be removed. Trading
state, pending orders/TP, and current candle caches are never targeted.
Private Cloudflare R2 archives are never touched.
"""
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone

SWEEP_INTERVAL_SECONDS = 10 * 86400
RETENTION_DAYS = 10
STATE_NAME = ".disk_retention.json"
OBSERVATION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")
REPORT_RE = re.compile(r"^crash_data_\d+(?:\.part\d{3})?\.zip(?:\.tmp)?$")
CACHE_RE = re.compile(r"^[A-Za-z0-9_-]{2,80}\.json$")
CACHE_DIRS = ("history_cache", "fallback_cache", "bybit_reference",
              "coinbase_reference", "urgent_bybit")
CACHE_PROTECTED = frozenset(("pending.json", "download_schedule.json"))


def _load_last_run(path):
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream).get("last_run_at")
        parsed = float(value)
        if math.isfinite(parsed) and parsed >= 0:
            return parsed
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def purge_disabled_risk_v1_raw(data_dir, log=None):
    """Explicit opt-in purge of ONLY raw daily risk_v1 JSONL files.

    Requires both the shadow collector off and the operator purge flag on.
    Never removes state.json, summary.json, exchange data or report ZIPs.
    """
    if (os.getenv("CRASH_RISK_V1_ENABLED", "0") == "1"
            or os.getenv("CRASH_RISK_V1_PURGE_LOGS", "0") != "1"):
        return None
    root = os.path.join(os.path.abspath(data_dir), "crash_research")
    target = os.path.join(root, "risk_v1")
    if (not os.path.isdir(root) or os.path.islink(root)
            or not os.path.isdir(target) or os.path.islink(target)):
        return None
    deleted = 0
    freed = 0
    errors = 0
    try:
        with os.scandir(target) as entries:
            for entry in entries:
                # Strict YYYY-MM-DD.jsonl ONLY; no recursion, no symlinks.
                if not OBSERVATION_RE.fullmatch(entry.name) or not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    size = entry.stat(follow_symlinks=False).st_size
                    os.unlink(entry.path)
                    deleted += 1
                    freed += size
                except OSError:
                    errors += 1
    except OSError:
        errors += 1
    if (deleted or errors) and callable(log):
        log(f"[CRASH RISK V1 RAW CLEANUP] deleted={deleted}; "
            f"freed_bytes={freed}; errors={errors}; "
            "state_summary_and_live_orders_untouched=1")
    return {"deleted_files": deleted, "freed_bytes": freed, "errors": errors}


def cleanup_if_due(data_dir, now=None, pending=None, log=None):
    """Run at most once per 10 days; first run cleans pre-existing stale files.

    Keep 10 UTC calendar dates of original daily observation logs, and preserve
    any cache/ZIP touched in the preceding 10*24h. A report awaiting transfer
    cannot be deleted, even when older than the retention horizon.
    """
    now = time.time() if now is None else float(now)
    if not math.isfinite(now) or now <= 0:
        raise ValueError("INVALID_RETENTION_TIME")
    # Independent of the normal ten-day retention throttle: raw shadow
    # research was explicitly retired to prevent disk exhaustion.
    purge_disabled_risk_v1_raw(data_dir, log=log)
    research = os.path.join(os.path.abspath(data_dir), "crash_research")
    # Do not traverse a symlink-controlled or absent research directory.
    if not os.path.isdir(research) or os.path.islink(research):
        return None
    marker = os.path.join(research, STATE_NAME)
    last_run = _load_last_run(marker)
    if last_run is not None and now < last_run + SWEEP_INTERVAL_SECONDS:
        return None

    today = datetime.fromtimestamp(now, timezone.utc).date()
    oldest_allowed = today - timedelta(days=RETENTION_DAYS - 1)
    mtime_cutoff = now - RETENTION_DAYS * 86400
    pending = pending if isinstance(pending, dict) else {}
    protected = set()
    for value in [pending.get("path")] + (pending.get("parts") if isinstance(pending.get("parts"), list) else []):
        if isinstance(value, str) and value:
            protected.add(os.path.abspath(value))
            protected.add(os.path.abspath(value + ".tmp"))

    removed = 0
    freed_bytes = 0
    errors = 0

    def scan(directory, acceptable):
        nonlocal removed, freed_bytes, errors
        if not os.path.isdir(directory) or os.path.islink(directory):
            return
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if entry.path in protected or not acceptable(entry):
                        continue
                    try:
                        size = entry.stat(follow_symlinks=False).st_size
                        os.unlink(entry.path)
                        removed += 1
                        freed_bytes += size
                    except OSError:
                        errors += 1
        except OSError:
            errors += 1

    def dated_observation(entry):
        if not OBSERVATION_RE.fullmatch(entry.name):
            return False
        try:
            day = datetime.strptime(entry.name[:-6], "%Y-%m-%d").date()
        except ValueError:
            return False
        return day < oldest_allowed

    def old_report(entry):
        return (REPORT_RE.fullmatch(entry.name) is not None
                and entry.stat(follow_symlinks=False).st_mtime < mtime_cutoff)

    def stale_cache(entry):
        return (entry.name not in CACHE_PROTECTED
                and CACHE_RE.fullmatch(entry.name) is not None
                and entry.stat(follow_symlinks=False).st_mtime < mtime_cutoff)

    # Observation JSONL files and orphaned report ZIPs only; no other root files.
    scan(research, lambda entry: dated_observation(entry) or old_report(entry))
    for directory in CACHE_DIRS:
        scan(os.path.join(research, directory), stale_cache)

    # The timestamp is written after cleanup, so restarts do not sweep again.
    report = {"last_run_at": now, "retention_days": RETENTION_DAYS,
              "interval_days": SWEEP_INTERVAL_SECONDS // 86400,
              "deleted_files": removed, "freed_bytes": freed_bytes,
              "errors": errors}
    tmp = marker + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, marker)
    except OSError:
        errors += 1
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if callable(log):
        log(f"[CRASH DISK RETENTION] deleted={removed}; freed_bytes={freed_bytes}; "
            f"errors={errors}; keep_days={RETENTION_DAYS}; next_sweep_days=10; "
            "orders_and_R2_untouched=1")
    return report
