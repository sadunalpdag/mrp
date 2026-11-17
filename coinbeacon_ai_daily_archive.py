"""One R2 JSON per Europe/Paris 01:00–01:00 Gemini research day.

Stores unfiltered paper candidates, 480 historical closed 15m candles, frozen
initial plans, 8-hour out-of-sample observations, outcomes and Gemini reviews.
Uses a persistent SQLite revision journal; reuploads the same key when a late
8-hour result arrives. No exchange order APIs or trade-engine imports.
"""
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
SCHEMA = "coinbeacon_gemini_daily_v2"
MIN_REUPLOAD_MS = 5 * 60_000


def research_day(observed_ms):
    when = datetime.fromtimestamp(int(observed_ms) / 1000, PARIS)
    if when.hour < 1:
        when -= timedelta(days=1)
    return when.date().isoformat()


def bounds(day):
    start = datetime.fromisoformat(day).replace(hour=1, tzinfo=PARIS)
    return start, start + timedelta(days=1)


def archive_key(day):
    year, month, _ = day.split("-")
    return f"coinbeacon-ai-learning/daily/{year}/{month}/gemini_learning_{day}.json"


def ensure_schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS daily_archives (
        research_day TEXT PRIMARY KEY,
        revision INTEGER NOT NULL DEFAULT 1,
        uploaded_revision INTEGER NOT NULL DEFAULT 0,
        last_attempt_ms INTEGER NOT NULL DEFAULT 0,
        last_success_ms INTEGER NOT NULL DEFAULT 0,
        last_error TEXT)""")
    names = {row[1] for row in db.execute("PRAGMA table_info(cases)")}
    if "research_day" not in names:
        db.execute("ALTER TABLE cases ADD COLUMN research_day TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_cases_research_day ON cases(research_day)")


def _mark_day(db, day):
    db.execute("""INSERT INTO daily_archives (research_day,revision)
        VALUES (?,1) ON CONFLICT(research_day)
        DO UPDATE SET revision=daily_archives.revision+1""", (day,))


def mark_dirty(db, observed_ms):
    _mark_day(db, research_day(observed_ms))


def backfill(db):
    """One-time migration for preexisting individual-case records."""
    ensure_schema(db)
    for identifier, raw in db.execute(
            "SELECT id,payload FROM cases WHERE research_day IS NULL").fetchall():
        try:
            day = research_day(json.loads(raw)["observed_ms"])
        except (ValueError, KeyError, TypeError, OverflowError):
            continue
        db.execute("UPDATE cases SET research_day=? WHERE id=?", (day, identifier))
        _mark_day(db, day)
    # Guard against already-labeled records missing a journal row.
    for (day,) in db.execute("SELECT DISTINCT research_day FROM cases "
                             "WHERE research_day IS NOT NULL").fetchall():
        db.execute("INSERT OR IGNORE INTO daily_archives (research_day) VALUES (?)", (day,))
    db.commit()


def build_payload(db, day, now_ms=None):
    """Deterministic, full-fidelity research data without future-leakage in the first plan."""
    records = [json.loads(raw) for (raw,) in db.execute(
        "SELECT payload FROM cases WHERE research_day=? ORDER BY rowid", (day,))]
    if not records:
        return None
    dispatches = [
        {"case_id": cid, "symbol": sym, "phase": phase,
         "sent_ms": sent, "outcome": outcome}
        for cid, sym, phase, sent, outcome in db.execute(
            "SELECT d.case_id,d.symbol,d.phase,d.sent_ms,d.outcome "
            "FROM gemini_dispatches d JOIN cases c ON c.id=d.case_id "
            "WHERE c.research_day=? ORDER BY d.sent_ms", (day,))]
    state_counts = Counter(r.get("state", "UNKNOWN") for r in records)
    plan_counts = Counter((r.get("initial_plan") or {}).get("direction", "NO_RESPONSE")
                          for r in records)
    labels = Counter((r.get("outcome") or {}).get("label", "NOT_YET_SCORED")
                     for r in records)
    tags = Counter(t for r in records
                   for t in (r.get("final_review") or {}).get("learning_tags", [])
                   if isinstance(t, str))
    quality = {
        "complete_8h_coverage": sum(
            (r.get("outcome") or {}).get("coverage_complete") is True for r in records),
        "incomplete_8h_coverage": sum(
            (r.get("outcome") or {}).get("coverage_complete") is False for r in records),
        "ambiguous_entry_tp_same_candle": sum(
            (r.get("outcome") or {}).get("tp_ambiguous_in_entry_candle") is True
            for r in records),
        "with_initial_gemini_response": sum(isinstance(r.get("initial_raw"), dict)
                                            for r in records),
        "with_final_gemini_review": sum(isinstance(r.get("final_raw"), dict)
                                       for r in records),
    }
    usage = Counter()
    for r in records:
        for phase in ("initial_raw", "final_raw"):
            metadata = (r.get(phase) or {}).get("_gemini_usage") or {}
            for name in ("promptTokenCount", "candidatesTokenCount",
                         "totalTokenCount", "cachedContentTokenCount"):
                value = metadata.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[name] += int(value)
    start, end = bounds(day)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    final = all(r.get("state") in ("COMPLETE", "AI_ERROR") for r in records)
    report = {
        "schema": SCHEMA,
        "research_day": day,
        "timezone": "Europe/Paris",
        "window_start": start.isoformat(),
        "window_end_exclusive": end.isoformat(),
        "updated_at_utc": datetime.fromtimestamp(now / 1000, timezone.utc).isoformat(),
        "finalized": final and now >= int(end.timestamp() * 1000),
        "mode": "PAPER_ONLY_NO_REAL_ORDERS",
        "data_notes": [
            "Signals are only those received and accepted for research; upstream API selection or missing candles may exclude others.",
            "Initial prediction uses pre-signal closed candles; future_candles are post-prediction observations.",
            "8-hour outcomes are observational, not executable orders or guaranteed fills.",
            "OHLC alone cannot resolve whether entry or TP happened first within the same 15m candle.",
            "Gemini text and learning tags are hypotheses, not model retraining."
        ],
        "summary": {
            "signals": len(records),
            "unique_coins": len({r.get("symbol") for r in records}),
            "states": dict(state_counts),
            "gemini_decisions": dict(plan_counts),
            "outcome_labels": dict(labels),
            "gemini_send_attempts": len(dispatches),
            "learning_tags": dict(tags),
            "quality": quality,
            "gemini_token_usage": dict(usage),
        },
        "gemini_dispatches": dispatches,
        "cases": records,
    }
    return report


def _put_r2(key, body):
    import boto3
    from crash_report_storage import R2_ENV_NAMES
    names = [os.environ[n].strip() for n in R2_ENV_NAMES]
    client = boto3.client("s3",
        endpoint_url=f"https://{names[0]}.r2.cloudflarestorage.com",
        aws_access_key_id=names[2], aws_secret_access_key=names[3], region_name="auto")
    client.put_object(Bucket=names[1], Key=key, Body=body,
                      ContentType="application/json")
    head = client.head_object(Bucket=names[1], Key=key)
    if head.get("ContentLength") != len(body):
        raise ValueError("R2_DAILY_ARCHIVE_SIZE_MISMATCH")


def upload_day(db_factory, day):
    """Upload one snapshot and acknowledge ONLY its snapshot revision."""
    from crash_report_storage import r2_configured
    if not r2_configured():
        return False
    db = db_factory()
    try:
        db.execute("BEGIN")
        rev_row = db.execute(
            "SELECT revision FROM daily_archives WHERE research_day=?", (day,)).fetchone()
        revision = rev_row[0] if rev_row else None
        payload = build_payload(db, day) if revision is not None else None
        db.commit()
    finally:
        db.close()
    if payload is None:
        return False
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    key = archive_key(day)
    _put_r2(key, body)
    db = db_factory()
    try:
        db.execute("UPDATE daily_archives SET uploaded_revision=MAX(uploaded_revision,?), "
                   "last_success_ms=?,last_error=NULL WHERE research_day=?",
                   (revision, int(time.time() * 1000), day))
        db.commit()
    finally:
        db.close()
    return True


def sync_due(db_factory, logger=None, now_ms=None):
    """Retry failed uploads, throttle active-day uploads and preserve late 8h updates."""
    from crash_report_storage import r2_configured
    if not r2_configured():
        return False
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    db = db_factory()
    try:
        row = db.execute(
            "SELECT research_day FROM daily_archives "
            "WHERE revision>uploaded_revision AND "
            "(last_attempt_ms=0 OR last_attempt_ms<=?) "
            "ORDER BY research_day DESC LIMIT 1",
            (now - MIN_REUPLOAD_MS,)).fetchone()
        if not row:
            return False
        day = row[0]
        db.execute("UPDATE daily_archives SET last_attempt_ms=? WHERE research_day=?",
                   (now, day))
        db.commit()
    finally:
        db.close()
    try:
        ok = upload_day(db_factory, day)
        if ok and logger:
            logger.info("[CB AI DAILY R2] uploaded key=%s", archive_key(day))
        return ok
    except Exception as exc:
        db = db_factory()
        try:
            db.execute("UPDATE daily_archives SET last_error=? WHERE research_day=?",
                       (type(exc).__name__, day))
            db.commit()
        finally:
            db.close()
        if logger:
            logger.warning("[CB AI DAILY R2] failed day=%s error=%s", day, type(exc).__name__)
        return False
