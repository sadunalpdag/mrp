"""Durable append-only trading audit stream on the Render persistent disk.

Records only selected public market fields / trading state, never API keys,
signed requests, account balances, or raw WebSocket payloads.
"""
import json
import os
from datetime import datetime, timezone


def append_trade_event(directory, event, at=None):
    if not isinstance(event, dict) or not isinstance(event.get("kind"), str):
        raise ValueError("INVALID_AUDIT_EVENT")
    instant = datetime.fromtimestamp(at, timezone.utc) if at is not None else datetime.now(timezone.utc)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, instant.strftime("%Y-%m-%d") + ".jsonl")
    data = dict(schema_version=1, timestamp=instant.isoformat(), **event)
    raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > 16384:
        raise ValueError("AUDIT_EVENT_TOO_LARGE")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    return path
