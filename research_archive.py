"""Append-only, date-partitioned research runtime records.

Only research/Crash status messages are stored. This does not replace
structured /data/crash_research/YYYY-MM-DD.jsonl market observations.
No API keys, signed URLs or request payloads are archived.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

PREFIXES = ('[CRASH', '[TOP GAINERS', '[BTC HOURLY')
BLOCKED = re.compile(r'(?i)(api[_ -]?key|secret|authorization|bot_token|signature=|x-mbx-apikey|https?://[^ ]*bot[0-9]{5,}:)')
MAX_MESSAGE_LENGTH = 1500


def append_research_runtime(data_dir, message, now=None):
    """Store selected sanitized runtime events in one daily JSONL per UTC day.

    Caller can point DATA_DIR at an existing Render persistent disk (/data);
    no rotation or deploy is allowed to truncate older files.
    """
    if not isinstance(message, str) or not message.startswith(PREFIXES):
        return False
    if BLOCKED.search(message):
        return False
    timestamp = time.time() if now is None else float(now)
    date = datetime.fromtimestamp(timestamp, timezone.utc)
    folder = os.path.join(data_dir, 'research_archive', 'runtime')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, date.strftime('%Y-%m-%d')+'.jsonl')
    record = {'ts_utc': date.isoformat(), 'message': message[:MAX_MESSAGE_LENGTH]}
    # No read/overwrite of previous dates: O_APPEND only.
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # os.fdopen owns fd after successful construction; the caller must
        # not silently assume a failed write was committed.
        raise
    return True


def archive_location_ok(data_dir, mount='/data'):
    """Indicate whether the configured data directory is under the disk mount."""
    root = os.path.realpath(mount)
    target = os.path.realpath(data_dir)
    return target == root or target.startswith(root + os.sep)
