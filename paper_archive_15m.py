"""Read-only Binance Futures DAILY ZIP archives; no Binance REST API calls.

UTC daily 15m klines become available after publication (usually next day).
Missing/unpublished days remain pending; never fabricate candles or fills.
Shared on-disk cache across HBREAK and CB2; one ZIP per symbol/day.
"""
import csv
import io
import logging
import os
import re
import threading
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import requests

LOG = logging.getLogger("paper_archive")
ROOT = Path(os.getenv("DATA_DIR", "/data")) / "paper_15m_archive"
HOST = "https://data.binance.vision/data/futures/um/daily/klines"
LOCK = threading.RLock()
DAY_MS = 86400000
BAR_MS = 900000
MAX_BYTES = 4_000_000
# Avoid retrying unpublished or missing archives every paper cycle.
MISS_RETRY_SEC = 3600
_misses = {}

def _day(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc).date()

def _load_day(symbol, day, http):
    name = day.isoformat()
    dest = ROOT / symbol / (name + ".csv")
    if dest.exists():
        return _parse(dest.read_text(encoding="utf-8"))
    key = (symbol, name)
    if time.time() - _misses.get(key, 0) < MISS_RETRY_SEC:
        return None
    url = f"{HOST}/{quote(symbol)}/15m/{quote(symbol)}-15m-{name}.zip"
    try:
        response = http.get(url, timeout=(5, 25))
        if response.status_code in (403, 404):
            _misses[key] = time.time()
            return None
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            raise ValueError("ARCHIVE_TOO_LARGE")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            files = [x for x in archive.namelist() if x.endswith(".csv") and "/" not in x]
            if len(files) != 1:
                raise ValueError("INVALID_ARCHIVE_CONTENTS")
            raw = archive.read(files[0])
        text = raw.decode("utf-8-sig")
        candles = _parse(text)
        if not candles:
            raise ValueError("EMPTY_ARCHIVE")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(dest)
        LOG.info("[PAPER ARCHIVE] downloaded symbol=%s day=%s candles=%d", symbol, name, len(candles))
        return candles
    except (requests.RequestException, ValueError, zipfile.BadZipFile, OSError) as exc:
        _misses[key] = time.time()
        LOG.warning("[PAPER ARCHIVE] pending symbol=%s day=%s reason=%s", symbol, name, type(exc).__name__)
        return None

def _parse(text):
    result = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 8 or not row[0].isdigit():
            continue
        try:
            if int(row[6]) - int(row[0]) not in (899999, 900000):
                continue
            float(row[2]); float(row[3]); float(row[4])
            result.append(row)
        except (ValueError, IndexError):
            continue
    return sorted(result, key=lambda c: int(c[0]))

def klines(symbol, tf="15m", start=None, limit=100, http=None):
    """Return contiguous available *published* 15m bars; no REST fallback.

    start=None: latest published history, up to limit.
    start supplied: from first 15m candle at/after start; stop on a missing day.
    """
    if tf != "15m" or not re.fullmatch(r"[A-Z0-9]{2,20}USDT", symbol):
        return []
    limit = max(1, min(int(limit), 1500))
    now = datetime.now(timezone.utc)
    # Never query today's ZIP, which cannot be complete.
    last_day = (now - timedelta(days=1)).date()
    own = http is None
    session = http or requests.Session()
    try:
        with LOCK:
            if start is None:
                days = [last_day - timedelta(days=i) for i in range(min(20, (limit + 95)//96 + 2))]
                output = []
                for day in days:
                    bars = _load_day(symbol, day, session)
                    if bars is None:
                        if output:
                            break
                        continue
                    output = bars + output
                    if len(output) >= limit:
                        break
                return output[-limit:]
            first = _day(int(start))
            if first > last_day:
                return []
            output = []
            day = first
            while day <= last_day and len(output) < limit and (day - first).days < 16:
                bars = _load_day(symbol, day, session)
                if bars is None:
                    break
                output.extend(c for c in bars if int(c[0]) >= int(start))
                day += timedelta(days=1)
            return output[:limit]
    finally:
        if own:
            session.close()
