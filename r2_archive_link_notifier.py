"""Read-only daily Telegram links for private R2 archives.

Uses existing Render R2 S3 and Telegram variables. No trading changes.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

LOG = logging.getLogger("r2_archive_link_notifier")
PARIS = ZoneInfo("Europe/Paris")
_STARTED = False
_LOCK = threading.Lock()


def _send_telegram(text):
    import requests
    token = (os.getenv("BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("CHAT_ID") or "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_NOT_CONFIGURED")
    r = requests.post(
        "https://api.telegram.org/bot" + token + "/sendMessage",
        data={"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"},
        timeout=20,
    )
    r.raise_for_status()
    if not r.json().get("ok"):
        raise RuntimeError("TELEGRAM_SEND_NOT_OK")


def _client():
    from crash_report_storage import R2_ENV_NAMES, r2_configured
    if not r2_configured():
        raise RuntimeError("R2_NOT_CONFIGURED")
    import boto3
    values = [os.environ[n].strip() for n in R2_ENV_NAMES]
    account, bucket, access, secret = values
    client = boto3.client(
        "s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=access, aws_secret_access_key=secret,
        region_name="auto",
    )
    return client, bucket


def _link(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        error = getattr(exc, "response", {}).get("Error", {})
        if error.get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key},
        ExpiresIn=7 * 86400,
    )


def _state_path():
    base = Path(os.getenv("DATA_DIR") or (Path(__file__).parent / "data"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "r2_archive_link_notification.json"


def _notify_once(now):
    current_day = now.date().isoformat()
    state_path = _state_path()
    try:
        last = json.loads(state_path.read_text(encoding="utf-8"))
        if last.get("sent_for") == current_day:
            return
    except (FileNotFoundError, ValueError, OSError):
        pass

    # Daily files are grouped by Europe/Paris dates.
    previous = now.date() - timedelta(days=1)
    stamp = previous.strftime("%Y/%m/%d")
    date = previous.isoformat()
    client, bucket = _client()
    files = [
        ("CoinBeacon JSON", f"coinbeacon-api2/{stamp}/coinbeacon_six_{date}.json"),
        ("HBREAK ZIP", f"hbreak/{stamp}/hbreak_15m_long_{date}.zip"),
    ]
    sections = [f"📦 R2 GÜNLÜK ARŞİV LİNKLERİ | {date}", "ChatGPT analiz ajanına iletebilirsin.", ""]
    available = 0
    for label, key in files:
        link = _link(client, bucket, key)
        if link:
            available += 1
            sections.extend([label + ":", link, ""])
        else:
            sections.extend([label + ": arşiv dosyası henüz bulunamadı.", ""])
    sections.append("Bağlantılar 7 gün geçerlidir. Gizli bağlantıları başkalarıyla paylaşma.")
    if available == 0:
        LOG.warning("[R2 LINK] no previous-day files found for %s; retrying later", date)
        return
    _send_telegram("\n".join(sections))
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sent_for": current_day, "archive_date": date}), encoding="utf-8")
    tmp.replace(state_path)
    LOG.info("[R2 LINK] Telegram sent for %s; available=%d/2", date, available)


def _run():
    # First run at/after 21:30 Paris local time, including DST changes.
    while True:
        try:
            now = datetime.now(PARIS)
            if (now.hour, now.minute) >= (21, 30):
                _notify_once(now)
        except Exception as exc:
            LOG.warning("[R2 LINK] %s: %s", type(exc).__name__, str(exc)[:120])
        time.sleep(300)


def start():
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        threading.Thread(target=_run, name="r2-archive-links", daemon=True).start()
