"""Non-blocking daily Crash research exports to a Google Drive folder.

Uses a separate persistent outbox and background worker. No trading/strategy hooks.
OAuth refresh token is recommended for personal My Drive; service accounts should
target a pre-shared folder in a Shared Drive (service accounts have no My Drive quota).
"""
import io
import json
import os
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DRIVE_API = "https://www.googleapis.com/drive/v3/files"
UPLOAD_API = "https://www.googleapis.com/upload/drive/v3/files"
SCOPE = "https://www.googleapis.com/auth/drive.file"


class DailyDriveBackup:
    def __init__(self, directory, notify, log=print, env=None, clock=time.time):
        self.env = os.environ if env is None else env
        self.enabled = self.env.get("CRASH_DRIVE_BACKUP_ENABLED", "0") == "1"
        self.root = Path(directory)
        self.notify, self.log, self.clock = notify, log, clock
        self.lock = threading.RLock()
        self.path = self.root / "drive_backup_outbox.json"
        self.tasks = {}
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("tasks"), dict):
                    raise ValueError("DRIVE_OUTBOX_INVALID")
                self.tasks = data["tasks"]

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"schema": 1, "tasks": self.tasks}, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def enqueue(self, zip_path, period_end):
        """Persist a job before Telegram/report success clears the pending ZIP."""
        if not self.enabled:
            return False
        key = Path(zip_path).name
        day = datetime.fromtimestamp(period_end, ZoneInfo("Europe/Paris"))
        with self.lock:
            if key in self.tasks:
                return False
            self.tasks[key] = {
                "path": str(zip_path), "period_end": period_end,
                "month": day.strftime("%Y-%m"),
                "stem": day.strftime("%Y-%m-%d_%H%M%S"),
                "zip_id": None, "csv_id": None, "done": False,
                "attempts": 0, "next_attempt": 0, "notified_error": False,
            }
            self._save()
        return True

    def _session(self):
        from google.auth.transport.requests import AuthorizedSession
        oauth = all(self.env.get(k, "").strip() for k in
                    ("GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET",
                     "GOOGLE_DRIVE_REFRESH_TOKEN"))
        if oauth:
            from google.oauth2.credentials import Credentials
            credentials = Credentials(
                token=None,
                refresh_token=self.env["GOOGLE_DRIVE_REFRESH_TOKEN"].strip(),
                client_id=self.env["GOOGLE_DRIVE_CLIENT_ID"].strip(),
                client_secret=self.env["GOOGLE_DRIVE_CLIENT_SECRET"].strip(),
                token_uri="https://oauth2.googleapis.com/token", scopes=[SCOPE])
            return AuthorizedSession(credentials)
        raw = (self.env.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
               or self.env.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip())
        if not raw:
            raise ValueError("DRIVE_CREDENTIALS_NOT_SET")
        from google.oauth2.service_account import Credentials
        try:
            info = json.loads(raw)
            credentials = Credentials.from_service_account_info(info, scopes=[SCOPE])
        except json.JSONDecodeError:
            credentials = Credentials.from_service_account_file(raw, scopes=[SCOPE])
        if not self.env.get("GOOGLE_DRIVE_PARENT_FOLDER_ID", "").strip():
            raise ValueError("DRIVE_SHARED_PARENT_FOLDER_REQUIRED_FOR_SERVICE_ACCOUNT")
        return AuthorizedSession(credentials)

    @staticmethod
    def _request(session, method, url, **kwargs):
        response = session.request(method, url, timeout=45, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    def _find(self, session, parent, name, mime=None):
        q = ("name = '" + name.replace("\\", "\\\\").replace("'", "\\'")
             + "' and '" + parent.replace("'", "") + "' in parents and trashed = false")
        if mime:
            q += " and mimeType = '" + mime + "'"
        response = self._request(session, "GET", DRIVE_API, params={
            "q": q, "fields": "nextPageToken,files(id,name,mimeType)",
            "pageSize": 100, "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true"})
        return next((f["id"] for f in response.get("files", []) if f.get("name") == name), None)

    def _folder(self, session, parent, name):
        folder_mime = "application/vnd.google-apps.folder"
        found = self._find(session, parent, name, folder_mime)
        if found:
            return found
        created = self._request(session, "POST", DRIVE_API,
            params={"supportsAllDrives": "true", "fields": "id"},
            json={"name": name, "mimeType": folder_mime, "parents": [parent]})
        return created["id"]

    def _upload(self, session, folder, name, data, mime):
        found = self._find(session, folder, name)
        if found:
            self._request(session, "PATCH", UPLOAD_API + "/" + found,
                params={"uploadType": "media", "supportsAllDrives": "true", "fields": "id"},
                headers={"Content-Type": mime}, data=data)
            return found
        boundary = "backup-" + uuid.uuid4().hex
        metadata = json.dumps({"name": name, "parents": [folder]}).encode("utf-8")
        body = (("--" + boundary + "\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n").encode()
                + metadata + ("\r\n--" + boundary + "\r\nContent-Type: " + mime
                + "\r\n\r\n").encode() + data
                + ("\r\n--" + boundary + "--\r\n").encode())
        reply = self._request(session, "POST", UPLOAD_API,
            params={"uploadType": "multipart", "supportsAllDrives": "true", "fields": "id"},
            headers={"Content-Type": "multipart/related; boundary=" + boundary}, data=body)
        return reply["id"]

    def process_once(self):
        """One bounded background attempt. Errors never interrupt signal processing."""
        if not self.enabled:
            return False
        now = self.clock()
        with self.lock:
            pending = [(k, dict(t)) for k, t in self.tasks.items()
                       if not t.get("done") and t.get("next_attempt", 0) <= now]
        if not pending:
            return False
        key, item = min(pending, key=lambda pair: pair[1]["period_end"])
        try:
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError("LOCAL_DAILY_ZIP_MISSING")
            parent = self.env.get("GOOGLE_DRIVE_PARENT_FOLDER_ID", "").strip() or "root"
            session = self._session()
            root_folder = self._folder(session, parent, "Crash_Bounce")
            month_folder = self._folder(session, root_folder, item["month"])
            stem = item["stem"]
            if not item.get("zip_id"):
                zip_id = self._upload(session, month_folder, "crash_" + stem + ".zip",
                                      path.read_bytes(), "application/zip")
                with self.lock:
                    self.tasks[key]["zip_id"] = zip_id
                    self._save()
            if not item.get("csv_id"):
                with zipfile.ZipFile(path) as archive:
                    csv_bytes = archive.read("failure_shadow.csv")
                csv_id = self._upload(session, month_folder, "risk_" + stem + ".csv",
                                      csv_bytes, "text/csv")
                with self.lock:
                    self.tasks[key]["csv_id"] = csv_id
                    self._save()
            with self.lock:
                self.tasks[key].update(done=True, completed_at=self.clock())
                # Retain completed history for idempotency across restarts.
                self._save()
            self.log("[CRASH DRIVE] ZIP and risk CSV backed up: " + stem)
            self.notify("✅ CRASH DRIVE YEDEK | " + stem + "\nZIP ve risk CSV yüklendi.")
        except Exception as error:
            name = type(error).__name__
            # No tokens, headers or credential values in logs or notifications.
            with self.lock:
                t = self.tasks[key]
                t["attempts"] += 1
                t["next_attempt"] = now + min(21600, 300 * (2 ** min(t["attempts"]-1, 6)))
                first = not t.get("notified_error")
                t["notified_error"] = True
                t["last_error"] = name if name != "ValueError" else str(error)[:90]
                self._save()
            self.log("[CRASH DRIVE] backup retry queued: " + name)
            if first:
                self.notify("⚠️ CRASH DRIVE YEDEK HATASI | " + item["stem"]
                            + "\nYeniden denenecek. Neden: " + name)
        return True

    def start(self):
        if not self.enabled:
            self.log("[CRASH DRIVE] OFF; enable after credentials are configured")
            return
        def run():
            while True:
                try:
                    self.process_once()
                except Exception as error:
                    self.log("[CRASH DRIVE WORKER ERR] " + type(error).__name__)
                time.sleep(60)
        threading.Thread(target=run, daemon=True, name="crash-drive-backup").start()
