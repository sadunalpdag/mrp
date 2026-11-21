"""Durable, paper-only AI Learning controls. Independent of Crash order settings."""
import json
import os
import threading
from pathlib import Path

_LOCK = threading.RLock()
DEFAULTS = {
    "enabled": False, "daily_call_cap": 250, "max_active": 30,
    "track_hours": 8, "check_minutes": 15, "analysis_phase": "both",
}
RANGES = {"daily_call_cap": (1, 250), "max_active": (1, 100),
          "track_hours": (1, 24)}
ALLOWED_PHASES = ("both", "initial")


def _path():
    return Path(os.getenv("DATA_DIR", "/data")) / "coinbeacon_ai_settings.json"


def _validate(key, value):
    if key not in DEFAULTS:
        raise ValueError("UNKNOWN_AI_SETTING")
    if key == "enabled":
        if type(value) is not bool:
            raise ValueError("AI_ON_OFF_REQUIRED")
    elif key == "analysis_phase":
        if value not in ALLOWED_PHASES:
            raise ValueError("INVALID_ANALYSIS_PHASE")
    elif key == "check_minutes":
        if type(value) is not int or value not in (15, 30, 60):
            raise ValueError("AI_INTERVAL_MUST_BE_15_30_60")
    else:
        if type(value) is not int:
            raise ValueError("AI_SETTING_MUST_BE_INTEGER")
        low, high = RANGES[key]
        if not low <= value <= high:
            raise ValueError("AI_SETTING_OUT_OF_RANGE")
    return value


def load_settings():
    """Read disk afresh to allow Dashboard edits without a Render redeploy."""
    values = dict(DEFAULTS)
    values["enabled"] = os.getenv("CB_AI_LEARNING_ENABLED", "0") == "1"
    for field, env in (("daily_call_cap", "CB_AI_DAILY_GEMINI_CALL_CAP"),
                       ("max_active", "CB_AI_MAX_ACTIVE")):
        try:
            values[field] = _validate(field, int(os.getenv(env, str(values[field]))))
        except ValueError:
            pass
    with _LOCK:
        try:
            data = json.loads(_path().read_text(encoding="utf8"))
        except FileNotFoundError:
            return values
        except (OSError, ValueError):
            raise ValueError("AI_SETTINGS_CORRUPT") from None
        if not isinstance(data, dict):
            raise ValueError("AI_SETTINGS_CORRUPT")
        for key, value in data.items():
            values[key] = _validate(key, value)
        # Explicit operator environment cap overrides an older persisted dashboard value.
        if os.getenv("CB_AI_DAILY_GEMINI_CALL_CAP"):
            values["daily_call_cap"] = _validate("daily_call_cap", int(os.environ["CB_AI_DAILY_GEMINI_CALL_CAP"]))
        return values


def update_setting(key, value):
    """Atomic, validated update to persistent disk. No secrets or trading flags."""
    _validate(key, value)
    with _LOCK:
        config = load_settings()
        config[key] = value
        filename = _path()
        filename.parent.mkdir(parents=True, exist_ok=True)
        temp = filename.with_suffix(".tmp")
        try:
            with temp.open("w", encoding="utf8") as handle:
                json.dump(config, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, filename)
        finally:
            temp.unlink(missing_ok=True)
        return config
