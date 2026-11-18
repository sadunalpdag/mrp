"""One-time read-only Gemini API model-access diagnostic for the BTC observer.

Never prints or persists the API key, prompt, response text, or arbitrary HTTP
body. Tests only one tiny prompt against a Google-documented model. Does NOT
change which model BTC/CB2 use. Does NOT access any broker or order endpoint.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://generativelanguage.googleapis.com/v1beta"
CANDIDATES = ("gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-flash-lite")
_LOCK = threading.Lock()


def _reason(response):
    try:
        status = (response.json().get("error") or {}).get("status")
        if isinstance(status, str) and status.isupper() and len(status) <= 48:
            return status
    except (ValueError, AttributeError, TypeError):
        pass
    return "UNKNOWN"


def _path():
    folder = Path(os.getenv("DATA_DIR", "/data"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "btc_gemini_access_probe_v1.json"


def probe_once():
    """List available generateContent models, then smoke-test ONE candidate."""
    with _LOCK:
        path = _path()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                if cached.get("date_utc") == today and cached.get("probe_version") == 1:
                    print("[BTC GEMINI PROBE] cached_today=1 result=" +
                          str(cached.get("probe_status", "UNKNOWN")), flush=True)
                    return cached
            except (OSError, ValueError, TypeError):
                pass
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            print("[BTC GEMINI PROBE] status=KEY_MISSING", flush=True)
            return {"probe_status": "KEY_MISSING"}
        result = {"probe_version": 1, "date_utc": today,
                  "probe_status": "INCOMPLETE", "models_status": None,
                  "candidate": None, "candidate_status": None}
        try:
            response = requests.get(API + "/models",
                                    headers={"x-goog-api-key": key},
                                    params={"pageSize": 100}, timeout=15)
            result["models_status"] = response.status_code
            if response.status_code != 200:
                result["models_error"] = _reason(response)
                result["probe_status"] = "LIST_FAILED"
                print(f"[BTC GEMINI PROBE] models_http={response.status_code} "
                      f"reason={result['models_error']}", flush=True)
                return result
            names = set()
            pages = 0
            while True:
                payload = response.json()
                pages += 1
                for m in payload.get("models") or []:
                    methods = m.get("supportedGenerationMethods") or []
                    name = m.get("name") or ""
                    if "generateContent" in methods and name.startswith("models/"):
                        names.add(name.removeprefix("models/"))
                token = payload.get("nextPageToken")
                if not token or pages >= 3:
                    break
                response = requests.get(API + "/models",
                                        headers={"x-goog-api-key": key},
                                        params={"pageSize": 100, "pageToken": token},
                                        timeout=15)
                if response.status_code != 200:
                    result["pagination_error"] = response.status_code
                    break
            available = [x for x in CANDIDATES if x in names]
            result["supported_count"] = len(names)
            result["recommended_available"] = available
            print(f"[BTC GEMINI PROBE] models_http=200 generate_content_models={len(names)} "
                  f"recommended={','.join(available) or 'NONE'}", flush=True)
            if not available:
                result["probe_status"] = "NO_SUPPORTED_CANDIDATE"
                return result
            model = available[0]
            result["candidate"] = model
            smoke = requests.post(
                API + "/models/" + model + ":generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [{"text": "Write OK"}]}],
                      "generationConfig": {"maxOutputTokens": 32, "temperature": 0}},
                timeout=25)
            result["candidate_status"] = smoke.status_code
            result["probe_status"] = ("SUCCESS" if smoke.status_code == 200 else "FAILED")
            if smoke.status_code != 200:
                result["candidate_error"] = _reason(smoke)
            print(f"[BTC GEMINI PROBE] model={model} generate_http={smoke.status_code} "
                  f"status={result['probe_status']} "
                  f"reason={result.get('candidate_error', 'NONE')}", flush=True)
            if smoke.status_code == 200:
                # Only token counts; NEVER log AI-generated text or API key.
                usage = (smoke.json().get("usageMetadata") or {})
                result["prompt_tokens"] = usage.get("promptTokenCount")
                result["output_tokens"] = usage.get("candidatesTokenCount")
                print(f"[BTC GEMINI PROBE] used_input_tokens={result['prompt_tokens']} "
                      f"used_output_tokens={result['output_tokens']}", flush=True)
            return result
        except (requests.RequestException, ValueError, TypeError) as exc:
            result["probe_status"] = "ERROR_" + type(exc).__name__
            print("[BTC GEMINI PROBE] exception=" + type(exc).__name__, flush=True)
            return result
        finally:
            try:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(result))
                tmp.replace(path)
            except OSError:
                print("[BTC GEMINI PROBE] local_record_unavailable", flush=True)
