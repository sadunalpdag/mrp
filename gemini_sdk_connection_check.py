"""Isolated Gemini SDK smoke test. No trading actions or market data access."""
import os
import sys

def main():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_SMOKE FAIL reason=API_KEY_MISSING")
        return 2
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("GEMINI_SMOKE FAIL reason=GOOGLE_GENAI_NOT_INSTALLED")
        return 2
    model = os.getenv("GEMINI_SMOKE_MODEL", "gemini-2.5-flash")
    try:
        with genai.Client(api_key=key) as client:
            response = client.models.generate_content(
                model=model,
                contents="Türkçe tek cümleyle cevap ver: BTC analiz servisi bağlantı testi başarılı mı? Bu yalnızca bağlantı testidir, piyasa verisi yok.",
                config=types.GenerateContentConfig(temperature=0),
            )
        answer = (response.text or "").strip()
        if not answer:
            print(f"GEMINI_SMOKE FAIL model={model} reason=EMPTY_RESPONSE")
            return 1
        usage = getattr(response, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", None)
        print(f"GEMINI_SMOKE OK model={model} response_chars={len(answer)} total_tokens={tokens}")
        return 0
    except Exception as exc:
        # Never print exception messages: they may include request URLs or secrets.
        print(f"GEMINI_SMOKE FAIL model={model} error_type={type(exc).__name__}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
