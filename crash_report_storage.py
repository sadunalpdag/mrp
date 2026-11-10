"""Private Cloudflare R2 archive for large daily Crash Bounce reports.

No public bucket. Keep credentials in Render environment variables only.
Signed Telegram download links expire after seven days; the private object
remains in R2, organized by Paris calendar date, until explicitly deleted.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

R2_ENV_NAMES = (
    "CRASH_REPORT_R2_ACCOUNT_ID",
    "CRASH_REPORT_R2_BUCKET",
    "CRASH_REPORT_R2_ACCESS_KEY_ID",
    "CRASH_REPORT_R2_SECRET_ACCESS_KEY",
)
PARIS = ZoneInfo("Europe/Paris")
LINK_EXPIRY_SECONDS = 7 * 86400


def r2_configured(env=None):
    env = os.environ if env is None else env
    present = [bool(str(env.get(name) or "").strip()) for name in R2_ENV_NAMES]
    if any(present) and not all(present):
        raise ValueError("INCOMPLETE_R2_CONFIGURATION")
    return all(present)


def upload_private_report(path, period_end, env=None):
    """Upload full intact ZIP via multipart; verify size and sign private link.

    This is called only from a dedicated report-delivery thread. It performs
    no operations on Binance and never changes report or trading state.
    """
    env = os.environ if env is None else env
    if not r2_configured(env):
        raise ValueError("R2_NOT_CONFIGURED")
    account = str(env[R2_ENV_NAMES[0]]).strip()
    bucket = str(env[R2_ENV_NAMES[1]]).strip()
    access = str(env[R2_ENV_NAMES[2]]).strip()
    secret = str(env[R2_ENV_NAMES[3]]).strip()
    if not os.path.isfile(path) or not path.lower().endswith(".zip"):
        raise ValueError("REPORT_ZIP_NOT_FOUND")
    if not (account.isascii() and account.isalnum()
            and 16 <= len(account) <= 64):
        raise ValueError("INVALID_R2_ACCOUNT_ID")

    # Import lazily, so a missing optional integration does not affect
    # Crash Bounce orders, research, Telegram summaries or ZIP fallback.
    import boto3
    from boto3.s3.transfer import TransferConfig
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="auto",
    )
    stamp = datetime.fromtimestamp(float(period_end), PARIS).strftime("%Y/%m/%d")
    prefix = str(env.get("CRASH_REPORT_R2_PREFIX") or "crash-bounce").strip().strip("/")
    if not prefix or ".." in prefix or "\\" in prefix:
        raise ValueError("INVALID_R2_PREFIX")
    name = os.path.basename(path)
    key = f"{prefix}/{stamp}/{name}"
    client.upload_file(
        path, bucket, key,
        ExtraArgs={"ContentType": "application/zip"},
        Config=TransferConfig(multipart_threshold=8 * 1024 * 1024,
                              multipart_chunksize=8 * 1024 * 1024,
                              max_concurrency=1),
    )
    size = os.path.getsize(path)
    remote_size = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    if size != remote_size:
        raise ValueError("R2_UPLOAD_SIZE_MISMATCH")
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=LINK_EXPIRY_SECONDS,
    )
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("INVALID_R2_SIGNED_LINK")
    return {"key": key, "size_bytes": size, "url": url,
            "link_days": LINK_EXPIRY_SECONDS // 86400}
