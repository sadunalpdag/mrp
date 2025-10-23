# ==============================================================
#  send_file.py — EMA ULTRA yardımcı modül
#  Render veya local'de data klasöründeki dosyaları gönderir / indirir
#  - Çoklu dosya desteği
#  - Telegram’a gönderim
#  - ZIP yedekleme
# ==============================================================

import os
import zipfile
import requests
from datetime import datetime, timezone, timedelta

# ============ Telegram Ayarları ============
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# ============ Zaman Fonksiyonları ============
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)

def today_ist_date():
    return now_ist_dt().strftime("%Y-%m-%d")

# ============ Log Fonksiyonu ============
def log(msg):
    print(f"[{now_ist_dt()}] {msg}", flush=True)

# ============ Telegram Gönderim Fonksiyonları ============
def send_tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] environment değişkenleri eksik")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        log(f"[TG] Mesaj gönderildi: {text[:50]}")
    except Exception as e:
        log(f"[TG_ERR] {e}")

def send_tg_document(filename, content):
    if not BOT_TOKEN or not CHAT_ID:
        log("[TG] environment değişkenleri eksik")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        files = {'document': (filename, content)}
        data = {'chat_id': CHAT_ID, 'caption': filename}
        requests.post(url, data=data, files=files, timeout=60)
        log(f"[TG] Dosya gönderildi: {filename}")
    except Exception as e:
        log(f"[TG_DOC_ERR] {e}")

# ============ Çoklu Dosya Gönderimi ============
def send_multiple_files(base_dir):
    """data klasöründeki tüm JSON/CSV/PKL dosyalarını gönderir"""
    sent_count = 0
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith((".json", ".csv", ".pkl")):
                file_path = os.path.join(root, f)
                with open(file_path, "rb") as file_content:
                    send_tg_document(f, file_content)
                    sent_count += 1
    log(f"[TG] Toplam {sent_count} dosya gönderildi.")

# ============ ZIP Yedekleme ============
def create_backup_zip(base_dir):
    """data klasöründeki dosyaları ZIP olarak arşivler"""
    try:
        files_to_zip = []
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith((".json", ".csv", ".pkl")):
                    files_to_zip.append(os.path.join(root, f))
        if not files_to_zip:
            log("[ZIP] Eklenecek dosya bulunamadı.")
            return None

        zip_name = f"backup_{today_ist_date()}.zip"
        zip_path = os.path.join(base_dir, zip_name)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)
        log(f"[ZIP] {len(files_to_zip)} dosya eklendi → {zip_path}")
        return zip_path
    except Exception as e:
        log(f"[ZIP_ERR] {e}")
        return None

# ============ Ana Fonksiyon ============
def main():
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(base_dir):
        log(f"[ERR] data klasörü bulunamadı: {base_dir}")
        return

    send_tg(f"📦 EMA ULTRA Backup başlatıldı…")
    zip_path = create_backup_zip(base_dir)
    if zip_path and os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            send_tg_document(os.path.basename(zip_path), f)
    send_multiple_files(base_dir)
    send_tg(f"✅ Tüm dosyalar gönderildi ({today_ist_date()})")

# ============ Çalıştır ============
if __name__ == "__main__":
    main()
