import requests, os

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Gönderilecek dosya yolu (örnek)
FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "closed_trades.csv")

def send_file(file_path):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ BOT_TOKEN veya CHAT_ID tanımlı değil.")
        return

    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": f}
        data = {"chat_id": CHAT_ID, "caption": os.path.basename(file_path)}
        response = requests.post(url, data=data, files=files)
    
    if response.status_code == 200:
        print(f"✅ Dosya gönderildi: {file_path}")
    else:
        print(f"❌ Hata: {response.text}")

if __name__ == "__main__":
    send_file(FILE_PATH)
