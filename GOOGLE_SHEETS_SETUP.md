# Google Sheets Integration - Setup Guide

Bu rehber, MRP trading bot'unuzun Google Sheets entegrasyonunu nasıl kuracağınızı açıklar.

## Gereksinimler

- Google hesabı
- Google Cloud Platform projesi
- Service Account oluşturma yetkisi

## Kurulum Adımları

### 1. Google Cloud Console'da Proje Oluşturma

1. [Google Cloud Console](https://console.cloud.google.com/) adresine gidin
2. Yeni bir proje oluşturun veya mevcut bir projeyi seçin
3. Proje Dashboard'una gidin

### 2. Google Sheets API'yi Etkinleştirme

1. Sol menüden "APIs & Services" > "Library" seçin
2. "Google Sheets API" aratın
3. "Enable" butonuna tıklayın
4. Aynı şekilde "Google Drive API"yi de etkinleştirin

### 3. Service Account Oluşturma

1. Sol menüden "APIs & Services" > "Credentials" seçin
2. "Create Credentials" > "Service Account" seçin
3. Service account detaylarını doldurun:
   - Name: `mrp-sheets-integration`
   - Description: `MRP Trading Bot Google Sheets Integration`
4. "Create and Continue" butonuna tıklayın
5. Role seçimi yapın (isteğe bağlı): "Editor" veya "Owner"
6. "Done" butonuna tıklayın

### 4. Service Account Key Oluşturma

1. Oluşturulan service account'a tıklayın
2. "Keys" sekmesine gidin
3. "Add Key" > "Create New Key" seçin
4. JSON formatını seçin
5. İndirilen JSON dosyasını güvenli bir yere kaydedin
6. Bu dosyayı `credentials.json` olarak yeniden adlandırın
7. Dosyayı proje dizinine kopyalayın

### 5. Google Sheets Oluşturma

1. [Google Sheets](https://sheets.google.com/) adresine gidin
2. Yeni bir spreadsheet oluşturun
3. Spreadsheet'in URL'inden ID'yi kopyalayın:
   - URL formatı: `https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit`
   - `{SPREADSHEET_ID}` kısmını not edin

### 6. Erişim Verme

1. Google Sheets'teki spreadsheet'i açın
2. Sağ üstteki "Share" butonuna tıklayın
3. Service account'un email adresini ekleyin (credentials.json içindeki "client_email")
4. "Editor" yetkisi verin
5. "Send" ile onaylayın

## Environment Variables Ayarlama

Aşağıdaki environment variable'ları ayarlayın:

### Local Ortam İçin

`.env` dosyası oluşturun:

```bash
GOOGLE_CREDENTIALS_FILE=credentials.json
SPREADSHEET_ID=your-spreadsheet-id-here
DATA_DIR=./data
```

### Render.com İçin

1. Render Dashboard'a gidin
2. Service'inizi seçin
3. "Environment" sekmesine gidin
4. Aşağıdaki değişkenleri ekleyin:

```
GOOGLE_CREDENTIALS_FILE=/data/credentials.json
SPREADSHEET_ID=your-spreadsheet-id-here
```

5. `credentials.json` dosyasını `/data` dizinine yükleyin

## Kullanım

### Manuel Export

```bash
python sheets_integration.py
```

### Programatik Kullanım

```python
from sheets_integration import export_all_data

# Tüm verileri export et
export_all_data()
```

### Otomatik Periyodik Export

`ema.py` veya ana bot dosyanıza ekleyebilirsiniz:

```python
from sheets_integration import export_all_data
import threading
import time

def periodic_sheets_export():
    """Her 1 saatte bir Google Sheets'e export yap"""
    while True:
        try:
            export_all_data()
        except Exception as e:
            print(f"Sheets export error: {e}")
        time.sleep(3600)  # 1 saat

# Background thread olarak başlat
sheets_thread = threading.Thread(target=periodic_sheets_export, daemon=True)
sheets_thread.start()
```

## Export Edilen Veriler

Integration aşağıdaki sayfaları oluşturur:

1. **Trades** - Tüm işlem detayları
   - Symbol, direction, power, exit reason, gain %, duration vb.

2. **Performance_Summary** - Performans analizi
   - Power band bazında trade count, avg gain, avg duration

3. **TP_Rate_Analysis** - Take Profit oranı analizi
   - Power band bazında TP/SL oranları

4. **Balance_History** - Bakiye geçmişi (varsa)
   - Zaman serisi balance verileri

## Güvenlik Notları

⚠️ **ÖNEMLİ:**
- `credentials.json` dosyasını ASLA Git'e commit etmeyin
- `.gitignore` dosyasına `credentials.json` eklenmiştir
- Service account key'leri düzenli olarak rotate edin
- Minimum gerekli izinleri verin

## Sorun Giderme

### "credentials.json not found" hatası
- Dosya yolunun doğru olduğundan emin olun
- `GOOGLE_CREDENTIALS_FILE` environment variable'ının doğru set edildiğini kontrol edin

### "Permission denied" hatası
- Service account email'in spreadsheet'e erişim yetkisi olduğunu kontrol edin
- Google Sheets API'nin etkin olduğunu doğrulayın

### Import hatası
- Gerekli kütüphanelerin yüklü olduğunu kontrol edin: `pip install -r requirements.txt`

## İletişim

Sorularınız için issue açabilirsiniz.
