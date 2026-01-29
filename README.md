# mrp
pyton ile mrp yazımı

## Google Sheets Entegrasyonu

Trading verilerini Google Sheets'e export edebilirsiniz.

### Kurulum
Detaylı kurulum talimatları için [GOOGLE_SHEETS_SETUP.md](GOOGLE_SHEETS_SETUP.md) dosyasına bakın.

### Hızlı Başlangıç
```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Environment variables ayarla
export SPREADSHEET_ID=your-spreadsheet-id
export GOOGLE_CREDENTIALS_FILE=credentials.json

# Verileri export et
python sheets_integration.py
```

### Özellikler
- ✅ Trading history export
- ✅ Performance analysis export
- ✅ Balance history export
- ✅ Otomatik güncelleme desteği
