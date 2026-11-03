# Crash Bounce V2 — kalıcı AI araştırma arşivi

## Production storage

Render'daki aktif `mrp` Background Worker'ın **/data** yolunda 2 GB kalıcı diski vardır. `render.yaml` dosyası `DATA_DIR=/data` tanımlar. **Önemli:** Dashboard üzerinden kurulmuş servislerde blueprint ayarları tek başına güncel ortam değişkenini garanti etmez. Render Environment bölümünde gerçek `DATA_DIR=/data` değeri kontrol edilmeli. Veri dizini değiştirilmeden önce **eski dosyaların taşınması ve emir durumunun korunması** zorunludur; eksik `crash_night_state.json` ile botun işlem geçmişi sıfırdan başlatılmamalı.

Kod bu dizinlerin üzerine yazmaz veya deploy sırasında silmez:

| Yol | Amaç |
| --- | --- |
| `/data/crash_research/YYYY-MM-DD.jsonl` | Zaman damgalı V2 olayları: sinyal, tarama, geçmiş mum bilgisi, takip |
| `/data/crash_research/history_cache/` | Kline arşivleri ve boşluk tamamlama |
| `/data/crash_research/derivatives/` | Gölge türev/OI/CVD ölçümleri |
| `/data/ai_top_gainers/top_gainers_state.json` | AI Top Gainers analizleri ve sanal takip geçmişi |
| `/data/night_execution/crash_night_state.json` | **Kritik** gerçek emir ve güvenlik kilidi kayıtları |
| `/data/research_archive/runtime/YYYY-MM-DD.jsonl` | Yeni: UTC günlük Crash/AI çalışma logları; sadece ekleme |
| `/data/log.txt` | Mevcut genel uygulama log dosyası |

`research_archive.py`, `[CRASH ...]`, `[TOP GAINERS ...]`, `[BTC HOURLY ...]` ile başlayan seçili çalışma satırlarını UTC tarihe göre JSONL dosyalarına **append-only** kaydeder. Anahtarlar, imzalı bağlantılar ve tüm log türleri kopyalanmaz. Geçmiş gün dosyaları açılıp yeniden yazılmaz. `Render Logs` ekranının saklama süresi bu dosyaları etkilemez.

## Sınırlar ve harici yedek

Bu yöntem **aynı Render persistent disk üzerinde deploy/restart kaynaklı veri kaybını engellemeye yöneliktir**; disk silinirse, servis/disk taşınırsa, kapasite dolar veya veri bozulursa tek kopya korunmuş sayılmaz. Disk kapasitesi 2 GB olduğundan kullanımın düzenli izlenmesi ve bağımsız yedek şarttır.

Önerilen ikinci aşama: `/data/crash_research`, `/data/research_archive`, `/data/ai_top_gainers` ve gerekli işlem durumu dosyalarını Cloudflare R2 / S3 uyumlu şifreli ve sürümlü depoya günlük veya saatlik yedeklemek. Harici erişim anahtarları olmadan **bu cloud yedekleme etkin değildir**. GitHub'a API anahtarları veya Binance hesabı verileri push edilmemelidir.

Mevcut Render panelinde daha önce görünürlüğünü kaybetmiş kayıtlar otomatik geri getirilemez; yalnızca persistent disk, daha önce indirilmiş ZIP raporları ve varsa ayrı yedekler kurtarılabilir.

## Manuel Close ve güvenli toparlanma

`crash_night.py`, daha önce `PROTECTED` durumunda bulunan bir işlemde TP iptal/expire/reject veya kayıt yoksa tüm sembollerde şu kontrolü yapar:

1. Binance V3'te ilgili `symbol + positionSide` için **gerçek `0` miktarı** veya tam hesap sorgusunda sembolün bulunmaması doğrulanır. V3, açık pozisyonu ya da açık emri olmayan sembolleri yanıttan çıkardığı için kayıp satır ancak **başarılı tam hesap pozisyon sorgusu + normal ve algo açık emir listelerinin aynı sembolde boş olması** halinde kapalı kabul edilir. API hatası veya eksik/bozuk yanıt kesinlikle sıfır sayılmaz.
2. Aynı sembolde normal/algo açık emir bulunmadığını doğrular; belirsiz veya kısmi işlemde kilidi korur.
3. Pozisyon gerçekten kapalıysa eski izlemeyi `CLOSED_EXTERNALLY` terminal durumuna geçirir; bunu TP başarısı saymaz. Geçmiş günlük sayaç değişmez.
4. Mevcut `halt` **otomatik kaldırılmaz**. Yeni girişler başka coinlerde de engellenmeye devam eder.
5. Yetkili Telegram hesabından `/otomatik_kilit_kaldir` gönderilebilir. Bot tüm bot pozisyonları, bekleyen BUY emirleri, açık LONG ve geçerli TP kapsamını **salt-okunur** sorgularla kontrol eder. Hata, eksik TP, geçersiz durum veya SHORT varsa kilit kalır.
6. Kontroller başarılıysa yalnızca güvenlik kilidi kalkar; **otomatik girişler KAPALI kalır**. Kullanıcı ayrıca `/otomatik_ac` yazmalıdır. Komutun sırf görülmesi yeni emir oluşturmaz.

Bu değişiklik **SL eklemez**, mevcut Crash Bounce V2 sinyal/eşik/TP kurallarını değiştirmez ve Binance'e otomatik kapatma emri göndermez.
