# Crash Bounce günlük rapor depolama — Cloudflare R2 ve Telegram yedek

## Tercih edilen: özel R2 arşivi

Bot büyük günlük raporu eksiksiz bir ZIP olarak özel Cloudflare R2 bucket'ına yükler. Telegram'a önce günlük metin özeti gelir, sonra 7 gün geçerli özel bir indirme bağlantısı gelir. Bağlantı süresi dolsa da dosya R2 üzerinde kalır.

**Hesap ve bucket henüz yoksa bot bunları kendiliğinden oluşturamaz.**

### Bir defalık kurulum

1. Cloudflare Dashboard > R2 Object Storage > Create bucket. Örnek bucket adı: crash-bounce-archives. Erişim private kalmalı; public access açmayın.
2. R2 > Manage API Tokens > Create token. Object Read & Write izni yalnız bu bucket için verin. Access Key ID ve Secret Access Key'i güvenli saklayın.
3. Render > çalışan mrp background worker > Environment bölümüne ekleyin:

- CRASH_REPORT_R2_ACCOUNT_ID = Cloudflare Account ID (32 karakter)
- CRASH_REPORT_R2_BUCKET = crash-bounce-archives
- CRASH_REPORT_R2_ACCESS_KEY_ID = R2 Access Key ID
- CRASH_REPORT_R2_SECRET_ACCESS_KEY = R2 Secret Access Key
- CRASH_REPORT_R2_PREFIX = crash-bounce (isteğe bağlı)

4. Render deploy sonrası raporlar Paris saatiyle 21:00'de hazırlanır. R2 içinde dosyaların yolu: crash-bounce/YYYY/MM/DD/crash_data_EPOCH.zip

R2 anahtarlarını GitHub, Telegram veya sohbete göndermeyin. Yalnız Render'ın özel Environment alanlarını kullanın.

### Bulut kullanılamadığında Telegram yedeği

R2 yapılandırması yoksa veya yükleme başarısızsa büyük ZIP, en fazla 49.000.000 byte boyutlu bağımsız ZIP parçalarına ayrılır. Her ZIP açılabilir. İlk bölümde MULTIPART_README.txt bulunur. Tekil çok büyük kaynak dosyalar numaralı parçalara bölünür; README birleştirme sırasını anlatır. Hiçbir araştırma kaydı atılmaz.

Başarısız dosya parçaları tekrar denenir. Tüm parçalar gönderilmeden günlük rapor tamamlandı sayılmaz. Büyük aktarım ve dosya parçalama ayrı thread üzerinde yürür; Crash Bounce tarama, gerçek emir ve TP kontrolleri korunur.

### Güvenlik ve kapasite

- Telegram Bot API dosya gönderimi en fazla 50 MB; sistem 1 MB pay bırakır.
- Cloudflare R2 Standard ücretsiz kota aylık 10 GB-month depolama, sınırlı ücretsiz API işlemleri. Yaklaşık 100 MB/gün × 30 gün ≈ 3 GB depolama eder.
- R2 özel indirme bağlantısı 7 gün geçerli. Bağlantıyı gören kişi süre içinde dosyayı açabilir.
- Loglar: CRASH REPORT R2 uploaded, CRASH REPORT daily cloud link sent, CRASH REPORT PARTS READY, CRASH REPORT PART SENT.
- Yeni kod, önceden yalnız metin özeti ile tamamlandı sayılmış eski raporları kendiliğinden tekrar göndermez.
