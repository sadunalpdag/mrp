# EMA Filter - Bot Status kontrol paneli

Mevcut EMA Filter dosyasindaki **Bot Status** sayfasinda A:B otomatik durum
raporu olarak kalir. D:I araligi kullanici ayar panelidir. Sinyal sayfasina
mudahale edilmez.

- D: Ayar adi
- E: Calisan bottaki gercek deger (salt okunur)
- F: Yeni istenen deger
- G: Aciklama ve aralik
- H: Onay/hata sonucu
- I: Tek seferlik onay komutu

F sutununa yeni degeri, AYNI SATIRDAKI I sutununa gerekli onayi yazin.
Bot yaklasik 5 dakikada bir kontrol eder. Basariliysa F ve I bosalir;
hataliysa I bosalir, F duzeltme icin kalir. Sonuc H sutununda gorunur.
Ilk kurulumda eski onaylar guvenlik icin temizlenir.

## Kontroller

| Ayar | Deger | I sutununda onay |
| --- | --- | --- |
| Gercek otomatik alis | AÇIK / KAPALI | Acarken CANLI ONAY; kapatirken UYGULA |
| Sanal gunluk kar takibi | AÇIK / KAPALI | UYGULA |
| Gunluk kar hedefi (%) | 0.5-20; ornek 7 veya 7,5 | UYGULA; ertesi Paris gunu |
| Gunluk giris limiti | 1-50 tam sayi | Artista RISK ONAY, azalista UYGULA |
| Pozisyon USDT | 1-1000 islem degeri, teminat degil | Artista RISK ONAY, azalista UYGULA |
| Eszamanli sembol | 1-50 | Artista RISK ONAY, azalista UYGULA |
| Acik pozisyon limiti | 1-50 | Artista RISK ONAY, azalista UYGULA |
| Bekleyen alis limiti | 1-50 | Artista RISK ONAY, azalista UYGULA |

Bunlar yalnizca **yeni alislar** icin kullanilir: var olan pozisyonlar
ve TP'ler kapanmaz, gonderilmis emirler degismez. Parametreler mevcut
NightExecutor kontrol kuyrugu uzerinden dogrulanip /data altinda saklanir.
Binance sinirlari ve TP denetimi aynen devam eder.

Gercek alislari acmak icin I sutunundaki CANLI ONAY'a ek olarak Render
CRASH_SHEETS_ALLOW_LIVE_ON=1 ayari gereklidir. Bu ayar varsayilan olarak
kapalidir; tek basina botu acmaz. Kapatmak bu ek izne bagli degildir.

Sanal gunluk kar % hedefi gercek pozisyon kapatmaz, TP koymaz, emir
silmez. Gercek kar-kilidi kapanis islemi henüz uygulanmamistir.

Read-only satirlar: bugunku kullanilan sayac (sifirlayarak limit atlamayi
engellemek icin kilitli), 10x kaldirac, manuel TP acilis ayari, HBREAK,
CoinBeacon API2 ve gercek gunluk kar kapanisi.

## Guvenlik

**Sheets F/I hucrelerini duzenleme izni olan herkes ayar talebi
gonderebilir.** Google Sheets API hangi kullanicinin hucreyi degistirdigini
güvenilir bicimde bu akista ayirt etmez. Bu dosyayi sadece guvendiginiz
kisilere duzenleme izniyle paylasin; gereken yerlerde Render canli
acilis iznini kapali tutun.

Onaysiz degisiklik uygulanmaz. Islem sonuclanamadiginda otomatik tekrar
denenmez; E/H sutunlarinda dogrulayip yeniden onaylamaniz gerekir.
Dashboard okuma/yazma sorunu Binance emir takibini durdurmaz.
