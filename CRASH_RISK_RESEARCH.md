# Crash Bounce: 15 günlük gözlemsel risk araştırması

Sinyal seçimi, V2 doğrulaması, güç puanı, giriş ve mevcut TP kuralları değişmez.
`ema.py` yalnız yeni WS `OPEN` geçişlerini bağımsız `crash_risk_research.py`
modülüne yollar. Emir veya yeni Telegram sinyali gönderilmez. Araştırma hatası
sinyali engellemez. Daha önce açılmış işlemlere sonradan spread uydurulmaz.

## Deney

İlk çalışmadan itibaren 15 gün yeni sanal giriş alınır. Her giriş en fazla
5 gün izlenir; son günün girişleri için nihai değerlendirme 20. güne uzar.
15. gün raporunda açık işlemler ayrıca değerlendirilir, TP sayılmaz.
Her senaryo aynı gözlenen giriş fiyatını referans alır; mevcut botun teorik
giriş seviyesine göre hesaplanan TP oranıyla birebir aynı değildir.

144 senaryo: TP %1 / %1,2 / %1,5; SL yok / %2 / %3 / %5;
asgari güç puanı filtresiz / 30 / 50 / 70; likidite filtresiz / spread / likidite.
Spread filtresi <=5 baz puan (%0,05). Likidite ayrıca sinyal anındaki 24 saatlik
hacim >=10 milyon USDT ve her iki en iyi fiyat seviyesinde >=1000 USDT ister.
Bunlar test varsayımlarıdır, optimize edilmiş eşikler değildir.
Eksik veriyle filtreyi geçmek yerine `missing_filter_data` sayılır.

Her giriş 50 USDT, referans sermaye 1000 USDT. Sanal slot sayısı sınırsızdır;
eşzamanlı pozisyon ihtiyacını ölçmek için 5 slot sınırı uygulanmaz.
Kaldıraç, likidasyon ve funding modellenmez. Hesap getirisi iddiası yoktur.
Giriş alış kotasyonu, çıkış satış kotasyonu üzerinden modellenir; her yöne
2 baz puan ilave kayma ve 5 baz puan komisyon varsayılır. Bunlar gerçek hesap
ücretleri değildir. En fazla 5 saniyelik, işlem zamanından sonra olmayan
kotasyon kullanılır. Eksik kotasyonda işlem fiyatı vekili kullanılır ve
`entry_quote_missing` / `unknown_cost_exits` ile açıkça işaretlenir.
Tam emir defteri etkisi, gerçek dolum ve gerçek işlem maliyeti bilinmez.
Hedef/SL ilk gözlenen geçişte değerlendirilir; hedef fiyatından garantili dolum
varsayılmaz. 5 günlük süre sonrasında ilk yeni fiyatla TIMEOUT kapanışı yapılır;
veri gelmezse pozisyon eski fiyatla açık ve stale olarak kalır.

## Rapor

Mevcut günlük ZIP'e `risk_scenarios/summary.json` ve `state.json` eklenir.
Özet: TP/SL/TIMEOUT/açık sayıları, kapalı ve açık net sonuçlar, toplam net getiri,
referans sermayeye göre getiri, gözlenen maksimum düşüş, maksimum eşzamanlı
pozisyon, en kötü işlem, ortalama kapanmış net sonuç ve maliyet varsayımları.
Maksimum düşüş açık pozisyon değerlerini de içerir. Gerçek zamanlı örneklerde
ölçülür; bağlantı kesintilerindeki hareketler bilinmez.
`modeled_fill_drag_usdt`, işlem fiyatı vekiline kıyasla kotasyon ve kayma modelinin
etkisidir; doğrudan ölçülmüş spread maliyeti değildir.
`sl_then_target`: SL ile kapandıktan SONRA aynı senaryonun hedefini 5 gün içinde
gören işlem sayısı. TP'nin SL'den önce görülmesi bu sayıya eklenmez. Kesintilerde
kaçan hareketler nedeniyle bu sayı eksik kalabilir.
Puan eşikleri iç içedir; yüksek puan için toplam getiri yanında örnek sayısı,
ortalama net sonuç ve aynı dönemin kayıpları karşılaştırılmalıdır.

`DATA_DIR/crash_research/risk_v1` yeniden başlatmada yüklenir. Kalıcı disk şarttır;
bu klasörü kaybeden dağıtım yeni deney başlatır. Ham kotasyon örnekleri (en fazla
10 saniyede bir/sembol), giriş/çıkış ve bağlantı olayları günlük JSONL dosyalarında
20 gün saklanır. Özet ve pozisyon kayıtları silinmez. Ham JSONL, ZIP boyutunu
şişirmemek için günlük Telegram ZIP'ine dahil edilmez; giriş/çıkış kotasyonları
state içinde bulunur. Checkpoint 15 saniyede bir yazılır; ani kapanma bu aralıktaki
olayları kaybettirebilir. Restart/kopma/kuyruk taşması ve >30 saniye işlem boşluğu
`coverage_incomplete` olarak belirtilir. Bu işlemler kesin başarı kabul edilmemeli.

Araştırma WS bağlantıları bağımsızdır: `/public/ws` bookTicker,
`/market/ws` aggTrade. Ping/pong, 5–60 saniye kademeli yeniden bağlantı ve jitter
vardır. Hata mesajı, kapanma kodu, gecikme, eski olaylar, kuyruk ve bağlantı
sağlığı kaydedilir. Bekleyen güçlü V2 adayları girişten önce kotasyon için izlenir;
ilk abonelik yetişmezse eksik kotasyon açıkça işaretlenir.
Kaynak: https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Connect

History kuyruk hatalarında sembol, sinyal ID, aşama, zaman, hata sınıfı ve hata
mesajı yazılır. Aynı iş 300 saniye sonra yeniden denenir. Geçersiz timestamp
kuyruğa alınmaz; disk yazımı başarısızsa RAM kuyruğu bozulmaz. Ekran görüntüsündeki
ValueError'ın kesin kök nedeni mevcut logda yoktur; yeni log bunu ayırt eder.

Çevrimdışı test: `python -m unittest discover -p 'test_crash*.py' -q`.
15 günlük sonuçlar oluşmadan eşiklerin getiriyi artırdığı sonucuna varılmamalıdır.

## Five-day context v1 (2026-10-21)

`crash_v2.assess_entry` now attaches `five_day_context`, calculated from exactly
its existing 480 validated, consecutive, closed 15m candles. No additional API
requests are made. The original V2 acceptance, strength score, entry/TP logic,
order settings and 15-day enrollment / 5-day research follow-up are unchanged.

The context includes five-day return and drawdown from the window high, price
position within the window range, previous four-day return, last-day return,
five rolling 24h block returns, and last-day USDT quote volume divided by the
previous four days' daily average. Missing/invalid quote volume produces null,
never an estimated volume or an acceptance failure. These rolling blocks are
not exchange calendar days. Trend labels describe price history, not predictions.

`weak_green_and_falling_rsi` marks green-candle volume share <0.20 AND RSI
one-hour change <0, using the existing score metrics. The separate experimental
selling-pressure index is 100 * (1-green_share) * clip(-RSI_delta/10, 0, 1).
The scaling is a research hypothesis, not fitted or validated and not a loss
probability. It does not alter the existing strength score or filter signals.
Green-candle volume is not a direct measure of aggressive buyer order flow.

Telegram candidate messages include context, volume ratio and pressure. Candidate
decision logs and signals_and_outcomes preserve the diagnostics. Report schema 5
adds daily and rolling flagged/not_flagged/unknown outcome cohorts. Records with
no v1 context remain unknown; historical results are not relabeled. Cohorts show
all statuses and fraction of entries reaching TP so far, not a closed-only win
rate. This is descriptive tracking, not a full equity/SL backtest.

Validation: all 81 offline Crash Bounce tests passed. Replaying all 124 candidates
from crash_data_1792618617.zip preserved every original assessment field and
strength score. The entered subset reproduced 6 TP / 4 OPEN flagged and 18 TP /
2 OPEN not flagged. This same-sample result is not prospective evidence. Existing
stale EPIC observations are not repaired by this change.
