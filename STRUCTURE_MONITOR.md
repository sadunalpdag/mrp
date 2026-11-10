# Structure Agent Test

`STRUCTURE_MONITOR_ENABLED=1` (default) enables an independent five-minute
rule-based observer in `ema.py`. It does not invoke an LLM, use broker methods,
change Crash candidate eligibility, or submit orders. `0` disables the observer
and its extra kline subscriptions.

The existing EMA Filter workbook receives one new worksheet, **Structure Agent
Test**. Other worksheets are not edited or deleted. The observer ranks the top
50 active USDT perpetual contracts by 24-hour USDT quote turnover in the
background, but the sheet displays ONLY up to 5 active **DÖNÜŞ** and 5 active
**DEVAM** candidates (up to 10 rows). No noncandidate or missing-data coin is
shown as an active watchlist item. Unavailable age or price data is not
interpreted as eligibility. Contract age must be at
least 30 days. Only five days of candles are required, not 30 days of candles.
BTC/ETH changes are shown as context, not imposed as a trading filter.

The observer merges the existing history-cache files with closed WebSocket
bars, preferring Binance to alternate-venue candles. All 480 expected slots
must exist before structural analysis. Complete UTC 4-hour bars are aggregated
from 16 contiguous real 15-minute bars. A pivot is known only after two bars
have closed to its right.

Two independent LONG research hypotheses are monitored:

- Reversal: negative 24h change, prior descending confirmed 4h swings, first
  body-close above the last confirmed lower high, first 15m contact within 16
  bars at +/-0.20 ATR14, and bullish close above the level on that contact.
- Continuation: rising confirmed 4h swings, positive 24h change, >=1 ATR bullish
  impulse with >=1.5 times preceding 20-bar volume, 1-4 bearish pullback bars
  holding the impulse low, then a bullish close above the pullback highs.

At most five current candidates per hypothesis are selected, prioritizing
fresh confirmations then turnover rank. The scan reassesses the same 50-coin
universe every five minutes; when a selected candidate expires, loses its
setup, becomes ineligible, or is rejected, the next eligible candidate of
that method takes its place without a separate API query. A bare upward/downward
4h trend is **YAPI YOK**, not **İZLE**. **İZLE** requires an actual in-progress
reversal breakout/retest or continuation impulse/pullback; a failed first
reversal touch does not occupy a slot indefinitely. If fewer than five
candidates meet the conditions in either method, those slots stay empty. Indicative entry is the observed
ticker price, never a claimed historical fill. Stop is below the relevant
pullback low by 0.20 ATR. The illustrative target is 1.5 times price risk.
Known overhead pivot resistance must leave room for that target. Spread and
order-book depth are explicitly **unmeasured**, so confirmations are never
marked execution-eligible. There is no paper-fill/P&L simulator in this version.

Confirmed observations are deduplicated by symbol/method/candle timestamp and
retained locally for 30 days (maximum 2,000 records); historical confirmations
and all 50 diagnostic rows remain in the local JSON but do not clutter the
active Google Sheets table. The existing single fixed-range Sheets write clears
old 50-coin rows and the older event list. Restarting does not backdate a missed first touch.
State is `DATA_DIR/structure_agent_monitor.json`.

The observer uses the existing cached/WS candles for all 50; the display
change adds no new market-data REST calls. Missing bars still propose only one
warmup every five minutes, with a rotating cursor and
at most three structure warmups outstanding in the existing history worker.
Due Crash jobs take precedence at submission. The existing archive checksum,
alternate-venue price validation, request pacing, IP cooldown and last-resort
REST budget remain enforced; no synthetic bars are inserted. A failed source
can leave a symbol waiting. Sheet/network failures remain in the observer's
daemon and do not stop the Crash loop. Logs distinguish `sheets_verified=1`
from local-only output and always include `execution=OFF`.

## Neden 50 coin taranırken Google Sheets boş görünebilir?

Bu izleyici 50 coin'i **24 saatlik USDT işlem hacmine** göre seçer; 50'ye kadar
pozisyon veya 50 hazır formasyon olduğu anlamına gelmez. Her coin için 30 gün
sözleşme yaşı, kesintisiz **480/480 gerçek 15 dakikalık kapanış**, yeterli 4 saatlik
swing yapısı ve son dönemde aktif geri-test/impuls koşulları aranır. 5 dakikalık
tarama aralığı, 15 dakikalık fiyat teyidi yerine geçmez.

**Structure Agent Test** sayfasında 5. satırda yeni bir özet bulunur:
veri bekleyen, 30 gün + 480 mum koşulunu karşılayan, 4 saatlik yapı
hesaplanabilen, başlıca elenme nedenleri ve halen aktif formasyon sayısı.
Boş aday alanları bir hata değildir; seçilmeye uygun formasyon yoksa boş
kalır. Ana tabloda hâlâ en çok 5 dönüş ve 5 devam adayı gösterilir; diğer
50 coinin tek tek araştırma verileri Sheet'e sinyal diye basılmaz.

Render logunda `[STRUCTURE MONITOR]` satırı artık `age_ready`,
`complete_480`, `eligible_history`, `four_ready`, `decisions` ve
`top_reasons` sayılarını gösterir. `[STRUCTURE HISTORY]` satırındaki
`accepted=0`, eksik mum düzeltme talebinin o döngüde öncelik/kuyruk
sebebiyle kabul edilmediğini gösterir; başarılı veri indirme anlamına gelmez.
Bu tanılama izleme/Sheets görünürlüğünü iyileştirir, gerçek Binance emirlerini
etkilemez, sinyal koşullarını gevşetmez ve sahte mum üretmez.
