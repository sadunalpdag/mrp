"""Prospective structure research. No broker, execution, LLM or trading-state access.

Uses closed, contiguous 15m candles, UTC-aligned complete 4h aggregates and
confirmed 2-left/2-right pivots. Only its own worksheet/state are written.
"""
import json
import math
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from crash_data_sources import valid_rows, source_counts

STEP = 900000
FOUR_H = 16 * STEP
TAB = "Structure Agent Test"
VERSION = "structure_monitor_v1"
HEADERS = ["Coin", "Hacim sırası", "24s hacim USDT", "24s değişim %",
           "15m veri", "Tam 4s mum", "4s yapı", "Yöntem", "Karar",
           "Seviye", "Gösterge giriş", "Sanal stop", "Sanal TP 1,5R",
           "Teyit zamanı Paris", "Eksik / beklenen", "Veri kaynakları",
           "Spread / derinlik", "Mod"]


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def load(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path + ".tmp", path)


def universe(tickers, market, onboard, now):
    """Top 50 by quote turnover FIRST; missing age never promotes rank 51."""
    ranked = {}
    for item in tickers:
        if not isinstance(item, dict) or item.get("symbol") not in market:
            continue
        symbol = item["symbol"]
        volume = number(item.get("quoteVolume"))
        price = number(item.get("lastPrice"))
        change = number(item.get("priceChangePercent"))
        if volume is not None and volume > 0 and price is not None and price > 0 and change is not None:
            ranked[symbol] = dict(item, quoteVolume=volume, lastPrice=price,
                                  priceChangePercent=change)
    top = sorted(ranked.values(), key=lambda t: (-t["quoteVolume"], t["symbol"]))[:50]
    for rank, item in enumerate(top, 1):
        item["rank"] = rank
        listed = number(onboard.get(item["symbol"]))
        item["age_ready"] = listed is not None and 0 < listed <= (now - 30*86400)*1000
        item["age_reason"] = ("Listeleme tarihi eksik" if listed is None else
                              "30 günlük sözleşme geçmişi bekleniyor")
    return top


def cached_rows(directory, symbol, ws_rows, now):
    end = int(now * 1000) // STEP * STEP
    start = end - 480 * STEP
    path = os.path.join(directory, "crash_research", "history_cache", symbol + ".json")
    disk = load(path, [])
    found = valid_rows(disk if isinstance(disk, list) else [], start, end)
    alternates = {"BYBIT_LINEAR", "COINBASE_SPOT", "OKX_SWAP"}
    for ts, row in valid_rows(ws_rows, start, end).items():
        old = found.get(ts)
        if old is None or (row[12] if len(row)>12 else "BINANCE") not in alternates or (
                old[12] if len(old)>12 else "BINANCE") in alternates:
            found[ts] = row
    return [found[t] for t in sorted(found)]


def aggregate(rows):
    """Never compress gaps into a 4h bar, or include an unfinished UTC block."""
    slots = {int(r[0]): r for r in rows}
    result = []
    for start in sorted({t // FOUR_H * FOUR_H for t in slots}):
        times = range(start, start + FOUR_H, STEP)
        if not all(t in slots for t in times):
            continue
        group = [slots[t] for t in times]
        result.append([start, float(group[0][1]), max(float(r[2]) for r in group),
                       min(float(r[3]) for r in group), float(group[-1][4]),
                       sum(float(r[5]) for r in group), start + FOUR_H - 1])
    return result


def pivots(bars, field, high):
    result = []
    for i in range(2, len(bars)-2):
        value = float(bars[i][field])
        neighbors = [float(bars[j][field]) for j in (i-2, i-1, i+1, i+2)]
        if (value > max(neighbors)) if high else (value < min(neighbors)):
            result.append((i, value))
    return result


def structure(bars):
    highs, lows = pivots(bars, 2, True), pivots(bars, 3, False)
    if len(highs) < 2 or len(lows) < 2:
        return "BELİRSİZ", highs, lows
    if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
        return "YÜKSELİŞ", highs, lows
    if highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
        return "DÜŞÜŞ", highs, lows
    return "YATAY / KARIŞIK", highs, lows


def atr(rows):
    return sum(max(float(r[2])-float(r[3]), abs(float(r[2])-float(p[4])),
                   abs(float(r[3])-float(p[4])))
               for p, r in zip(rows[-15:-1], rows[-14:])) / 14


def analyse(rows, now, change):
    end = int(now*1000)//STEP*STEP
    base = dict(count=len(rows), four_count=0, trend="BELİRSİZ", method="—",
                decision="VERİ BEKLENİYOR", reason="480 gerçek kapanmış 15m mum gerekli",
                sources=source_counts(rows))
    if len(rows) != 480 or [int(r[0]) for r in rows] != list(range(end-480*STEP, end, STEP)):
        return base
    bars = aggregate(rows)
    base["four_count"] = len(bars)
    # A gap-free sequence is mandatory even when all individual blocks exist.
    if len(bars) < 20 or any(int(b[0])-int(a[0]) != FOUR_H for a,b in zip(bars,bars[1:])):
        base["reason"] = "En az 20 ardışık tam 4s mum gerekli"
        return base
    trend, highs, lows = structure(bars)
    base.update(trend=trend, decision="İZLE", reason="Uygun yapı / ilk geri test bekleniyor",
                resistances=[value for _,value in highs])
    volatility = atr(rows)
    if volatility <= 0:
        base.update(decision="REDDET", reason="ATR sıfır")
        return base
    # Detect at the break's original time: right-side pivot bars must ALREADY
    # be closed before the break. Later pivots cannot alter earlier decisions.
    for i in range(len(bars)-1, max(5, len(bars)-3), -1):
        prior_trend, prior_highs, _ = structure(bars[:i])
        if prior_trend != "DÜŞÜŞ" or not prior_highs or change >= 0:
            continue
        level = prior_highs[-1][1]
        if float(bars[i-1][4]) > level or float(bars[i][4]) <= level:
            continue
        break_end = int(bars[i][0]) + FOUR_H
        after = [r for r in rows if break_end <= int(r[0]) < break_end+16*STEP]
        base.update(method="DÖNÜŞ", level=level, reason="İlk geri test bekleniyor")
        zone = .20 * volatility
        for pos, row in enumerate(after):
            if float(row[3]) <= level+zone and float(row[2]) >= level-zone:
                # Freeze first contact: later bounces are not relabeled first.
                if pos != len(after)-1 or int(row[0]) != end-STEP:
                    base["reason"] = "İlk temas geçmişte; yeni sinyal üretilmedi"
                    return base
                if float(row[4]) > level and float(row[4]) > float(row[1]):
                    stop = min(float(r[3]) for r in after[:pos+1]) - zone
                    base.update(decision="TEYİT VAR", stop=stop, confirmed=int(row[6]),
                                reason="İlk geri testte 15m alıcı teyidi; yalnız gözlem")
                else:
                    base["reason"] = "İlk temas var; kapanış teyidi yok"
                return base
        if end >= break_end+16*STEP:
            base.update(decision="SÜRESİ DOLDU", reason="16 mum içinde ilk geri test gelmedi")
        return base
    # Independent momentum experiment: high-timeframe uptrend, a >=1 ATR
    # impulse, 1-4 bearish pullback bars, latest close breaks pullback high.
    if trend == "YÜKSELİŞ" and change > 0:
        base.update(method="DEVAM", reason="Hacimli itki ve ilk geri çekilme bekleniyor")
        last = rows[-1]
        for size in range(1, 5):
            pull = rows[-size-1:-1]
            impulse = rows[-size-2]
            volume_mean = sum(float(r[5]) for r in rows[-size-22:-size-2])/20
            if (float(impulse[4])-float(impulse[1]) >= volatility
                    and float(impulse[5]) >= 1.5*volume_mean
                    and all(float(r[4]) < float(r[1]) for r in pull)
                    and min(float(r[3]) for r in pull) > float(impulse[3])
                    and float(last[4]) > max(float(r[2]) for r in pull)
                    and float(last[4]) > float(last[1])):
                base.update(decision="TEYİT VAR", level=max(float(r[2]) for r in pull),
                            stop=min(float(r[3]) for r in pull)-.20*volatility,
                            confirmed=int(last[6]), reason="İlk geri çekilme sonrası kapanış kırılımı; yalnız gözlem")
                break
    base["resistances"] = [value for _,value in highs]
    return base


def scan(directory, tickers, market, onboard, candles_get, now, previous):
    top = universe(tickers, market, onboard, now)
    diagnostics = []
    for ticker in top:
        symbol = ticker["symbol"]
        rows = cached_rows(directory, symbol, candles_get(symbol), now)
        result = analyse(rows, now, ticker["priceChangePercent"])
        result.update(symbol=symbol, rank=ticker["rank"], volume=ticker["quoteVolume"],
                      change=ticker["priceChangePercent"], age_ready=ticker["age_ready"])
        if not ticker["age_ready"]:
            result.update(decision="VERİ BEKLENİYOR", reason=ticker["age_reason"])
        if result["decision"] == "TEYİT VAR":
            # Snapshot price is indicative, never an asserted fill at candle close.
            entry, stop = ticker["lastPrice"], result["stop"]
            risk = entry-stop
            target = entry+1.5*risk
            overhead = [p for p in result.get("resistances", []) if p > entry]
            if risk <= 0 or (overhead and min(overhead) < target):
                result.update(decision="REDDET", reason="Stop geçersiz veya yakın direnç 1,5R alan bırakmıyor")
            else:
                result.update(entry=entry, target=target)
        result.pop("resistances", None)
        diagnostics.append(result)
    # Up to 5 per strategy. No artificial filling when no setup exists.
    selected = []
    for method in ("DÖNÜŞ", "DEVAM"):
        candidates = [r for r in diagnostics if r["method"] == method and r["age_ready"]
                      and r["decision"] in ("TEYİT VAR", "İZLE")]
        candidates.sort(key=lambda r: (r["decision"] != "TEYİT VAR", r["rank"]))
        selected.extend(r["symbol"] for r in candidates[:5])
    events = previous.get("events", []) if isinstance(previous, dict) else []
    events = [e for e in events if now-e.get("observed_at", 0) < 30*86400]
    known = {e["id"] for e in events}
    for r in diagnostics:
        key = f'{r["symbol"]}:{r["method"]}:{r.get("confirmed", 0)}'
        if r["symbol"] in selected and r["decision"] == "TEYİT VAR" and key not in known:
            events.append(dict(r, id=key, observed_at=now, mode="MONITOR_ONLY",
                               execution_eligible=False))
    return dict(version=VERSION, observed_at=now, mode="MONITOR_ONLY", rows=diagnostics,
                selected=selected, events=events[-2000:], references=[
                    {"symbol":t["symbol"], "change":t.get("priceChangePercent")}
                    for t in tickers if t.get("symbol") in ("BTCUSDT", "ETHUSDT")])


def paris(ms):
    return datetime.fromtimestamp(ms/1000, ZoneInfo("Europe/Paris")).strftime("%d.%m.%Y %H:%M") if ms else ""


def sheet_values(obj):
    values = [[TAB, "YALNIZ İZLEME — GERÇEK EMİR YOK", paris(obj["observed_at"]*1000)],
              ["Model", VERSION, "4s yapı / 15m teyit", "Giriş, stop ve TP gösterge seviyeleridir"],
              ["BTC / ETH", json.dumps(obj.get("references", []), ensure_ascii=False)],
              ["Seçilenler", ", ".join(obj["selected"]) or "Henüz uygun aday yok"],
              ["Eksik veri", "Mevcut Crash history kuyruğu; kaynaklar işaretli; yapay mum yok"],
              HEADERS]
    for r in obj["rows"]:
        values.append([r["symbol"], r["rank"], r["volume"], r["change"], f'{r["count"]}/480',
                       r["four_count"], r["trend"], r["method"], r["decision"],
                       r.get("level", ""), r.get("entry", ""), r.get("stop", ""),
                       r.get("target", ""), paris(r.get("confirmed")), r["reason"],
                       json.dumps(r["sources"], ensure_ascii=False),
                       "ÖLÇÜLMEDİ — işlem uygunluğu onayı yok", "YALNIZ İZLEME"])
    values += [[""]*18 for _ in range(60-len(values))]
    values += [["Son teyit kayıtları (gerçek işlem değil)"], HEADERS]
    for r in reversed(obj.get("events", [])[-100:]):
        values.append([r["symbol"], r["rank"], r["volume"], r["change"], f'{r["count"]}/480',
                       r["four_count"], r["trend"], r["method"], "TEYİT KAYDI",
                       r.get("level", ""), r.get("entry", ""), r.get("stop", ""),
                       r.get("target", ""), paris(r.get("confirmed")), r["reason"],
                       json.dumps(r["sources"]), "ÖLÇÜLMEDİ", "YALNIZ İZLEME"])
    return values + [[""]*18 for _ in range(162-len(values))]


def publish(obj):
    from crash_sheets_dashboard import _credentials, _name
    client, book_id = _credentials()
    if client is None:
        return False
    book = client.open_by_key(book_id)
    if "emafilter" not in _name(book.title):
        raise ValueError("STRUCTURE_WRONG_WORKBOOK")
    tabs = {w.title:w for w in book.worksheets()}
    ws = tabs.get(TAB)
    if ws is None:
        ws = book.add_worksheet(title=TAB, rows=162, cols=18)
        book.batch_update({"requests":[
            {"updateSheetProperties":{"properties":{"sheetId":ws.id,
              "gridProperties":{"frozenRowCount":6}},"fields":"gridProperties.frozenRowCount"}},
            {"repeatCell":{"range":{"sheetId":ws.id,"startRowIndex":5,"endRowIndex":6},
             "cell":{"userEnteredFormat":{"backgroundColor":{"red":.13,"green":.26,"blue":.42},
                     "textFormat":{"bold":True,"foregroundColor":{"red":1,"green":1,"blue":1}}}},
             "fields":"userEnteredFormat"}},
            {"updateDimensionProperties":{"range":{"sheetId":ws.id,"dimension":"COLUMNS", "startIndex":0,"endIndex":18},
             "properties":{"pixelSize":155},"fields":"pixelSize"}}]})
    book.values_batch_update({"valueInputOption":"RAW", "data":[{
        "range":f"'{TAB}'!A1:R162", "values":sheet_values(obj)}]})
    # Read back a small, stable range so logs distinguish successful writes.
    check = ws.get("A1:B1")
    if not check or check[0][:2] != [TAB, "YALNIZ İZLEME — GERÇEK EMİR YOK"]:
        raise ValueError("STRUCTURE_SHEET_VERIFY_FAILED")
    return True


class StructureMonitor:
    def __init__(self, directory, logger, queue_history, clock=time.time):
        self.directory, self.logger, self.queue_history, self.clock = directory, logger, queue_history, clock
        self.path = os.path.join(directory, "structure_agent_monitor.json")
        self.lock = threading.Lock()
        self.next_due = 0

    def tick(self, tickers, candles_get, market, onboard):
        now = self.clock()
        if now < self.next_due or not self.lock.acquire(False):
            return
        self.next_due = now+300
        snapshot = [dict(t) for t in tickers if isinstance(t, dict)]
        def work():
            try:
                obj = scan(self.directory, snapshot, market, onboard, candles_get, now, load(self.path, {}))
                save(self.path, obj)
                # One bounded repair proposal per cycle, actual queue/pacing is
                # owned by ema.py and can refuse while Crash jobs need service.
                missing = [r for r in obj["rows"] if r["age_ready"] and r["count"] < 480]
                if missing:
                    # Persist a round-robin cursor so a bad symbol cannot starve the others.
                    cursor = load(self.path+".cursor", {}).get("cursor", 0)
                    row = missing[cursor % len(missing)]
                    self.queue_history(row["symbol"], now)
                    save(self.path+".cursor", {"cursor":cursor+1})
                sent = publish(obj)
                self.logger(f'[STRUCTURE MONITOR] rows={len(obj["rows"])}; selected={len(obj["selected"])}; '
                            f'events={len(obj["events"])}; sheets_verified={int(sent)}; execution=OFF')
            except Exception as exc:
                self.logger(f'[STRUCTURE MONITOR ERROR] {type(exc).__name__}; execution=OFF')
            finally:
                self.lock.release()
        threading.Thread(target=work, name="structure-agent-monitor", daemon=True).start()
