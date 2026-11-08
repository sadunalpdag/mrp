"""Read-only Crash Bounce V2 -> existing EMA Filter 'Bot Status' / 'Sinyal' tabs.

Never deletes worksheets, never creates new tabs, never calls Binance. All failures
remain in this daemon exporter and cannot gate real orders or WS scanning.
"""
import copy
import json
import math
import os
import re
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from crash_policy import TRACKING_SECONDS, is_active_candidate

INTERVAL = 300
MAX_ROWS = 60
# User-confirmed EMA Filter workbook link; independent of legacy GOOGLE_SHEETS_ID.
# Only this dashboard uses this key. Optional override is deliberate and explicit.
DEFAULT_EMA_FILTER_ID = "1qJ1rcmmAdhhELmdzUovYq1n1UIF5lxdCb1lT3CDPF58"
HEADERS = [
    "Coin", "24s değişim %", "Düşüş sırası", "Son fiyat", "V2 durumu",
    "RSI14", "4s toparlanma %", "Güç /100", "Önceki güç", "Puan farkı",
    "Giriş", "TP", "Giriş uzaklık %", "Risk /3", "Zayıf hacim",
    "RSI/EMA düşüyor", "Tekrar düşük sinyal", "Pozisyon kuralı",
    "Uygunluk", "Sinyal takibi", "Güncelleme", "Sinyal zamanı",
    "Trend kırılımı (5g / 15m)", "Üst çizgi uzaklık %", "Kırılım onay zamanı",
    "Kırılım yönü", "5g veri (15m)", "Destek 1 (5g)", "Direnç 1 (5g)",
    "Fibo 0 dip", "Fibo 23,6", "Fibo 38,2", "Fibo 50",
    "Fibo 61,8", "Fibo 78,6", "Fibo 100 tepe", "Fibo bandı",
    "Sinyal anı V2 mum", "V2 eksik veri açıklaması",
    "Sonradan V2 kontrolü (emirsiz)",
]
FLAGS = ("weak_volume", "falling_rsi_ema", "repeat_lower_entry")
PARIS = ZoneInfo("Europe/Paris")
COLORS = {
    "green": {"red": 0.79, "green": 0.91, "blue": 0.82},
    "yellow": {"red": 1.0, "green": 0.94, "blue": 0.72},
    "red": {"red": 0.98, "green": 0.82, "blue": 0.82},
    "grey": {"red": 0.91, "green": 0.93, "blue": 0.95},
    "header": {"red": 0.13, "green": 0.26, "blue": 0.42},
}

def _num(value):
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError, OverflowError):
        return None

def _time(ts):
    val = _num(ts)
    return (datetime.fromtimestamp(val, PARIS).strftime("%d.%m.%Y %H:%M")
            if val is not None and val > 0 else "—")

def _name(title):
    return re.sub(r"[^a-z0-9]", "", str(title).lower())

def _risk(rec):
    item = rec.get("failure_shadow") or {}
    flags = item.get("flags") if isinstance(item, dict) else None
    if not isinstance(flags, dict):
        # Never claim YOK when the underlying signal-time measure is unknown.
        return None, ["VERİ EKSİK"] * 3, "EKSİK: otomatik alış yok"
    # Show each known criterion independently; unknown is never interpreted as safe.
    marks = [("🔴 VAR" if flags[k] else "🟢 YOK")
             if type(flags.get(k)) is bool else "VERİ EKSİK" for k in FLAGS]
    if any(type(flags.get(k)) is not bool for k in FLAGS):
        return None, marks, "EKSİK: otomatik alış yok"
    count = sum(flags[k] for k in FLAGS)
    rule = ("⛔ İŞLEM YOK" if count == 3 else
            "⚖️ %50 / TP %0,5" if count == 2 else "✅ %100 / TP %1")
    return count, marks, rule


def _score(rec):
    feat = rec.get("v2_features") or {}
    retrospective = rec.get("retrospective_v2") or {}
    scored = (feat.get("strength_score") or rec.get("strength_score") or
              retrospective.get("strength_score") or {})
    val = _num(scored.get("score")) if isinstance(scored, dict) else None
    return round(val, 1) if val is not None and 0 <= val <= 100 else None

def _status(rec, risk):
    if not rec:
        return "⚪ AKTİF ADAY KAYDI YOK", "grey"
    v2 = rec.get("v2_status")
    if risk == 3 or risk is None and v2 == "STRONG_V2":
        return "🔴 ALIŞ YOK (RİSK)", "red"
    if v2 == "STRONG_V2" and risk is not None:
        return ("🟢 V2 ONAYLI" if risk <= 1 else "🟡 V2 / YARIM"), ("green" if risk <= 1 else "yellow")
    if v2 == "PENDING_5D_HISTORY":
        return "🟡 VERİ BEKLENİYOR", "yellow"
    if v2 == "MISSING_5D_HISTORY":
        return "🟡 VERİ EKSİK – DEĞERLENDİRİLEMEDİ", "yellow"
    return "🔴 V2 UYGUN DEĞİL", "red"



def _no_candidate_status(change, live):
    """Dashboard-only description: a missing candidate is not a negative V2 verdict.

    The scanner does not currently persist per-symbol skip reasons. Never
    translate missing history, an entry-level gate or pruned state to
    "SİNYAL YOK" or a passing/failing V2 assessment.
    """
    if change is None:
        return "ADAY KAYDI YOK — 24s VERİSİ BELİRSİZ"
    if change > -12:
        return "EŞİK DIŞI — 24s düşüş %12 altında"
    if isinstance(live, dict) and live.get("status") == "MISSING_5D_HISTORY":
        return "ADAY KAYDI YOK — GÖSTERGE VERİSİ EKSİK"
    return "ADAY KAYDI YOK — NEDEN KAYDEDİLMEMİŞ"


_MISSING_REASONS = {
    "TOO_FEW_5D_CANDLES": "5 günlük mum sayısı 478 altında",
    "LAST_CLOSED_CANDLE_MISSING": "Sinyal anındaki son kapanmış 15dk mumu yok",
    "RECENT_4H_GAPS": "Son 4 saatte 14 mumdan az veri",
}


def _missing_explanation(rec, feat):
    if rec.get("v2_status") not in ("MISSING_5D_HISTORY", "PENDING_5D_HISTORY"):
        return "—"
    cause = _MISSING_REASONS.get(feat.get("missing_reason"), "Veri tamamlanmadı / neden kaydedilmemiş")
    if rec.get("v2_reason") == "HISTORY_NOT_READY_IN_TIME":
        return "Aday süresi içinde karar verilemedi; " + cause
    return cause


def _retrospective_v2(rec, rescore):
    """Re-check original event time using cached real candles; NEVER create orders."""
    if rec.get("v2_status") != "MISSING_5D_HISTORY":
        return "—"
    stored = rec.get("retrospective_v2") or {}
    if isinstance(stored, dict) and stored.get("status") in ("STRONG_V2", "FILTERED_V2"):
        candles = stored.get("candles", "—")
        if stored["status"] == "STRONG_V2":
            return f"SONRADAN V2 ONAYLI ({candles}/480); canlı emir yok"
        return f"SONRADAN FİLTREYİ GEÇMEDİ ({candles}/480); canlı emir yok"
    if not callable(rescore) or not _num(rec.get("created_at")):
        return "—"
    try:
        result = rescore(rec, rec["created_at"])
        if not isinstance(result, dict):
            return "—"
        candles = result.get("candles", "—")
        if result.get("status") == "STRONG_V2":
            return f"SONRADAN V2 ONAYLI ({candles}/480); canlı emir yok"
        if result.get("status") == "FILTERED_V2":
            return f"SONRADAN FİLTREYİ GEÇMEDİ ({candles}/480); canlı emir yok"
        reason = _MISSING_REASONS.get(result.get("missing_reason"), "veri hâlâ yetersiz")
        return f"SONRADAN DA EKSİK ({candles}/480): {reason}"
    except Exception:
        return "Tekrar değerlendirme yapılamadı"

def build_rows(state, tickers, trading, ws, now, score_cache=None, rescore=None, candle_get=None, directory=None):
    """Pure data preparation; unknown measurements remain blank, never invented."""
    score_cache = score_cache or {}
    symbols = state.get("symbols") or {}
    if not isinstance(symbols, dict):
        symbols = {}
    ranking = []
    latest = {}
    for item in tickers or []:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol", "")
        change, price = _num(item.get("priceChangePercent")), _num(item.get("lastPrice"))
        if not re.fullmatch(r"[A-Z0-9]{2,24}USDT", symbol) or change is None or price is None or price <= 0:
            continue
        latest[symbol] = (change, price)
        ranking.append((change, symbol))
    ranking.sort()
    top = [sym for _, sym in ranking[:20]]
    rank = {sym: i + 1 for i, sym in enumerate(top)}
    # Add the most recent V2 signals still being observed, even if no longer top 20.
    extra = sorted(
        ((sym, rec) for sym, rec in symbols.items()
         if isinstance(rec, dict) and is_active_candidate(rec, now)),
        key=lambda pair: _num(pair[1].get("created_at")) or 0, reverse=True)
    names = (top + [sym for sym, _ in extra if sym not in rank])[:MAX_ROWS]
    previous = {}
    rows, colors = [], []
    risk_summary = {0: 0, 1: 0, 2: 0, 3: 0, "unknown": 0}
    eligible = 0
    for sym in names:
        rec = symbols.get(sym) or {}
        if not is_active_candidate(rec, now):
            rec = {}
        v2 = rec.get("v2_features") or {}
        has_signal = bool(rec.get("signal_id"))
        risk, flags, rule = _risk(rec) if has_signal else (None, ["—"] * 3, "—")
        current = _score(rec)
        live = None  # Optional cache-only view of the current CLOSED-candle window.
        # Display-only CURRENT 5-day strength on each ranked coin when cached
        # CLOSED candles allow it; it does not create a V2 signal or BUY.
        # Historical MISSING_5D_HISTORY uses original-time backfill instead.
        if callable(rescore) and (not has_signal or rec.get("v2_status") == "STRONG_V2"):
            try:
                sample = rec if has_signal else {
                    "symbol": sym,
                    "btc_change_24h": next(
                        (t.get("priceChangePercent") for t in tickers or []
                         if isinstance(t, dict) and t.get("symbol") == "BTCUSDT"), None),
                }
                live = rescore(sample, now)
                fresh = (live.get("strength_score") or {}).get("score") if isinstance(live, dict) else None
                fresh = _num(fresh)
                if fresh is not None and 0 <= fresh <= 100:
                    current = round(fresh, 1)
            except Exception:
                pass
        old = score_cache.get(sym) or {}
        old_score = _num(old.get("score")) if old.get("signal_id") == rec.get("signal_id") else None
        difference = round(current - old_score, 1) if current is not None and old_score is not None else "—"
        if current is not None:
            # Keep last dashboard score for non-signal coins, too, so the
            # previous score and delta columns can populate on the next tick.
            previous[sym] = {"signal_id": rec.get("signal_id"), "score": current}
        change, price = latest.get(sym, (None, None))
        entry = _num(rec.get("entry")) if has_signal else None
        tp = _num(rec.get("tp")) if has_signal else None
        dist = round((price / entry - 1) * 100, 2) if price and entry and entry > 0 else "—"
        status, color = _status(rec if has_signal else None, risk)
        # This original-signal snapshot is different from the CURRENT rolling
        # five-day coverage shown in the trend columns.
        original_coverage = (f'{v2.get("candles")}/480'
                             if has_signal and isinstance(v2.get("candles"), int)
                             else "—")
        missing_explanation = _missing_explanation(rec, v2) if has_signal else "—"
        retrospective = _retrospective_v2(rec, rescore) if has_signal else "—"
        trend = {"status": "VERİ YOK", "direction": "—", "coverage": "0/480",
                 "distance_pct": None, "confirmed_at": None,
                 "support": None, "resistance": None, "fib": {}, "fib_band": "—"}
        if callable(candle_get):
            try:
                from crash_levels_5d import analyze, load_cached
                # Cached five-day local research + existing closed WS bars ONLY.
                # No archive requests, no Binance calls, and no entry impact.
                trend = analyze(load_cached(directory or "", sym, candle_get(sym), now), now)
            except Exception:
                pass
        trend_status = trend["status"]
        trend_label = {
            "KIRDI": "🟢 KIRDI", "YAKLAŞIYOR": "🟡 YAKLAŞIYOR",
            "KIRMADI": "🔴 KIRMADI", "SAHTE KIRILIM": "🔴 SAHTE KIRILIM",
            "KIRILIM BEKLEME": "🟡 KIRILIM BEKLEME",
            "KANAL YOK": "⚪ KANAL YOK", "VERİ YOK": "⚪ VERİ YOK",
        }.get(trend_status, "⚪ VERİ YOK")
        if color == "green":
            eligible += 1
        if has_signal:
            risk_summary[risk if risk is not None else "unknown"] += 1
        # For a coin with no V2 candidate, use current real CLOSED candles to
        # display RSI/recovery. Never mistake this dashboard-only calculation
        # for a V2 signal, Telegram event, or automatic entry approval.
        visible_v2 = v2 if has_signal else (live if isinstance(live, dict) else {})
        rows.append([
            sym, round(change, 2) if change is not None else "—", rank.get(sym, "—"),
            price if price is not None else "—", (rec.get("v2_status") or "V2 DURUMU KAYITLI DEĞİL") if has_signal
            else _no_candidate_status(change, live),
            visible_v2.get("rsi14", "—"), visible_v2.get("recovery_4h_pct", "—"),
            (current if current is not None else
             "VERİ EKSİK" if has_signal and rec.get("v2_status") in (
                 "PENDING_5D_HISTORY", "MISSING_5D_HISTORY") else
             "HESAPLANAMADI"),
            old_score if old_score is not None else "—", difference,
            entry if entry is not None else "—", tp if tp is not None else "—", dist,
            risk if risk is not None else "—", *flags, rule, status,
            rec.get("tracking_status", "—") if has_signal else "—",
            _time(now), _time(rec.get("created_at")) if has_signal else "—",
            trend_label, trend["distance_pct"] if trend["distance_pct"] is not None else "—",
            _time(trend["confirmed_at"]),
            trend["direction"], trend["coverage"],
            trend["support"] if trend["support"] is not None else "—",
            trend["resistance"] if trend["resistance"] is not None else "—",
            *[(trend["fib"].get(k) if trend["fib"].get(k) is not None else "—")
              for k in ("0", "23.6", "38.2", "50", "61.8", "78.6", "100")],
            trend["fib_band"], original_coverage, missing_explanation, retrospective,
        ])
        colors.append((color, 'grey' if risk is None else 'green' if risk <= 1 else 'yellow' if risk == 2 else 'red'))
    history = state.get("history") or []
    since = now - 86400
    recent = [r for r in history if isinstance(r, dict) and (_num(r.get("created_at")) or 0) > since]
    scores = [_score(r) for r in recent]
    scores = [x for x in scores if x is not None]
    status_rows = [
        ["Gösterge", "Değer"],
        ["Son güncelleme (Paris)", _time(now)],
        ["Bot", "ÇALIŞIYOR (dashboard güncellemesi)"],
        ["Binance WS", "BAĞLI" if ws.get("connected") else "BAĞLI DEĞİL / VERİ BEKLENİYOR"],
        ["Otomatik alış", trading.get("auto", "BİLİNMİYOR")],
        ["Güvenlik kilidi", trading.get("halt") or "YOK"],
        ["Açık / bekleyen gerçek işlemler (yerel kayıt)", trading.get("open", "BİLİNMİYOR")],
        ["Günlük giriş", trading.get("daily", "BİLİNMİYOR")],
        ["Günlük giriş limiti", trading.get("max_daily", "BİLİNMİYOR")],
        ["Eşzamanlı sembol sınırı", trading.get("max_symbols", "BİLİNMİYOR")],
        ["Varsayılan pozisyon USDT", trading.get("amount", "BİLİNMİYOR")],
        ["İzlenen futures fiyatı (WS)", len(latest)],
        ["İlk 20 düşen coin", len(top)],
        ["-12% ve üzeri düşüş (ilk 20)", sum(1 for sym in top if latest[sym][0] <= -12)],
        ["Sinyal listesi satırı", len(rows)],
        ["V2 onaylı / düşük risk", eligible],
        ["Son 24 saat V2 adayı", len(recent)],
        ["Son 24 saat ort. güç puanı", round(sum(scores)/len(scores), 1) if scores else "—"],
        ["0 risk sinyali", risk_summary[0]],
        ["1 risk sinyali", risk_summary[1]],
        ["2 risk sinyali", risk_summary[2]],
        ["3 risk sinyali", risk_summary[3]],
        ["Eksik risk verisi", risk_summary["unknown"]],
        ["Not", "Risk/puan bilgilendirme amaçlıdır; Binance emirlerini değiştirmez."],
    ]
    return status_rows, [HEADERS] + rows, colors, previous

def _credentials():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sheet_id = (os.environ.get("CRASH_SHEETS_TARGET_ID") or DEFAULT_EMA_FILTER_ID).strip()
    if not raw or not sheet_id:
        return None, None
    from google.oauth2.service_account import Credentials
    import gspread
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None
    creds = (Credentials.from_service_account_info(obj, scopes=scopes) if isinstance(obj, dict)
             else Credentials.from_service_account_file(raw, scopes=scopes))
    return gspread.authorize(creds), sheet_id

def _tabs(sheet):
    # Fail closed if Render points to the wrong file. Never add/delete worksheets.
    if "emafilter" not in _name(sheet.title):
        raise ValueError("SHEETS_TITLE_NOT_EMA_FILTER")
    tabs = sheet.worksheets()
    by_name = {_name(w.title): w for w in tabs}
    status = by_name.get("botstatus")
    signal = by_name.get("sinyal") or by_name.get("signal") or by_name.get("signals")
    if not status or not signal:
        raise ValueError("BOT_STATUS_OR_SINYAL_TAB_MISSING")
    return status, signal

def _format(sheet, status, signal, colors, signal_rows):
    requests = []
    for tab, end_col in ((status, 2), (signal, len(HEADERS))):
        requests.append({"repeatCell": {"range": {"sheetId": tab.id, "startRowIndex": 0,
            "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": end_col},
            "cell": {"userEnteredFormat": {"backgroundColor": COLORS["header"],
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}}},
            "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat"}})
        requests.append({"updateSheetProperties": {
            "properties": {"sheetId": tab.id, "gridProperties": {"frozenRowCount": 1}},
            "fields": "gridProperties.frozenRowCount"}})
    # Values are refreshed over the complete dashboard rectangle, but older
    # background fills used to survive when a coin moved to a different row.
    # Clear ONLY panel-owned indicator/risk backgrounds before recoloring the
    # currently measured cells (no other workbook content or tabs touched).
    for first_col, last_col in ((5, 8), (13, 17)):
        requests.append({"repeatCell": {
            "range": {"sheetId": signal.id, "startRowIndex": 1,
                      "endRowIndex": MAX_ROWS + 1,
                      "startColumnIndex": first_col,
                      "endColumnIndex": last_col},
            "cell": {"userEnteredFormat": {}},
            "fields": "userEnteredFormat.backgroundColor"}})
    for i, (kind, risk_kind) in enumerate(colors, start=1):
        data = signal_rows[i]
        extra = []
        rsi, recovery, score, distance = (_num(data[j]) for j in (5, 6, 7, 12))
        if rsi is not None:
            extra.append((5, "green" if rsi >= 20 else "yellow" if rsi >= 18 else "red"))
        if recovery is not None:
            extra.append((6, "green" if recovery >= 1 else "yellow" if recovery >= 0.7 else "red"))
        if score is not None:
            # Display-only quality score, NOT an entry filter.
            extra.append((7, "green" if score >= 70 else "yellow" if score >= 45 else "red"))
        if distance is not None:
            extra.append((12, "green" if abs(distance) <= 0.3 else
                          "yellow" if abs(distance) <= 1 else "red"))
        for j in (14, 15, 16):
            if data[j] in ("🟢 YOK", "🔴 VAR"):
                extra.append((j, "green" if data[j] == "🟢 YOK" else "red"))
        trend_status = str(data[22])
        trend_tone = ("green" if trend_status.startswith("🟢") else
                      "yellow" if trend_status.startswith("🟡") else
                      "red" if trend_status.startswith("🔴") else "grey")
        extra.append((22, trend_tone))
        direction_tone = ("green" if data[25] == "YUKARI" else
                          "red" if data[25] == "AŞAĞI" else "grey")
        extra.append((25, direction_tone))
        distance_to_upper = _num(data[23])
        if distance_to_upper is not None:
            extra.append((23, "green" if trend_tone == "green" else
                          "yellow" if abs(distance_to_upper) <= 0.5 else "grey"))
        for column, tone in [(18, kind), (13, risk_kind)] + extra:
            requests.append({"repeatCell": {"range": {"sheetId": signal.id,
                "startRowIndex": i, "endRowIndex": i + 1,
                "startColumnIndex": column, "endColumnIndex": column + 1},
                "cell": {"userEnteredFormat": {"backgroundColor": COLORS[tone]}},
                "fields": "userEnteredFormat.backgroundColor"}})
    if requests:
        sheet.batch_update({"requests": requests})

def _read_previous(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def _write_previous(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)

def publish(state, tickers, trading, ws, now, directory, rescore=None, candle_get=None):
    client, sheet_id = _credentials()
    if not client:
        return False
    sheet = client.open_by_key(sheet_id)
    status, signal = _tabs(sheet)
    # Expand only the existing Sinyal tab; never create or delete worksheets.
    if getattr(signal, "col_count", len(HEADERS)) < len(HEADERS):
        sheet.batch_update({"requests": [{"updateSheetProperties": {
            "properties": {"sheetId": signal.id,
                           "gridProperties": {"columnCount": len(HEADERS)}},
            "fields": "gridProperties.columnCount"}}]})
    os.makedirs(directory, exist_ok=True)
    prev_path = os.path.join(directory, "crash_sheets_score_previous.json")
    status_rows, signal_rows, colors, previous = build_rows(
        state, tickers, trading, ws, now, _read_previous(prev_path), rescore, candle_get, directory)
    # Update only panel-owned rectangular ranges. Old worksheets and other data untouched.
    status_rows = status_rows + [["", ""]] * max(0, 26 - len(status_rows))
    signal_rows = signal_rows + [[""] * len(HEADERS)] * max(0, MAX_ROWS + 1 - len(signal_rows))
    def column_letter(n):
        letters = ""
        while n:
            n, rem = divmod(n - 1, 26)
            letters = chr(65 + rem) + letters
        return letters
    sheet.values_batch_update({"valueInputOption": "RAW", "data": [
        {"range": "'" + status.title.replace("'", "''") + "'!A1:B26", "values": status_rows},
        {"range": "'" + signal.title.replace("'", "''") + "'!A1:" + column_letter(len(HEADERS)) + str(MAX_ROWS + 1), "values": signal_rows},
    ]})
    _format(sheet, status, signal, colors, signal_rows)
    _write_previous(prev_path, previous)
    return True

class DashboardPublisher:
    def __init__(self, directory, logger, rescore=None, candle_get=None):
        self.directory, self.logger, self.rescore = directory, logger, rescore
        self.candle_get = candle_get
        self.lock = threading.Lock()
        self.next_due = 0.0

    def tick(self, state, tickers, executor, ws):
        if not os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON"):
            return
        now = time.time()
        if now < self.next_due or not self.lock.acquire(blocking=False):
            return
        self.next_due = now + INTERVAL
        try:
            # Small immutable snapshot; never send state files or API keys to Sheets.
            symbols = copy.deepcopy(state.get("symbols", {}))
            recent = copy.deepcopy([r for r in state.get("history", [])
                if isinstance(r, dict) and (_num(r.get("created_at")) or 0) > now - 86400])
            trading = {}
            if executor is not None:
                trading = {
                    "auto": "AÇIK" if executor.enabled.is_set() else "KAPALI",
                    "halt": executor.state.get("halt"),
                    "open": sum(o.get("phase") not in ("DONE", "NO_FILL", "CLOSED_EXTERNALLY", "ENTRY_NOT_FOUND_FLAT")
                        for o in list(executor.state.get("orders", {}).values())),
                    "daily": executor.daily_count(), "max_daily": executor.max_daily,
                    "max_symbols": executor.max_symbols, "amount": str(executor.position_usdt),
                }
            copied_ws = dict(ws or {})
            copied_tickers = [dict(t) for t in tickers or []]
        except Exception:
            self.lock.release()
            return
        def writer():
            try:
                if publish({"symbols": symbols, "history": recent}, copied_tickers,
                           trading, copied_ws, now, self.directory, self.rescore, self.candle_get):
                    self.logger("[CRASH SHEETS] Bot Status / Sinyal updated")
            except Exception as exc:
                # Never log service-account JSON or raw Google error response.
                self.logger("[CRASH SHEETS] update skipped: " + type(exc).__name__)
            finally:
                self.lock.release()
        threading.Thread(target=writer, daemon=True, name="crash-sheets-dashboard").start()
