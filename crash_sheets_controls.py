"""Owner-managed Google Sheets Bot Status controls for the Crash Bounce worker.

Only the existing EMA Filter "Bot Status" worksheet is used. A:B remains the
read-only dashboard; D:I is an explicit-approval settings panel. No trading
orders are placed here. All real configuration changes are delegated to the
existing serialized NightExecutor control queue.
"""
import os
from decimal import Decimal, InvalidOperation

HEADER = ["AYAR", "MEVCUT", "YENI DEGER", "ACIKLAMA", "SONUC", "ONAY"]
ROWS = [
    ("LIVE", "Gercek otomatik alis", "AÇIK/KAPALI. Acmak icin CANLI ONAY; kapatmak icin UYGULA.", True),
    ("PAPER", "Sanal gunluk kar takibi", "AÇIK/KAPALI. Emir olusturmaz; yalniz gozlem.", True),
    ("TARGET", "Gunluk kar hedefi (%)", "0,5-20. Yeni oran ertesi Paris gunu uygulanir. SANAL.", True),
    ("DAILY", "Gunluk yeni giris limiti", "1-50. Artirmak icin RISK ONAY; azaltmak icin UYGULA.", True),
    ("AMOUNT", "Pozisyon USDT", "1-1000 USDT ISLEM TUTARI (teminat degil). Artis: RISK ONAY.", True),
    ("SYMBOLS", "Eszamanli sembol limiti", "1-50. Acik + bekleyen kapasitesi. Artis: RISK ONAY.", True),
    ("OPEN", "Acik pozisyon limiti", "1-50. Yeni alislari sinirlar; mevcut isleme dokunmaz.", True),
    ("PENDING", "Bekleyen alis limiti", "1-50. Artis: RISK ONAY; dusus: UYGULA.", True),
    ("COUNTER", "Bugunku kullanilan giris", "Sadece bilgi; sayac sifirlama guvenlik icin kilitli.", False),
    ("MANUAL_TP", "Elle acilan pozisyona TP", "Yalniz Render baslangic ayari; bu panelden degismez.", False),
    ("LEVERAGE", "Kaldirac", "10x, kodda sabit. Bu panelden degismez.", False),
    ("HBREAK", "HBREAK sanal test", "Yalniz durum; bu panelden acilip kapatilmaz.", False),
    ("API2", "CoinBeacon API2 sanal test", "Alti strateji. Bu panelden acilip kapatilmaz.", False),
    ("REAL_PROFIT", "Gunluk karla GERCEK kapanis", "GELISTIRILMEDI: Yalniz sanal fotoğraf; GERCEK TP YOK.", False),
]
COUNT = len(ROWS) + 1
KINDS = {"DAILY": ("limit", "max_daily", 50),
         "AMOUNT": ("amount", "position_usdt", 1000),
         "SYMBOLS": ("symbols", "max_symbols", 50),
         "OPEN": ("open_limit", "max_open_positions", 50),
         "PENDING": ("pending_limit", "max_pending_entries", 50)}


def _mode(raw):
    value = str(raw).strip().upper()
    if value in ("AÇIK", "ACIK", "ON", "1"):
        return True
    if value in ("KAPALI", "OFF", "0"):
        return False
    raise ValueError("AÇIK veya KAPALI yazin")


def _decimal(raw, minimum, maximum, integer=False):
    value = str(raw).strip().replace(",", ".").removesuffix("%").strip()
    try:
        number = Decimal(value)
        if (not number.is_finite() or not Decimal(str(minimum)) <= number <= Decimal(str(maximum))
                or (integer and number != number.to_integral_value())):
            raise ValueError
    except (ValueError, InvalidOperation):
        raise ValueError(f"Aralik {minimum}-{maximum}" + (" (tam sayi)" if integer else "")) from None
    return number


def _values(executor):
    from daily_profit_lock_paper import load_settings
    settings = load_settings()
    result = {
        "PAPER": "AÇIK" if settings["enabled"] and os.getenv("DAILY_PROFIT_PAPER_ENABLED", "0") == "1" else "KAPALI",
        "TARGET": str(settings["target_pct"]),
        "HBREAK": "AKTIF (sanal)" if os.getenv("HBREAK_PAPER_ENABLED") == "1" else "KAPALI",
        "API2": "AKTIF (sanal)" if os.getenv("COINBEACON_SIX_PAPER_ENABLED") == "1" else "KAPALI",
        "LEVERAGE": "10x", "REAL_PROFIT": "SANAL / GERCEK EMIR YOK",
    }
    if executor is None:
        result.update({key: "MODUL YOK" for key in
                       ("LIVE", "DAILY", "AMOUNT", "SYMBOLS", "OPEN", "PENDING", "COUNTER", "MANUAL_TP")})
    else:
        result.update({
            "LIVE": "AÇIK" if executor.enabled.is_set() else "KAPALI",
            "DAILY": str(executor.max_daily),
            "AMOUNT": str(executor.position_usdt),
            "SYMBOLS": str(executor.max_symbols),
            "OPEN": str(executor.max_open_positions),
            "PENDING": str(executor.max_pending_entries),
            "COUNTER": str(executor.daily_count()),
            "MANUAL_TP": "AÇIK" if executor.manual_tp_enabled else "KAPALI",
        })
    return result


def _apply(key, raw, approval, executor):
    """Return (success, result). Requires explicit confirmation per request."""
    text = str(approval).strip().upper()
    if key in ("LIVE", "PAPER"):
        on = _mode(raw)
        expected = ("CANLI ONAY" if on else "UYGULA") if key == "LIVE" else "UYGULA"
        if text != expected:
            raise ValueError("I sutununa " + expected + " yazin")
        if key == "PAPER":
            if on and os.getenv("DAILY_PROFIT_PAPER_ENABLED", "0") != "1":
                raise ValueError("Sanal izleyici Render'da kapali")
            from daily_profit_lock_paper import control, load_settings
            control("paper_on" if on else "paper_off")
            if load_settings()["enabled"] != on:
                raise ValueError("Sanal takip dogrulanamadi")
            return True, "✅ SANAL takip " + ("acildi" if on else "kapatildi")
        if executor is None:
            raise ValueError("Gercek islem modulu bulunamadi")
        if on and os.getenv("CRASH_SHEETS_ALLOW_LIVE_ON", "0") != "1":
            raise ValueError("Canli acilis Render yetkisi kapali")
        if bool(executor.enabled.is_set()) == on and (on or bool(executor.state.get("paused"))):
            return True, "✅ Zaten " + ("acik" if on else "kapali")
        response = executor.request_control("on" if on else "off")
        if bool(executor.enabled.is_set()) == on and (on or bool(executor.state.get("paused"))):
            return True, "✅ Gercek yeni alislar " + ("ACIK" if on else "KAPALI")
        return False, "⚠️ Islem onaylanmadi: " + str(response)[:95]
    if key == "TARGET":
        if text != "UYGULA":
            raise ValueError("I sutununa UYGULA yazin")
        pct = _decimal(raw, "0.5", 20)
        from daily_profit_lock_paper import control, load_settings
        control("paper_target:" + str(pct))
        if Decimal(load_settings()["target_pct"]) != pct:
            raise ValueError("Hedef ayari kaydedilemedi")
        return True, "✅ %" + str(pct) + " sonraki Paris gunu"
    if key in KINDS:
        if executor is None:
            raise ValueError("Gercek islem modulu bulunamadi")
        action, attr, limit = KINDS[key]
        value = _decimal(raw, 1, limit, integer=(key != "AMOUNT"))
        old = Decimal(str(getattr(executor, attr)))
        confirm = "RISK ONAY" if value > old else "UYGULA"
        if text != confirm:
            raise ValueError("I sutununa " + confirm + " yazin")
        if old == value:
            return True, "✅ Zaten " + str(value)
        response = executor.request_control(action + ":" + str(value))
        if Decimal(str(getattr(executor, attr))) != value:
            return False, "⚠️ Degisiklik dogrulanamadi: " + str(response)[:90]
        return True, "✅ " + str(value) + " kaydedildi (yeni girisler)"
    raise ValueError("Bu satir sadece okunur")


def _rect(rows, size):
    return [(list(row) + [""] * 6)[:6] for row in list(rows[:size]) + [[]] * max(0, size - len(rows))]


def sync_panel(sheet, tab, executor, logger=None):
    """Read edits BEFORE writing anything; never touch dashboard A:B or signal tab.

    First run builds clean D:I panel and deliberately discards any old/stale
    confirmations. Every later cycle consumes only explicitly approved edits.
    """
    if getattr(tab, "col_count", 0) < 9:
        sheet.batch_update({"requests": [{"updateSheetProperties": {
            "properties": {"sheetId": tab.id, "gridProperties": {"columnCount": 9}},
            "fields": "gridProperties.columnCount"}}]})
    observed = _rect(tab.get(f"D1:I{COUNT}"), COUNT)
    ready = (observed[0] == HEADER and
             all(observed[i + 1][0] == key for i, (key, _, _, _) in enumerate(ROWS)))
    current = _values(executor)
    left, right, clears, applied = [HEADER[:2]], [[HEADER[3], HEADER[4]]], [], 0
    for i, (key, label, description, editable) in enumerate(ROWS, start=2):
        old = observed[i - 1]
        result = old[4] if ready else "Hazir"
        wanted, approval = str(old[2]).strip(), str(old[5]).strip()
        if ready and approval:
            try:
                if not editable:
                    raise ValueError("Bu ayar sadece okunur")
                if not wanted:
                    raise ValueError("Once F sutununa yeni deger yazin")
                success, result = _apply(key, wanted, approval, executor)
                if success:
                    applied += 1
                    current = _values(executor)
            except Exception as error:
                # Do not expose credentials or raw HTTP responses in the sheet.
                result = "⚠️ " + (str(error) if isinstance(error, ValueError) else type(error).__name__)[:100]
                success = False
            # Always disarm the consumed approval; prevents automatic retries,
            # especially if an executor command response was uncertain.
            clears.append((i, success))
        elif ready and wanted:
            result = "⏳ Onay bekleniyor (I sutunu)"
        left.append([label, current.get(key, "—")])
        right.append([description, result])
    prefix = "'" + tab.title.replace("'", "''") + "'!"
    data = [
        {"range": prefix + f"D1:E{COUNT}", "values": left},
        {"range": prefix + f"G1:H{COUNT}", "values": right},
    ]
    if not ready:
        data.extend([
            {"range": prefix + f"F2:F{COUNT}", "values": [[""] for _ in ROWS]},
            {"range": prefix + f"I2:I{COUNT}", "values": [[""] for _ in ROWS]},
        ])
    else:
        for row, success in clears:
            data.append({"range": prefix + f"I{row}", "values": [[""]]})
            if success:
                data.append({"range": prefix + f"F{row}", "values": [[""]]})
    sheet.values_batch_update({"valueInputOption": "RAW", "data": data})
    if not ready:
        # Formatting does not change other tabs or any existing A:B status data.
        try:
            sheet.batch_update({"requests": [
                {"repeatCell": {"range": {"sheetId": tab.id, "startRowIndex": 0,
                                          "endRowIndex": 1, "startColumnIndex": 3,
                                          "endColumnIndex": 9},
                 "cell": {"userEnteredFormat": {
                     "backgroundColor": {"red": 0.13, "green": 0.26, "blue": 0.42},
                     "textFormat": {"bold": True, "foregroundColor": {
                         "red": 1, "green": 1, "blue": 1}}}},
                 "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat"}},
                {"repeatCell": {"range": {"sheetId": tab.id, "startRowIndex": 1,
                                          "endRowIndex": COUNT, "startColumnIndex": 5,
                                          "endColumnIndex": 6},
                 "cell": {"userEnteredFormat": {"backgroundColor": {
                     "red": 0.96, "green": 0.96, "blue": 0.76}}},
                 "fields": "userEnteredFormat.backgroundColor"}},
                {"repeatCell": {"range": {"sheetId": tab.id, "startRowIndex": 1,
                                          "endRowIndex": COUNT, "startColumnIndex": 8,
                                          "endColumnIndex": 9},
                 "cell": {"userEnteredFormat": {"backgroundColor": {
                     "red": 0.96, "green": 0.96, "blue": 0.76}}},
                 "fields": "userEnteredFormat.backgroundColor"}},
            ]})
        except Exception as exc:
            if logger:
                logger("[CRASH SHEETS CONTROL] format skipped: " + type(exc).__name__)
        if logger:
            logger("[CRASH SHEETS CONTROL] settings panel initialized D:I; no commands executed")
    elif logger and (applied or clears):
        logger(f"[CRASH SHEETS CONTROL] applied={applied}; consumed={len(clears)}")
    return applied
