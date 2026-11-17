"""Gemini settings in existing EMA Filter > Bot Status J:O, independent of D:I."""
import os

from coinbeacon_ai_settings import load_settings, update_setting

HEADER = ["GEMINI AI AYARI", "MEVCUT", "YENI DEGER", "ACIKLAMA", "SONUC", "ONAY"]
ROWS = [
    ("enabled", "AI Learning", "AÇIK/KAPALI. Sadece sanal AI; gerçek emir göndermez."),
    ("daily_call_cap", "Gunluk Gemini cagrisi", "1-200. Ilk ve sonuc analizleri toplam."),
    ("max_active", "Eszamanli AI sinyal", "1-100. Kuyruk + takip + sonuc bekleyen."),
    ("track_hours", "Takip suresi (saat)", "1-24. Yeni vakalarda uygulanir."),
    ("check_minutes", "Kontrol araligi (dk)", "15/30/60. Mum verisi her 15dk saklanir."),
    ("analysis_phase", "Gemini analizi", "BASLANGIC+SONUC veya YALNIZ BASLANGIC."),
]
COUNT = len(ROWS) + 1


def _display(key, value):
    if key == "enabled":
        return "AÇIK" if value else "KAPALI"
    if key == "analysis_phase":
        return "BASLANGIC+SONUC" if value == "both" else "YALNIZ BASLANGIC"
    return str(value)


def _parse(key, text):
    text = str(text).strip().upper()
    if key == "enabled":
        if text in ("AÇIK", "ACIK", "ON", "1"):
            return True
        if text in ("KAPALI", "OFF", "0"):
            return False
        raise ValueError("AÇIK veya KAPALI yazin")
    if key == "analysis_phase":
        if text in ("BASLANGIC+SONUC", "BAŞLANGIÇ+SONUÇ", "BOTH"):
            return "both"
        if text in ("YALNIZ BASLANGIC", "YALNIZ BAŞLANGIÇ", "INITIAL"):
            return "initial"
        raise ValueError("BASLANGIC+SONUC veya YALNIZ BASLANGIC")
    try:
        n = int(text)
    except ValueError:
        raise ValueError("Tam sayi gerekli") from None
    return n


def sync_panel(sheet, tab, logger=None):
    """Only J:O; approval is disarmed after every attempted command."""
    if getattr(tab, "col_count", 0) < 15:
        sheet.batch_update({"requests": [{"updateSheetProperties": {
            "properties": {"sheetId": tab.id,
                           "gridProperties": {"columnCount": 15}},
            "fields": "gridProperties.columnCount"}}]})
    rows = [list(r) for r in tab.get(f"J1:O{COUNT}")]
    rows = [(r + [""] * 6)[:6] for r in rows[:COUNT]]
    rows += [[""] * 6 for _ in range(COUNT - len(rows))]
    ready = rows[0] == HEADER and all(
        rows[i + 1][0] == item[1] for i, item in enumerate(ROWS))
    settings = load_settings()
    left = [HEADER[:2]]
    right = [[HEADER[3], HEADER[4]]]
    changes = []
    pending = 0
    for index, (key, label, description) in enumerate(ROWS, 2):
        prior = rows[index - 1]
        desired, approval = str(prior[2]).strip(), str(prior[5]).strip()
        result = prior[4] if ready else "Hazir"
        if ready and approval:
            success = False
            try:
                if approval.upper() != "UYGULA":
                    raise ValueError("O sutununa UYGULA yazin")
                if not desired:
                    raise ValueError("L sutununa deger yazin")
                value = _parse(key, desired)
                if key == "enabled" and value:
                    if not os.getenv("GEMINI_API_KEY"):
                        raise ValueError("GEMINI_API_KEY Render'da eksik")
                    if os.getenv("CB2_LIVE_MOMENTUM_ENABLED") != "1":
                        raise ValueError("CB2 momentum kaynagi kapali")
                update_setting(key, value)
                settings = load_settings()
                if key == "enabled" and value:
                    from coinbeacon_ai_learning import start
                    if not start():
                        raise ValueError("Gemini calistirilamadi")
                success = True
                pending += 1
                result = "✅ Kaydedildi"
            except Exception as error:
                result = "⚠️ " + (str(error) if isinstance(error, ValueError)
                                 else type(error).__name__)[:90]
            changes.append((index, success))
        elif ready and desired:
            result = "⏳ Onay bekliyor (O)"
        left.append([label, _display(key, settings[key])])
        right.append([description, result])
    prefix = "'" + tab.title.replace("'", "''") + "'!"
    data = [
        {"range": prefix + f"J1:K{COUNT}", "values": left},
        {"range": prefix + f"M1:N{COUNT}", "values": right},
    ]
    if not ready:
        data += [
            {"range": prefix + "L1", "values": [[HEADER[2]]]},
            {"range": prefix + "O1", "values": [[HEADER[5]]]},
            {"range": prefix + f"L2:L{COUNT}", "values": [[""] for _ in ROWS]},
            {"range": prefix + f"O2:O{COUNT}", "values": [[""] for _ in ROWS]},
        ]
    else:
        for index, success in changes:
            data.append({"range": prefix + f"O{index}", "values": [[""]]})
            if success:
                data.append({"range": prefix + f"L{index}", "values": [[""]]})
    sheet.values_batch_update({"valueInputOption": "RAW", "data": data})
    if logger and (changes or not ready):
        logger(f"[CB AI DASHBOARD] applied={pending} consumed={len(changes)}")
    return pending
