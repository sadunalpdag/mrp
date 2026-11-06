#!/usr/bin/env python3
"""Reset only the EMA Filter Google spreadsheet, with a persistent backup.

Run from the MRP Render worker shell with its existing environment.
Without --apply this is read-only. No Binance requests, trades or bot imports.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import tempfile

STATUS_TAB = "BOT_STATUS"
TRACK_TAB = "COIN_TRACKING"
STATUS_ROWS = [
    ["MRP — CRASH BOUNCE V2", "BOT TAKİP PANELİ"],
    ["Güncelleme", "CANLI BAĞLANTI HENÜZ KURULMADI"],
    ["Otomatik alış", "—"],
    ["Genel güvenlik kilidi", "—"],
    ["ENTRY_UNKNOWN", "—"],
    ["TP_UNKNOWN", "—"],
    ["Piyasa WebSocket", "—"],
    ["Hesap WebSocket", "—"],
    ["İzlenen coin", "—"],
    ["V2 kriterlerine uyan", "—"],
    ["Son tarama", "—"],
    ["Sheets güncelleme hatası", "—"],
]
TRACK_HEADERS = [
    "Coin", "Fiyat", "24s Değişim %", "Düşüş Sırası", "5g Mum Adedi",
    "RSI14", "RSI≥20", "4s Dipten Toparlanma %", "Toparlanma≥1%",
    "Giriş Fiyatı", "Sinyal Durumu", "Hacim Riski", "RSI/EMA Riski",
    "Tekrar Düşüş Riski", "Risk Sayısı", "Öneri TP %", "Son Güncelleme",
]


def guard_target(sheet_id, title):
    """Prevent accidentally cleaning the hard-coded legacy fallback workbook."""
    if not sheet_id or not re.fullmatch(r"[A-Za-z0-9_-]{20,100}", sheet_id):
        raise ValueError("Render Environment içinde geçerli GOOGLE_SHEETS_ID gerekli")
    normalized = re.sub(r"[^a-z0-9]", "", str(title).casefold())
    if "emafilter" not in normalized:
        raise ValueError("Dosya adı EMA Filter içermiyor; yanlış dosya olabilir, durduruldu")


def auth_client(env):
    raw = env.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sheet_id = (env.get("GOOGLE_SHEETS_ID") or env.get("GOOGLE_SHEET_ID") or "").strip()
    if not raw or not sheet_id:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON ve GOOGLE_SHEETS_ID tanımlı olmalı")
    from google.oauth2.service_account import Credentials
    import gspread
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    try:
        info = json.loads(raw)
        credentials = Credentials.from_service_account_info(info, scopes=scopes)
    except json.JSONDecodeError:
        credentials = Credentials.from_service_account_file(raw, scopes=scopes)
    return gspread.authorize(credentials), sheet_id


def archive_sheet(spreadsheet, tabs, dest):
    """Archive formulas and values before any tab is removed."""
    payload = {
        "sheet_id": spreadsheet.id,
        "title": spreadsheet.title,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "tabs": [],
    }
    for tab in tabs:
        payload["tabs"].append({
            "title": tab.title,
            "id": tab.id,
            "rows": tab.row_count,
            "cols": tab.col_count,
            "values_and_formulas": tab.get_all_values(value_render_option="FORMULA"),
        })
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".ema_filter_", delete=False) as temp:
        tmp = Path(temp.name)
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        if tmp.stat().st_size <= 0:
            raise IOError("Yedek dosyası boş")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def reset_sheet(spreadsheet):
    """Replace old tabs and formats; retain original spreadsheet ID and sharing."""
    old_ids = [tab.id for tab in spreadsheet.worksheets()]
    if not old_ids:
        raise ValueError("Sekme bulunamadı")
    temp_name = "__EMA_RESET_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    tmp = spreadsheet.add_worksheet(title=temp_name, rows=1, cols=1)
    requests = (
        [{"deleteSheet": {"sheetId": tab_id}} for tab_id in old_ids]
        + [
            {"addSheet": {"properties": {"title": STATUS_TAB,
                "gridProperties": {"rowCount": 40, "columnCount": 8}}}},
            {"addSheet": {"properties": {"title": TRACK_TAB,
                "gridProperties": {"rowCount": 220, "columnCount": 17}}}},
            {"deleteSheet": {"sheetId": tmp.id}},
        ]
    )
    spreadsheet.batch_update({"requests": requests})
    status = spreadsheet.worksheet(STATUS_TAB)
    track = spreadsheet.worksheet(TRACK_TAB)
    spreadsheet.values_batch_update({"valueInputOption": "RAW", "data": [
        {"range": "'BOT_STATUS'!A1:B12", "values": STATUS_ROWS},
        {"range": "'COIN_TRACKING'!A1:Q1", "values": [TRACK_HEADERS]},
    ]})
    header_format = {
        "backgroundColor": {"red": 0.08, "green": 0.16, "blue": 0.28},
        "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
    }
    spreadsheet.batch_update({"requests": [
        {"repeatCell": {"range": {"sheetId": status.id, "startRowIndex": 0, "endRowIndex": 1},
            "cell": {"userEnteredFormat": header_format}, "fields": "userEnteredFormat"}},
        {"repeatCell": {"range": {"sheetId": track.id, "startRowIndex": 0, "endRowIndex": 1},
            "cell": {"userEnteredFormat": header_format}, "fields": "userEnteredFormat"}},
        {"updateSheetProperties": {"properties": {"sheetId": track.id,
            "gridProperties": {"frozenRowCount": 1}}, "fields": "gridProperties.frozenRowCount"}},
        {"autoResizeDimensions": {"dimensions": {"sheetId": status.id,
            "dimension": "COLUMNS", "startIndex": 0, "endIndex": 2}}},
        {"autoResizeDimensions": {"dimensions": {"sheetId": track.id,
            "dimension": "COLUMNS", "startIndex": 0, "endIndex": 17}}},
    ]})
    return [STATUS_TAB, TRACK_TAB]


def main(argv=None, env=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Yedek aldıktan sonra eski sekmeleri sil")
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    client, sheet_id = auth_client(env)
    spreadsheet = client.open_by_key(sheet_id)
    guard_target(sheet_id, spreadsheet.title)
    tabs = spreadsheet.worksheets()
    print("Dosya:", spreadsheet.title)
    print("ID:", spreadsheet.id)
    print("Eski sekmeler:", ", ".join(tab.title for tab in tabs))
    if not args.apply:
        print("ÖNİZLEME: Hiçbir şey silinmedi. Temizlik için --apply gerekir.")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(env.get("DATA_DIR", "/data")) / "ema_filter_backups" / ("ema_filter_" + stamp + ".json.gz")
    archive_sheet(spreadsheet, tabs, dest)
    print("Yedek:", dest)
    print("Yeni sekmeler:", ", ".join(reset_sheet(spreadsheet)))
    print("Temizlik tamamlandı; bu script canlı veri bağlantısı kurmaz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
