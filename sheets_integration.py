# ==============================================================
#  sheets_integration.py — Google Sheets Integration Module
#  Export trading data and analysis to Google Sheets
#  - Trade history export
#  - Performance reports
#  - Real-time updates
# ==============================================================

import os
import json
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone, timedelta

# ============ Configuration ============
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Environment variables for Google Sheets credentials
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
DATA_DIR = os.getenv("DATA_DIR", "./data")

# ============ Time Functions ============
def now_ist_dt():
    return (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)

def log(msg):
    print(f"[{now_ist_dt()}] [SHEETS] {msg}", flush=True)

# ============ Google Sheets Client ============
def get_sheets_client():
    """Initialize and return Google Sheets client"""
    try:
        if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
            log(f"Credentials file not found: {GOOGLE_CREDENTIALS_FILE}")
            return None
        
        creds = Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_FILE,
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        log("Google Sheets client initialized successfully")
        return client
    except Exception as e:
        log(f"Error initializing Google Sheets client: {e}")
        return None

# ============ Export Functions ============
def export_trades_to_sheet(client, spreadsheet_id, trades_file):
    """Export trades from JSON file to Google Sheets"""
    try:
        if not os.path.exists(trades_file):
            log(f"Trades file not found: {trades_file}")
            return False
        
        # Load trades data
        with open(trades_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            log("No trade data to export")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Open spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Create or get worksheet
        worksheet_name = "Trades"
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
        
        # Convert DataFrame to list format for Google Sheets
        values = [df.columns.tolist()] + df.values.tolist()
        
        # Update worksheet
        worksheet.update('A1', values)
        log(f"Exported {len(df)} trades to sheet '{worksheet_name}'")
        return True
        
    except Exception as e:
        log(f"Error exporting trades: {e}")
        return False

def export_performance_report(client, spreadsheet_id, trades_file):
    """Export performance analysis to Google Sheets"""
    try:
        if not os.path.exists(trades_file):
            log(f"Trades file not found: {trades_file}")
            return False
        
        # Load and analyze data
        with open(trades_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            log("No data for performance report")
            return False
        
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_cols = ["symbol", "dir", "power", "exit_reason", "gain_pct", "duration_sec"]
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # Power band analysis
        bins = [0, 60, 70, 80, 90, 100]
        labels = ["<60", "60-70", "70-80", "80-90", ">90"]
        df["power_band"] = pd.cut(df["power"], bins=bins, labels=labels, include_lowest=True)
        
        # Calculate summary statistics
        summary = df.groupby(["power_band", "exit_reason"]).agg(
            trade_count=("exit_reason", "count"),
            avg_gain_pct=("gain_pct", "mean"),
            avg_duration_min=("duration_sec", lambda x: (x.mean()/60) if len(x) > 0 else 0)
        ).reset_index()
        
        # TP Rate calculation
        pivot = df.pivot_table(
            index="power_band",
            columns="exit_reason",
            values="gain_pct",
            aggfunc="count",
            fill_value=0
        )
        if "TP" in pivot.columns and "SL" in pivot.columns:
            pivot["TP_Rate(%)"] = (pivot["TP"] / (pivot["TP"] + pivot["SL"] + 1e-6)) * 100
        pivot = pivot.reset_index()
        
        # Open spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Export summary
        worksheet_name = "Performance_Summary"
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=100, cols=10)
        
        # Add summary data
        summary_values = [summary.columns.tolist()] + summary.values.tolist()
        worksheet.update('A1', summary_values)
        
        # Export TP Rate pivot
        worksheet_name_pivot = "TP_Rate_Analysis"
        try:
            worksheet_pivot = spreadsheet.worksheet(worksheet_name_pivot)
            worksheet_pivot.clear()
        except gspread.exceptions.WorksheetNotFound:
            worksheet_pivot = spreadsheet.add_worksheet(title=worksheet_name_pivot, rows=100, cols=10)
        
        pivot_values = [pivot.columns.tolist()] + pivot.values.tolist()
        worksheet_pivot.update('A1', pivot_values)
        
        log(f"Performance report exported successfully")
        return True
        
    except Exception as e:
        log(f"Error exporting performance report: {e}")
        return False

def export_balance_history(client, spreadsheet_id, balance_file):
    """Export balance history to Google Sheets"""
    try:
        if not os.path.exists(balance_file):
            log(f"Balance file not found: {balance_file}")
            return False
        
        with open(balance_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            log("No balance history data")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Open spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Create or get worksheet
        worksheet_name = "Balance_History"
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=5000, cols=10)
        
        # Update worksheet
        values = [df.columns.tolist()] + df.values.tolist()
        worksheet.update('A1', values)
        
        log(f"Exported balance history with {len(df)} entries")
        return True
        
    except Exception as e:
        log(f"Error exporting balance history: {e}")
        return False

# ============ Main Export Function ============
def export_all_data():
    """Export all available data to Google Sheets"""
    if not SPREADSHEET_ID:
        log("SPREADSHEET_ID environment variable not set")
        return False
    
    client = get_sheets_client()
    if not client:
        log("Failed to initialize Google Sheets client")
        return False
    
    log(f"Starting export to spreadsheet: {SPREADSHEET_ID}")
    
    success_count = 0
    
    # Export closed trades
    closed_file = os.path.join(DATA_DIR, "sim_closed.json")
    if os.path.exists(closed_file):
        if export_trades_to_sheet(client, SPREADSHEET_ID, closed_file):
            success_count += 1
        if export_performance_report(client, SPREADSHEET_ID, closed_file):
            success_count += 1
    
    # Export real closed trades if available
    real_closed_file = os.path.join(DATA_DIR, "real_closed.json")
    if os.path.exists(real_closed_file):
        if export_trades_to_sheet(client, SPREADSHEET_ID, real_closed_file):
            success_count += 1
    
    # Export balance history if available
    balance_file = os.path.join(DATA_DIR, "balance_history.json")
    if os.path.exists(balance_file):
        if export_balance_history(client, SPREADSHEET_ID, balance_file):
            success_count += 1
    
    log(f"Export completed. {success_count} datasets exported successfully")
    return success_count > 0

# ============ Run as standalone ============
if __name__ == "__main__":
    log("Starting Google Sheets export")
    result = export_all_data()
    if result:
        log("✅ Export completed successfully")
    else:
        log("❌ Export failed")
