#!/usr/bin/env python3
"""
Example script demonstrating Google Sheets integration usage
"""

import os
import sys
from sheets_integration import (
    get_sheets_client,
    export_all_data,
    export_trades_to_sheet,
    export_performance_report,
    log
)

def main():
    """Main function to demonstrate Google Sheets integration"""
    
    log("=" * 60)
    log("Google Sheets Integration Example")
    log("=" * 60)
    
    # Check environment variables
    spreadsheet_id = os.getenv("SPREADSHEET_ID")
    credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    
    if not spreadsheet_id:
        log("ERROR: SPREADSHEET_ID environment variable not set")
        log("Please set it using: export SPREADSHEET_ID=your-spreadsheet-id")
        return 1
    
    if not os.path.exists(credentials_file):
        log(f"ERROR: Credentials file not found: {credentials_file}")
        log("Please follow the setup guide in GOOGLE_SHEETS_SETUP.md")
        return 1
    
    log(f"✓ Credentials file found: {credentials_file}")
    log(f"✓ Spreadsheet ID: {spreadsheet_id}")
    
    # Test connection
    log("\n" + "=" * 60)
    log("Testing Google Sheets connection...")
    log("=" * 60)
    
    client = get_sheets_client()
    if not client:
        log("ERROR: Failed to connect to Google Sheets")
        return 1
    
    log("✓ Successfully connected to Google Sheets")
    
    # Export all data
    log("\n" + "=" * 60)
    log("Exporting all data to Google Sheets...")
    log("=" * 60)
    
    success = export_all_data()
    
    if success:
        log("\n" + "=" * 60)
        log("✅ SUCCESS! Data exported to Google Sheets")
        log(f"View your data at: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
        log("=" * 60)
        return 0
    else:
        log("\n" + "=" * 60)
        log("❌ FAILED: Could not export data")
        log("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
