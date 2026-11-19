"""Minimal CoinBeacon REST client for BTC-only 15m research.

Independent of removed CB2 scanners; read-only and quota guarded.
"""
import logging
import os
import time

LOG = logging.getLogger("btc_coinbeacon_client")
API = "https://api.coinbeacon.io"
RESERVE = 0

class MarketClient:
    def __init__(self, http):
        self.http = http
        self.key = (os.getenv("api2") or os.getenv("API2") or "").strip()
        self.daily_left = None
        self.minute_left = None
        self.usage_checked = 0

    def usage(self):
        if not self.key:
            return False
        if time.time() - self.usage_checked < 590 and self.daily_left is not None:
            return True
        self.usage_checked = time.time()
        try:
            r = self.http.get(API + "/public/usage", headers={"X-API-Key": self.key}, timeout=12)
            r.raise_for_status()
            usage = r.json()
            self.daily_left = (usage.get("today") or {}).get("creditsLeft")
            self.minute_left = (usage.get("thisMinute") or {}).get("creditsLeft")
            if self.daily_left is None or self.minute_left is None:
                return False
            self.daily_left = float(self.daily_left)
            self.minute_left = float(self.minute_left)
            return True
        except Exception as exc:
            self.daily_left = None
            LOG.warning("BTC CoinBeacon quota unavailable: %s", type(exc).__name__)
            return False

    def cb(self, path, cost, params=None):
        if not self.usage() or self.daily_left is None:
            return None
        if self.daily_left < RESERVE + cost or self.minute_left is None or self.minute_left < cost + 5:
            return None
        try:
            r = self.http.get(API + path, params=params,
                              headers={"X-API-Key": self.key}, timeout=15)
            if r.status_code == 429:
                self.daily_left = None
                return None
            r.raise_for_status()
            self.daily_left -= cost
            self.minute_left -= cost
            return r.json()
        except Exception as exc:
            LOG.warning("BTC CoinBeacon request unavailable: %s", type(exc).__name__)
            return None
