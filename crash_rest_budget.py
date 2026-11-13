"""Process-wide conservative Binance USD-M REST scheduler.

All Crash V2 research and live broker calls share this one in-process IP
weight view. Each response updates the actual server weight. Delayed research
never uses capacity reserved for account/TP work. A second Render service on
the same egress IP cannot be controlled, so limits cannot be guaranteed.
"""
import math
import threading
import time

# Well below the usual 2400/minute; absolute ceilings for ALL local calls.
# Leave generous headroom for other clients on a shared Render egress IP.
# These are conservative software caps, NOT exchange guarantees.
RESEARCH_CEILING = 180
# PAPER-ONLY HBREAK/API2 is strictly lower than Crash V2 research.
PAPER_CEILING = 120
ACCOUNT_CEILING = 1400
PROTECT_CEILING = 1750
PACED_SECONDS = {"paper": 20.0, "research": 20.0, "account": 3.0, "protect": 0.5}


class RateDeferred(RuntimeError):
    def __init__(self, kind, seconds, reason):
        self.kind, self.seconds, self.reason = kind, max(0, float(seconds)), reason
        super().__init__(f"BINANCE_IP_BUDGET_{reason}: retry_after={self.seconds:.1f}s")


def estimated_weight(path, params=None, method=None):
    """Account for documented USD-M IP weight; still fail closed on unknowns.

    New futures orders consume ORDER-COUNT limits but zero REQUEST_WEIGHT.
    The order counter and exchange 429/418 protections are separate from
    this IP weight estimator and are never bypassed.
    """
    params = params or {}
    if path == "/fapi/v1/order":
        if method == "POST":
            return 0  # Binance IP weight 0, ORDERS 10s/1m counters still apply.
        if method in ("GET", "DELETE"):
            return 1
    if path == "/fapi/v1/leverage" and method == "POST":
        return 1
    if path in ("/fapi/v1/klines", "/api/v3/klines"):
        try:
            size = int(params.get("limit", 500))
        except (TypeError, ValueError):
            size = 1500
        if size < 100:
            return 1
        if size < 500:
            return 2
        if size <= 1000:
            return 5
        return 10
    if path in ("/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders"):
        return 1 if params.get("symbol") else 40
    if path in ("/fapi/v3/positionRisk", "/fapi/v2/positionRisk"):
        return 5
    if path == "/fapi/v1/income":
        return 30  # Binance income-history documented IP weight.
    if path == "/fapi/v1/positionSide/dual":
        return 30  # conservative
    if path.startswith("/futures/data/"):
        return 5  # separate endpoint limits may apply
    return 5  # a conservative default


def priority_for(method, path, params=None):
    if path == "/fapi/v1/order":
        # TP placement/verification and cancelling aged entries outrank entry preflight.
        if method == "DELETE" or (method == "POST" and (params or {}).get("side") == "SELL"):
            return "protect"
        if method == "GET":
            return "protect"
    return "account"


class IpBudget:
    def __init__(self, clock=time.time, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.lock = threading.RLock()
        self.minute = None
        self.estimated_used = 0.0
        self.server_used = 0.0
        self.blocked_until = 0.0
        self.next_per_kind = {k: 0.0 for k in PACED_SECONDS}
        self.next_global = 0.0
        self.last_notice = {}
        self.research_next_allowed = 0.0
        self.endpoint_counts = {}
        self.endpoint_weights = {}
        self.endpoint_deferred = {}
        self.last_server_peak = 0.0

    def _roll(self, now):
        minute = int(now // 60)
        if minute != self.minute:
            if self.minute is not None and (self.endpoint_counts or self.endpoint_deferred):
                items = sorted(self.endpoint_counts, key=lambda k: self.endpoint_weights[k], reverse=True)
                top = ",".join(f"{k}:{self.endpoint_counts[k]}calls/{self.endpoint_weights[k]}w" for k in items[:12])
                deferred = ",".join(f"{k}:{v}" for k, v in sorted(self.endpoint_deferred.items()))
                print(f"[CRASH REST ENDPOINT AUDIT 1M] minute={self.minute}; admitted={sum(self.endpoint_counts.values())}; estimated_weight={sum(self.endpoint_weights.values())}; server_peak={int(self.last_server_peak)}; top={top or '-'}; deferred={deferred or '-'}; scope=local_admissions_not_confirmed_http", flush=True)
            self.endpoint_counts = {}
            self.endpoint_weights = {}
            self.endpoint_deferred = {}
            self.last_server_peak = 0.0
            self.minute = minute
            self.estimated_used = 0.0
            self.server_used = 0.0

    def _notice(self, kind, reason, wait, now):
        key = (kind, reason)
        if now >= self.last_notice.get(key, 0):
            print(f"[CRASH IP BUDGET WAIT] tier={kind}; reason={reason}; "
                  f"retry_seconds={int(wait)+1}; server_weight={int(self.server_used)}; "
                  f"estimated_weight={int(self.estimated_used)}", flush=True)
            self.last_notice[key] = now + 60

    def reserve(self, kind, weight, endpoint="unknown"):
        if kind not in PACED_SECONDS or not 0 <= weight <= 40:
            raise ValueError("INVALID_IP_BUDGET_REQUEST")
        ceiling = {"paper": PAPER_CEILING, "research": RESEARCH_CEILING, "account": ACCOUNT_CEILING,
                   "protect": PROTECT_CEILING}[kind]
        while True:
            with self.lock:
                now = self.clock()
                self._roll(now)
                if now < self.blocked_until:
                    wait = self.blocked_until - now
                    self._notice(kind, "EXCHANGE_COOLDOWN", wait, now)
                    raise RateDeferred(kind, wait, "EXCHANGE_COOLDOWN")
                projected = max(self.estimated_used, self.server_used) + weight
                if projected > ceiling:
                    key = kind + ":" + endpoint
                    self.endpoint_deferred[key] = self.endpoint_deferred.get(key, 0) + 1
                    wait = (int(now // 60)+1)*60 + 5 - now
                    self._notice(kind, "IP_WEIGHT_SOFT_CEILING", wait, now)
                    raise RateDeferred(kind, wait, "IP_WEIGHT_SOFT_CEILING")
                # Once external/server IP usage is high, stop background
                # research for the rest of the minute, keeping TP capacity.
                if kind in ("paper", "research") and self.server_used >= ceiling:
                    wait = (int(now // 60)+1)*60 + 5 - now
                    self._notice(kind, "RESEARCH_SERVER_WEIGHT", wait, now)
                    raise RateDeferred(kind, wait, "RESEARCH_SERVER_WEIGHT")
                remaining = max(self.next_per_kind[kind], self.next_global) - now
                if remaining <= 0:
                    key = kind + ":" + endpoint
                    self.endpoint_counts[key] = self.endpoint_counts.get(key, 0) + 1
                    self.endpoint_weights[key] = self.endpoint_weights.get(key, 0) + weight
                    self.estimated_used = projected
                    self.next_per_kind[kind] = now + PACED_SECONDS[kind]
                    self.next_global = now + 0.15
                    return
            # Never hold the shared mutex while pacing. TP requests can bypass
            # a queued research sleep and still consume only safe capacity.
            self.sleep(min(remaining, 10))

    def observe(self, headers, status_code, retry_after=0):
        with self.lock:
            now = self.clock()
            self._roll(now)
            try:
                used = float((headers or {}).get("X-MBX-USED-WEIGHT-1M", 0))
                if math.isfinite(used) and used >= 0:
                    self.server_used = max(self.server_used, used)
                    self.last_server_peak = max(self.last_server_peak, used)
            except (ValueError, TypeError):
                pass
            if status_code in (418, 429):
                try:
                    delay = float(retry_after)
                    if not math.isfinite(delay) or delay < 0:
                        delay = 0
                except (TypeError, ValueError):
                    delay = 0
                # Shared across research and live broker. Never bypass a real ban.
                self.blocked_until = max(self.blocked_until, now + max(
                    900 if status_code == 418 else 120, delay) + 5)

    def stats(self):
        with self.lock:
            self._roll(self.clock())
            return {"server_weight": int(self.server_used),
                    "reserved_weight": int(self.estimated_used),
                    "blocked_until": self.blocked_until}


BUDGET = IpBudget()
