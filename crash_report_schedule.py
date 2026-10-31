"""Paris calendar schedule; independent of process uptime, DST-safe."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
ZONE = ZoneInfo('Europe/Paris')
HOUR = 21


def slot(now, tomorrow=False):
    day = datetime.fromtimestamp(now, ZONE).date() + timedelta(days=int(tomorrow))
    return datetime(day.year, day.month, day.day, HOUR, tzinfo=ZONE).timestamp()


def prepare(report, now):
    if report.get('schedule_version') != 2:
        report.update(schedule_version=2, next_due_at=slot(now))
        report.setdefault('last_success_at', now-86400)
    return report


def success(report, now):
    report.update(last_success_at=now, next_due_at=slot(now, tomorrow=True), schedule_version=2)
    report.pop('pending',None)
    report.pop('retry_after',None)
