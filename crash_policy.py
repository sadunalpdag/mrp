"""Shared Crash Bounce tracking horizon and new-candidate symbol policy."""
import re

TRACKING_SECONDS = 150 * 3600


def supported_symbol(symbol):
    """Only ASCII USDT symbols; never normalize an exchange symbol."""
    return (isinstance(symbol, str) and symbol.endswith('USDT')
            and re.fullmatch(r'[A-Z0-9_]{1,40}', symbol) is not None)
