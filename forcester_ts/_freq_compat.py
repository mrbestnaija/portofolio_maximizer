"""
Pandas frequency-alias compatibility shim.

Pandas >= 2.2 deprecated uppercase frequency aliases (e.g. 'H', 'T', 'S',
'L', 'U', 'N') in favour of their lowercase equivalents ('h', 'min', 's',
'ms', 'us', 'ns').  This module provides a single normalizer applied wherever
``freqstr`` or ``inferred_freq`` values enter the forecasting layer.
"""

from typing import Optional

# Deprecated alias -> replacement (pandas >= 2.2)
_DEPRECATED_ALIASES = {
    "H": "h",
    "T": "min",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
    # Multi-period variants (e.g. "2H" -> "2h")
}


def normalize_freq(freq: Optional[str]) -> Optional[str]:
    """Return *freq* with deprecated uppercase aliases replaced.

    Handles bare aliases ('H' -> 'h') and prefixed variants ('2H' -> '2h').
    Returns ``None`` unchanged.
    """
    if not freq:
        return freq

    # Fast path: most common case is already fine
    if freq in _DEPRECATED_ALIASES:
        return _DEPRECATED_ALIASES[freq]

    # Prefixed variants like '2H', '4T', '30S'
    # Split into numeric prefix + alpha suffix
    i = 0
    while i < len(freq) and (freq[i].isdigit() or freq[i] == '.'):
        i += 1
    if i > 0:
        prefix, suffix = freq[:i], freq[i:]
        if suffix in _DEPRECATED_ALIASES:
            return prefix + _DEPRECATED_ALIASES[suffix]

    return freq
