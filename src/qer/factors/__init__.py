"""Factor library.

Importing this package registers every factor in ``FACTORS`` (see ``base``).
Price/volume factors work on the current data; size needs shares outstanding;
value and quality need ingested fundamentals.
"""
# NEW

# Import each module for its registration side-effect.
from qer.factors import (  # noqa: F401
    liquidity,
    momentum,
    quality,
    reversal,
    size,
    skewness,
    value,
    volatility,
)
from qer.factors.base import Factor, all_factors, compute_factor_panel, get_factor, register

__all__ = [
    "Factor",
    "all_factors",
    "compute_factor_panel",
    "get_factor",
    "register",
    "FACTORS",
]

from qer.factors.base import FACTORS  # noqa: E402
