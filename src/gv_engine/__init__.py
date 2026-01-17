"""
gv_engine

Core constraint-based engine for computing GV-style scalar risk and coupling metrics.
"""

from .core import Constraint, gv_score
from .coupling import coupling_drift

__all__ = [
    "Constraint",
    "gv_score",
    "coupling_drift",
]
