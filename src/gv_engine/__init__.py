"""
gv_engine

A minimal constraint-based engine for computing GV-style scalar risk and coupling metrics.

Public API:
- Constraint dataclass
- gv_score(...) scalar strain score
- coupling_drift(...) coupling deviation scalar
"""

from .core import Constraint, gv_score
from .coupling import coupling_drift

__all__ = [
    "Constraint",
    "gv_score",
    "coupling_drift",
]
