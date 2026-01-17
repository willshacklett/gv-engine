from dataclasses import dataclass
from typing import Iterable
import math


@dataclass
class Constraint:
    name: str
    value: float
    capacity: float
    weight: float = 1.0

    def utilization(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(self.value / self.capacity, 1.0)


def gv_score(constraints: Iterable[Constraint], p: float = 2.0) -> float:
    """
    Scalar strain score.
    0.0 = ideal
    higher = worse
    """
    total = 0.0
    for c in constraints:
        u = c.utilization()
        k = 8.0
        penalty = (math.exp(k * u) - 1) / (math.exp(k) - 1)
        total += c.weight * (penalty ** p)
    return total
