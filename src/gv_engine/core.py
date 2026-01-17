from dataclasses import dataclass
from typing import Iterable
import math


@dataclass(frozen=True)
class Constraint:
    """
    A measurable constraint with a capacity limit.
    """
    name: str
    value: float
    capacity: float
    weight: float = 1.0

    def utilization(self) -> float:
        return self.value / self.capacity


def gv_score(constraints: Iterable[Constraint], p: float = 2.0) -> float:
    """
    Compute a scalar strain score.
    0.0 = ideal
    higher = worse
    """
    total = 0.0
    for c in constraints:
        u = c.utilization()
        k = 8.0
        total += c.weight * math.log(1 + math.exp(k * (u - 1))) ** p
    return total
