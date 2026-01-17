from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, List
import math

import matplotlib.pyplot as plt


# -------------------------
# Static constraint scoring
# -------------------------

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
        penalty = max(0.0, u - 1.0)
        total += c.weight * (penalty ** p) * math.exp(k * penalty)
    return total


# -------------------------
# Dynamic simulation engine
# -------------------------

class SimpleSystem:
    def __init__(self, coupled: bool = True):
        self.coupled = coupled

        self.A: List[float] = [1.0]
        self.B: List[float] = [0.5]

        self.scalars: Dict[str, List[float]] = {
            "saturation": [0.0],
            "coupling_drift": [0.0],
            "reversibility_loss": [0.0],
        }

    def step(self):
        a, b = self.A[-1], self.B[-1]

        if self.coupled:
            da = 0.4 * (1 - a / 50) + 0.02 * b
            db = 0.3 * (1 - b / 50) + 0.01 * a
        else:
            da = 0.4 * (1 - a / 50)
            db = 0.3 * (1 - b / 50)

        a2 = a + da
        b2 = b + db

        self.A.append(a2)
        self.B.append(b2)

        saturation = 1 - math.exp(-a2 / 40)
        reversibility = 1 - math.exp(-b2 / 60)

        expected_db = 0.3 * (1 - b / 50)
        coupling_drift = abs(db - expected_db)
        coupling_drift = min(1.0, coupling_drift)

        self.scalars["saturation"].append(saturation)
        self.scalars["reversibility_loss"].append(reversibility)
        self.scalars["coupling_drift"].append(coupling_drift)

    def run(self, steps: int = 200):
        for _ in range(steps):
            self.step()

    def plot(self, title_suffix: str = ""):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(self.A, label="A")
        axes[0].plot(self.B, label="B")
        axes[0].set_title(f"Constraint Trajectories {title_suffix}")
        axes[0].legend()

        for k, v in self.scalars.items():
            axes[1].plot(v, label=k)

        axes[1].set_title(f"GV-style Scalars {title_suffix}")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def print_final_scalars(self, label: str = ""):
        print(f"\nFinal scalar values {label}:")
        for k, v in self.scalars.items():
            print(f"  {k}: {v[-1]:.3f}")
