"""
Minimal toy system for pressure-testing GV-style scalars.

This example intentionally avoids engine internals.
It exists to answer one question:
Do simple GV signals light up *before* failure in a tiny, inspectable system?
"""

from dataclasses import dataclass
import math
import random

# Optional plotting (safe for CI)
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


@dataclass
class ConstraintState:
    name: str
    value: float
    min_val: float
    max_val: float
    inflow: float
    base_leak_rate: float


class SimpleSystem:
    def __init__(self):
        self.step_idx = 0

        # Two coupled constraints
        self.states = {
            "A": ConstraintState(
                name="Memory",
                value=0.0,
                min_val=0.0,
                max_val=100.0,
                inflow=2.0,
                base_leak_rate=0.05,
            ),
            "B": ConstraintState(
                name="Cache",
                value=0.0,
                min_val=0.0,
                max_val=50.0,
                inflow=1.0,
                base_leak_rate=0.03,
            ),
        }

        self.history = {k: [] for k in self.states}
        self.scalars = {
            "saturation": [],
            "coupling_drift": [],
            "reversibility_loss": [],
        }

    def step(self):
        self.step_idx += 1

        # --- Update constraint A ---
        A = self.states["A"]
        A.value += A.inflow
        leak_A = A.base_leak_rate * (A.value - A.min_val)

        # Reversibility loss: leak weakens as A saturates
        saturation_A = A.value / A.max_val
        leak_A *= max(0.2, 1.0 - 0.8 * saturation_A)

        A.value = max(A.min_val, A.value - leak_A)

        # --- Update constraint B ---
        B = self.states["B"]
        B.value += B.inflow
        leak_B = B.base_leak_rate * (B.value - B.min_val)

        # Coupling: B leaks more as A fills
        coupling = 0.4 * saturation_A * B.value
        B.value = max(B.min_val, B.value - leak_B - coupling)

        # Record state history
        for k, s in self.states.items():
            self.history[k].append(s.value)

        self.compute_scalars()

    def compute_scalars(self):
        # 1. Saturation signal (max normalized fill)
        sat = max(s.value / s.max_val for s in self.states.values())
        self.scalars["saturation"].append(sat)

        # 2. Coupling drift (rolling correlation proxy)
        if len(self.history["A"]) > 10:
            a = self.history["A"][-10:]
            b = self.history["B"][-10:]
            mean_a = sum(a) / len(a)
            mean_b = sum(b) / len(b)

            num = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b))
            den = math.sqrt(
                sum((ai - mean_a) ** 2 for ai in a)
                * sum((bi - mean_b) ** 2 for bi in b)
            )
            drift = (num / den) ** 2 if den > 0 else 0.0
        else:
            drift = 0.0

        self.scalars["coupling_drift"].append(drift)

        # 3. Reversibility loss proxy
        # How far leak effectiveness has fallen from baseline
        A = self.states["A"]
        current_effective_leak = A.base_leak_rate * (
            1.0 - 0.8 * (A.value / A.max_val)
        )
        rev_loss = 1.0 - (current_effective_leak / A.base_leak_rate)
        self.scalars["reversibility_loss"].append(max(0.0, rev_loss))

    def run(self, steps=200, stop_inflow_at=None):
        for i in range(steps):
            if stop_inflow_at is not None and i >= stop_inflow_at:
                for s in self.states.values():
                    s.inflow = 0.0
            self.step()

    def plot(self):
        if not HAS_PLOT:
            print("matplotlib not available; skipping plot.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Constraint trajectories
        for k, vals in self.history.items():
            axes[0].plot(vals, label=k)
        axes[0].set_title("Constraint Trajectories")
        axes[0].legend()

        # Scalar signals
        for k, vals in self.scalars.items():
            axes[1].plot(vals, label=k)
        axes[1].set_title("GV-style Scalars")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    system = SimpleSystem()

    # Baseline run
    system.run(steps=200)

    # Uncomment to test recovery behavior:
    # system = SimpleSystem()
    # system.run(steps=200, stop_inflow_at=100)

    system.plot()

    print("Final scalar values:")
    for k, v in system.scalars.items():
        print(f"  {k}: {v[-1]:.3f}")
