"""
Experiment 01 — Baseline Coupled vs Decoupled

Purpose:
Establish a clean control experiment for the GV engine by comparing
constraint trajectories and scalar metrics under:

1) Coupled dynamics
2) Decoupled (independent) dynamics

This file is intentionally verbose and explicit.
It is the reference experiment all future experiments compare against.
"""

import math

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# -----------------------------
# Constraint system definition
# -----------------------------

class ConstraintSystem:
    def __init__(
        self,
        capacity_a=50.0,
        capacity_b=50.0,
        coupling_strength=0.0,
        leak_rate=0.01,
    ):
        self.capacity_a = capacity_a
        self.capacity_b = capacity_b
        self.coupling_strength = coupling_strength
        self.leak_rate = leak_rate

        self.a = 1.0
        self.b = 1.0

        self.history_a = []
        self.history_b = []

        self.scalars = {
            "saturation": [],
            "coupling_drift": [],
            "reversibility_loss": [],
        }

    def step(self):
        # Independent growth
        da = (self.capacity_a - self.a) * 0.05
        db = (self.capacity_b - self.b) * 0.05

        # Coupling term
        coupling = self.coupling_strength * (self.a - self.b)

        # Apply dynamics
        self.a += da - coupling
        self.b += db + coupling

        # Leakage
        self.a -= self.leak_rate * self.a
        self.b -= self.leak_rate * self.b

        # Record trajectories
        self.history_a.append(self.a)
        self.history_b.append(self.b)

        # Scalars
        self._compute_scalars(coupling)

    def _compute_scalars(self, coupling):
        # Saturation: how close we are to capacity
        sat = 0.5 * (
            self.a / self.capacity_a +
            self.b / self.capacity_b
        )

        # Coupling drift: magnitude of entanglement
        drift = min(abs(coupling) * 10.0, 1.0)

        # Reversibility loss: monotonic entropy-like cost
        rev = 1.0 - math.exp(-0.01 * len(self.history_a))

        self.scalars["saturation"].append(sat)
        self.scalars["coupling_drift"].append(drift)
        self.scalars["reversibility_loss"].append(rev)

    def run(self, steps=200):
        for _ in range(steps):
            self.step()


# -----------------------------
# Experiment runner
# -----------------------------

def run_experiment(coupled=True):
    system = ConstraintSystem(
        coupling_strength=0.02 if coupled else 0.0
    )
    system.run()
    return system


# -----------------------------
# Plotting
# -----------------------------

def plot_results(system, title_suffix):
    if not HAS_PLOT:
        print("matplotlib not available; skipping plot.")
        return

    x = range(len(system.history_a))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Trajectories
    ax1.plot(x, system.history_a, label="A")
    ax1.plot(x, system.history_b, label="B")
    ax1.set_title(f"Constraint Trajectories ({title_suffix})")
    ax1.legend()

    # Scalars
    ax2.plot(system.scalars["saturation"], label="saturation")
    ax2.plot(system.scalars["coupling_drift"], label="coupling_drift")
    ax2.plot(system.scalars["reversibility_loss"], label="reversibility_loss")
    ax2.set_title(f"GV-style Scalars ({title_suffix})")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    coupled_system = run_experiment(coupled=True)
    decoupled_system = run_experiment(coupled=False)

    plot_results(coupled_system, "coupled")
    plot_results(decoupled_system, "decoupled")

    print("Final scalar values (coupled):")
    for k, v in coupled_system.scalars.items():
        print(f"  {k}: {v[-1]:.3f}")

    print("\nFinal scalar values (decoupled):")
    for k, v in decoupled_system.scalars.items():
        print(f"  {k}: {v[-1]:.3f}")
