"""
Simple trajectory experiment for gv-engine.

Demonstrates:
- Decoupled vs coupled constraint growth
- GV-style scalar metrics
"""

import numpy as np
import matplotlib.pyplot as plt

from gv_engine import Constraint, gv_score, coupling_drift


# -------------------------
# Parameters
# -------------------------
STEPS = 200
CAPACITY = 50.0
INFLOW_A = 0.6
INFLOW_B = 0.4
COUPLING = 0.25


# -------------------------
# Decoupled dynamics
# -------------------------
def run_decoupled():
    a_vals = []
    b_vals = []

    a = 2.0
    b = 2.0

    for _ in range(STEPS):
        a += INFLOW_A * (1.0 - a / CAPACITY)
        b += INFLOW_B * (1.0 - b / CAPACITY)
        a_vals.append(a)
        b_vals.append(b)

    return np.array(a_vals), np.array(b_vals)


# -------------------------
# Coupled dynamics
# -------------------------
def run_coupled():
    a_vals = []
    b_vals = []

    a = 2.0
    b = 2.0

    for _ in range(STEPS):
        a += INFLOW_A * (1.0 - a / CAPACITY) + COUPLING * (b / CAPACITY)
        b += INFLOW_B * (1.0 - b / CAPACITY) + COUPLING * (a / CAPACITY)
        a_vals.append(a)
        b_vals.append(b)

    return np.array(a_vals), np.array(b_vals)


# -------------------------
# Scalar metrics
# -------------------------
def compute_scalars(a_vals, b_vals):
    saturation = []
    coupling_vals = []
    reversibility = []

    prev_a = a_vals[0]

    for a, b in zip(a_vals, b_vals):
        ca = Constraint("A", a, CAPACITY)
        cb = Constraint("B", b, CAPACITY)

        saturation.append(gv_score([ca, cb]))

        coupling_vals.append(
            coupling_drift(
                prev_value=prev_a,
                curr_value=a,
                inflow=INFLOW_A,
                base_leak_rate=0.0,
                min_val=0.0,
                max_val=CAPACITY,
            )
        )

        reversibility.append(1.0 - a / CAPACITY)
        prev_a = a

    return (
        np.array(saturation),
        np.array(coupling_vals),
        np.array(reversibility),
    )


# -------------------------
# Run experiments
# -------------------------
a_dec, b_dec = run_decoupled()
a_coup, b_coup = run_coupled()

sat_d, coup_d, rev_d = compute_scalars(a_dec, b_dec)
sat_c, coup_c, rev_c = compute_scalars(a_coup, b_coup)


# -------------------------
# Plot
# -------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(a_coup, label="A")
axs[0, 0].plot(b_coup, label="B")
axs[0, 0].set_title("Constraint Trajectories (coupled)")
axs[0, 0].legend()

axs[1, 0].plot(sat_c, label="saturation")
axs[1, 0].plot(coup_c, label="coupling_drift")
axs[1, 0].plot(rev_c, label="reversibility_loss")
axs[1, 0].set_title("GV-style Scalars (coupled)")
axs[1, 0].legend()

axs[0, 1].plot(a_dec, label="A")
axs[0, 1].plot(b_dec, label="B")
axs[0, 1].set_title("Constraint Trajectories (decoupled)")
axs[0, 1].legend()

axs[1, 1].plot(sat_d, label="saturation")
axs[1, 1].plot(coup_d, label="coupling_drift")
axs[1, 1].plot(rev_d, label="reversibility_loss")
axs[1, 1].set_title("GV-style Scalars (decoupled)")
axs[1, 1].legend()

plt.tight_layout()
plt.show()
