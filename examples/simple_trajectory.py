"""
INTERPRETATION GUIDE (plain English)

This script compares two systems:
1) Decoupled constraints (A and B grow independently)
2) Coupled constraints (A and B exchange growth to stay balanced)

Key ideas measured here are NOT performance or efficiency,
but pressure, optionality, and survivability.

Scalars:

sat_d (satisfaction drift, normalized):
- Measures how quickly total remaining headroom is consumed.
- High values mean the system is still using available freedom.
- Drops to zero when the system has exhausted optionality.
- Coupled systems tend to exhaust headroom earlier.

coup_d (coupling drift, engine):
- Measures instantaneous imbalance pressure between constraints.
- High values indicate latent force trying to equalize the system.
- This is NOT motion; it is stored tension.
- Goes to zero when symmetry or saturation is reached.

cumulative coupling drift:
- Integral of coupling drift over time.
- Represents total hidden work required to maintain balance.
- Decoupled systems defer this cost and accumulate it.
- Coupled systems pay it continuously and keep it low.

rev_d (reversibility proxy, normalized):
- Measures how hard it would be to undo the most recent change.
- High values mean the system still has escape routes.
- Zero means the system is effectively locked in.
- Coupled systems lock in earlier; decoupled systems remain flexible longer.

Takeaway:
Coupling trades long-term optionality for short-term coherence.
Decoupling preserves optionality at the cost of accumulating tension.
"""

"""
Simple trajectory experiment for gv-engine.

Demonstrates:
- Decoupled vs coupled constraint growth
- GV-style scalar metrics
- Normalized proxies + cumulative coupling drift (hidden work)
"""

import numpy as np
import matplotlib.pyplot as plt

from gv_engine import Constraint, gv_score, coupling_drift


# ------------------------------
# Parameters
# ------------------------------
STEPS = 200
CAPACITY = 50.0
INFLOW_A = 0.6
INFLOW_B = 0.4
COUPLING = 0.25

# coupling_drift signature: (a: float, b: float, strength: float = 0.01) -> float
COUPLING_DRIFT_STRENGTH = 0.01


def clamp(x, lo=0.0, hi=CAPACITY):
    return max(lo, min(hi, float(x)))


def simulate_decoupled(steps: int):
    """Two independent constraints filling toward capacity."""
    a = np.zeros(steps, dtype=float)
    b = np.zeros(steps, dtype=float)
    for t in range(1, steps):
        a[t] = clamp(a[t - 1] + INFLOW_A)
        b[t] = clamp(b[t - 1] + INFLOW_B)
    return a, b


def simulate_coupled(steps: int):
    """
    Two constraints with simple symmetric coupling:
    each step, a fraction of the delta tries to equalize A and B.
    """
    a = np.zeros(steps, dtype=float)
    b = np.zeros(steps, dtype=float)
    for t in range(1, steps):
        # raw inflow
        a_next = a[t - 1] + INFLOW_A
        b_next = b[t - 1] + INFLOW_B

        # coupling exchange (equalize)
        diff = a_next - b_next
        a_next -= COUPLING * diff
        b_next += COUPLING * diff

        a[t] = clamp(a_next)
        b[t] = clamp(b_next)
    return a, b


def compute_scalars(a: np.ndarray, b: np.ndarray):
    """
    Returns three scalar series (all normalized, scale-free):
    - sat_d: shrink of remaining headroom per step (normalized by total headroom 2*CAPACITY)
    - coup_d: gv_engine.coupling_drift(a, b, strength)
    - rev_d: magnitude of recent change per step (normalized by 2*CAPACITY)
    """
    sat_d = np.zeros_like(a)
    coup_d = np.zeros_like(a)
    rev_d = np.zeros_like(a)

    prev_a = float(a[0])
    prev_b = float(b[0])

    for t in range(1, len(a)):
        at = float(a[t])
        bt = float(b[t])

        headroom_prev = (CAPACITY - prev_a) + (CAPACITY - prev_b)
        headroom_now = (CAPACITY - at) + (CAPACITY - bt)

        # normalize by total headroom range (2*CAPACITY)
        sat_d[t] = (headroom_prev - headroom_now) / (2.0 * CAPACITY)

        # correct call for your current function signature
        coup_d[t] = coupling_drift(at, bt, strength=COUPLING_DRIFT_STRENGTH)

        # normalize by total possible per-step change scale (2*CAPACITY)
        rev_d[t] = (abs(at - prev_a) + abs(bt - prev_b)) / (2.0 * CAPACITY)

        prev_a, prev_b = at, bt

    return sat_d, coup_d, rev_d


def try_gv_score(a: np.ndarray, b: np.ndarray):
    """
    Optional: compute gv_score if the Constraint API matches.
    If not, return None (so the script still runs cleanly).
    """
    try:
        cA = Constraint(value=float(a[-1]), max_val=CAPACITY, name="A")
        cB = Constraint(value=float(b[-1]), max_val=CAPACITY, name="B")
        return gv_score([cA, cB])
    except Exception:
        return None


def main():
    # Simulate
    a_dec, b_dec = simulate_decoupled(STEPS)
    a_cpl, b_cpl = simulate_coupled(STEPS)

    # Scalars
    sat_d_dec, coup_d_dec, rev_d_dec = compute_scalars(a_dec, b_dec)
    sat_d_cpl, coup_d_cpl, rev_d_cpl = compute_scalars(a_cpl, b_cpl)

    # NEW: cumulative coupling drift (integral / hidden work)
    cum_coup_dec = np.cumsum(coup_d_dec)
    cum_coup_cpl = np.cumsum(coup_d_cpl)

    t = np.arange(STEPS)

    # 1) Trajectories
    plt.figure()
    plt.plot(t, a_dec, label="A (decoupled)")
    plt.plot(t, b_dec, label="B (decoupled)")
    plt.plot(t, a_cpl, label="A (coupled)")
    plt.plot(t, b_cpl, label="B (coupled)")
    plt.title("Constraint trajectories")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    # 2) Satisfaction drift (normalized)
    plt.figure()
    plt.plot(t, sat_d_dec, label="sat_d (decoupled)")
    plt.plot(t, sat_d_cpl, label="sat_d (coupled)")
    plt.title("Satisfaction drift (normalized proxy)")
    plt.xlabel("Step")
    plt.ylabel("sat_d")
    plt.legend()
    plt.tight_layout()

    # 3) Coupling drift (engine)
    plt.figure()
    plt.plot(t, coup_d_dec, label="coupling_drift (decoupled)")
    plt.plot(t, coup_d_cpl, label="coupling_drift (coupled)")
    plt.title("Coupling drift (engine)")
    plt.xlabel("Step")
    plt.ylabel("coupling_drift")
    plt.legend()
    plt.tight_layout()

    # 4) Cumulative coupling drift (hidden work)
    plt.figure()
    plt.plot(t, cum_coup_dec, label="cumulative coupling_drift (decoupled)")
    plt.plot(t, cum_coup_cpl, label="cumulative coupling_drift (coupled)")
    plt.title("Cumulative coupling drift (hidden work)")
    plt.xlabel("Step")
    plt.ylabel("cumulative_drift")
    plt.legend()
    plt.tight_layout()

    # 5) Reversibility proxy (normalized)
    plt.figure()
    plt.plot(t, rev_d_dec, label="rev_d (decoupled)")
    plt.plot(t, rev_d_cpl, label="rev_d (coupled)")
    plt.title("Reversibility proxy (normalized)")
    plt.xlabel("Step")
    plt.ylabel("rev_d")
    plt.legend()
    plt.tight_layout()

    # Optional gv_score printout
    gv_dec = try_gv_score(a_dec, b_dec)
    gv_cpl = try_gv_score(a_cpl, b_cpl)
    if gv_dec is not None or gv_cpl is not None:
        print(f"gv_score (decoupled end): {gv_dec}")
        print(f"gv_score (coupled end):   {gv_cpl}")

    plt.show()


if __name__ == "__main__":
    main()
