"""
Simple trajectory experiment for gv-engine.

Demonstrates:
- Decoupled vs coupled constraint growth
- GV-style scalar metrics
"""

import inspect
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


def clamp(x, lo=0.0, hi=CAPACITY):
    return max(lo, min(hi, float(x)))


def call_coupling_drift(fn, /, *args, **kwargs):
    """
    Call coupling_drift() safely across signature changes.

    - If the function expects prev_val (not prev_value), we rename it.
    - We drop any unexpected kwargs so the example never crashes.
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    # Back-compat rename (common mismatch)
    if "prev_value" in kwargs and "prev_value" not in params and "prev_val" in params:
        kwargs["prev_val"] = kwargs.pop("prev_value")

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


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
    Returns three scalar series:
    - satisfaction drift: how fast the "free capacity" is shrinking
    - coupling drift: wrapper around gv_engine.coupling_drift
    - reversibility proxy: how hard it would be to "undo" recent growth
    """
    sat_d = np.zeros_like(a)
    coup_d = np.zeros_like(a)
    rev_d = np.zeros_like(a)

    prev_a = float(a[0])
    prev_b = float(b[0])

    for t in range(1, len(a)):
        at = float(a[t])
        bt = float(b[t])

        # Satisfaction drift: shrinking of remaining headroom (simple proxy)
        headroom_prev = (CAPACITY - prev_a) + (CAPACITY - prev_b)
        headroom_now = (CAPACITY - at) + (CAPACITY - bt)
        sat_d[t] = headroom_prev - headroom_now  # >0 means consuming headroom

        # Coupling drift: call engine function safely
        # We pass both prev_value and prev_val style kwargs via wrapper logic.
        coup_d[t] = call_coupling_drift(
            coupling_drift,
            a=at,
            b=bt,
            prev_value=prev_a,   # old name used in your earlier example
            prev_a=prev_a,       # some signatures may use prev_a/prev_b
            prev_b=prev_b,
            max_val=CAPACITY,    # some signatures may want max_val/capacity
            capacity=CAPACITY,
            coupling=COUPLING,
        )

        # Reversibility proxy: larger recent step = harder to undo (simple proxy)
        rev_d[t] = abs(at - prev_a) + abs(bt - prev_b)

        prev_a, prev_b = at, bt

    return sat_d, coup_d, rev_d


def try_gv_score(a: np.ndarray, b: np.ndarray):
    """
    Optional: attempt to compute gv_score if the API matches.
    If gv_score signature differs, we just return None.
    """
    try:
        # Common pattern: make constraints and score
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

    # Plot trajectories
    t = np.arange(STEPS)

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

    # Plot scalars
    plt.figure()
    plt.plot(t, sat_d_dec, label="sat_d (decoupled)")
    plt.plot(t, sat_d_cpl, label="sat_d (coupled)")
    plt.title("Satisfaction drift (proxy)")
    plt.xlabel("Step")
    plt.ylabel("sat_d")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t, coup_d_dec, label="coupling_drift (decoupled)")
    plt.plot(t, coup_d_cpl, label="coupling_drift (coupled)")
    plt.title("Coupling drift (engine)")
    plt.xlabel("Step")
    plt.ylabel("coupling_drift")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t, rev_d_dec, label="rev_d (decoupled)")
    plt.plot(t, rev_d_cpl, label="rev_d (coupled)")
    plt.title("Reversibility proxy")
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
