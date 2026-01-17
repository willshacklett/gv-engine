import numpy as np
import matplotlib.pyplot as plt

from gv_engine import Constraint, gv_score, coupling_drift


def run(coupled: bool = True):
    steps = 200

    A = np.zeros(steps)
    B = np.zeros(steps)

    inflow_A = 0.6
    inflow_B = 0.6

    leak_A = 0.02
    leak_B = 0.02

    max_val = 100.0

    sat = []
    drift = []
    rev = []

    for t in range(1, steps):
        if coupled:
            B[t] = B[t-1] + inflow_B + 0.05 * A[t-1] - leak_B * B[t-1]
        else:
            B[t] = B[t-1] + inflow_B - leak_B * B[t-1]

        A[t] = A[t-1] + inflow_A - leak_A * A[t-1]

        cA = Constraint("A", A[t], max_val)
        cB = Constraint("B", B[t], max_val)

        sat.append(cA.utilization())
        drift.append(
            coupling_drift(B[t-1], B[t], inflow_B, leak_B, 0, max_val)
        )
        rev.append(1.0 - gv_score([cA, cB]))

    label = "coupled" if coupled else "decoupled"

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(A, label="A")
    ax[0].plot(B, label="B")
    ax[0].set_title(f"Constraint Trajectories ({label})")
    ax[0].legend()

    ax[1].plot(sat, label="saturation")
    ax[1].plot(drift, label="coupling_drift")
    ax[1].plot(rev, label="reversibility_loss")
    ax[1].set_title(f"GV-style Scalars ({label})")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run(coupled=True)
    run(coupled=False)
