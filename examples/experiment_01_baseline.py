import matplotlib.pyplot as plt
import numpy as np

from gv_engine.core import Constraint, gv_score
from gv_engine.coupling import coupling_drift


def run_simulation(coupled: bool):
    steps = 200

    A = np.zeros(steps)
    B = np.zeros(steps)

    A[0] = 2.0
    B[0] = 1.0

    inflow_A = 1.2
    inflow_B = 0.9
    leak = 0.02

    saturation = []
    coupling_vals = []
    reversibility = []

    for t in range(1, steps):
        if coupled:
            B[t] = B[t - 1] + inflow_B + 0.03 * A[t - 1] - leak * B[t - 1]
        else:
            B[t] = B[t - 1] + inflow_B - leak * B[t - 1]

        A[t] = A[t - 1] + inflow_A - leak * A[t - 1]

        cA = Constraint("A", A[t], 50)
        cB = Constraint("B", B[t], 50)

        saturation.append((cA.utilization() + cB.utilization()) / 2)

        drift = coupling_drift(
            prev_value=B[t - 1],
            curr_value=B[t],
            inflow=inflow_B,
            base_leak_rate=leak,
            min_val=0,
            max_val=50,
        )
        coupling_vals.append(drift if coupled else 0.0)

        reversibility.append(1 - math.exp(-0.01 * t))

    title = "coupled" if coupled else "decoupled"

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(A, label="A")
    ax[0].plot(B, label="B")
    ax[0].set_title(f"Constraint Trajectories ({title})")
    ax[0].legend()

    ax[1].plot(saturation, label="saturation")
    ax[1].plot(coupling_vals, label="coupling_drift")
    ax[1].plot(reversibility, label="reversibility_loss")
    ax[1].set_title(f"GV-style Scalars ({title})")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import math

    run_simulation(coupled=True)
    run_simulation(coupled=False)
