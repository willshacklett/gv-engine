import numpy as np
import matplotlib.pyplot as plt

from gv_engine import Constraint, gv_score, coupling_drift


def run(coupled: bool = True):
    steps = 200
    cap = 50.0

    a = 2.0
    b = 1.5

    A, B = [], []
    saturation = []
    coupling = []
    reversibility = []

    for t in range(steps):
        if coupled:
            drift = coupling_drift(a, b)
            a -= drift
            b += drift
            coupling.append(abs(drift))
        else:
            coupling.append(0.0)

        a += 0.6 * (1 - a / cap)
        b += 0.5 * (1 - b / cap)

        A.append(a)
        B.append(b)

        constraints = [
            Constraint("A", a, cap),
            Constraint("B", b, cap),
        ]

        score = gv_score(constraints)
        saturation.append(min(1.0, score / 10))
        reversibility.append(1 - math.exp(-t / 120))

    title = "coupled" if coupled else "decoupled"

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(A, label="A")
    ax[0].plot(B, label="B")
    ax[0].set_title(f"Constraint Trajectories ({title})")
    ax[0].legend()

    ax[1].plot(saturation, label="saturation")
    ax[1].plot(coupling, label="coupling_drift")
    ax[1].plot(reversibility, label="reversibility_loss")
    ax[1].set_title(f"GV-style Scalars ({title})")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import math
    run(coupled=True)
    run(coupled=False)
