import sys

from gv_engine.core import SimpleSystem


def run_experiment_a(steps: int = 200):
    # Coupled (default behavior)
    system = SimpleSystem(coupled=True)
    system.run(steps)
    system.plot(title_suffix="(coupled)")
    system.print_final_scalars(label="A (coupled)")


def run_experiment_b(steps: int = 200):
    # Decoupled
    system = SimpleSystem(coupled=False)
    system.run(steps)
    system.plot(title_suffix="(decoupled)")
    system.print_final_scalars(label="B (decoupled)")


def main():
    # Usage:
    #   python .\examples\run_experiment.py A
    #   python .\examples\run_experiment.py B
    #   python .\examples\run_experiment.py A 500
    if len(sys.argv) < 2:
        print("Usage: python .\\examples\\run_experiment.py [A|B] [steps]")
        raise SystemExit(2)

    which = sys.argv[1].strip().upper()
    steps = int(sys.argv[2]) if len(sys.argv) >= 3 else 200

    if which == "A":
        run_experiment_a(steps)
    elif which == "B":
        run_experiment_b(steps)
    else:
        print("Unknown experiment. Use A or B.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
