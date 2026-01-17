from gv_engine.core import SimpleSystem


if __name__ == "__main__":
    system = SimpleSystem(coupled=True)
    system.run(200)
    system.plot("(coupled)")
    system.print_final_scalars("coupled")

    system = SimpleSystem(coupled=False)
    system.run(200)
    system.plot("(decoupled)")
    system.print_final_scalars("decoupled")
