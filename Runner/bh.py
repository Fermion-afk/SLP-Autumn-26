from ss_compass_vqe import ss_compass_vqe
import multiprocessing as mp

from datetime import datetime
from pathlib import Path


def main():
    # -------------------- Molecular system --------------------

    molecule_name = "BH"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"{molecule_name}_{timestamp}.txt")

    bond_length = 1.232  # B-H bond length in Å
    basis_set = "sto-3g"

    coordinates = "\n".join([
        "B   0.000000   0.000000   0.000000",
        f"H   0.000000   0.000000   {bond_length:.6f}",
    ])

    print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Logging to: {log_path.resolve()}")
    print("Geometry passed to PySCFDriver:")
    print(coordinates)

    # -------------------- SS-Compass-VQE parameters --------------------

    num_states = 3
    weights = [10.0, 2.0, 2.0]

    active_space = [1, 2, 3, 4]
    method = "HartreeFock"

    # -------------------- Run SS-Compass-VQE --------------------

    result = ss_compass_vqe(
        coordinates=coordinates,
        basis_set=basis_set,
        weights=weights,
        active_space=active_space,
        method=method,
        d_eq=bond_length,
        no_init_states=num_states,
    )

    print("\nSS-Compass-VQE result:")
    print(result)


if __name__ == "__main__":
    mp.freeze_support()
    main()