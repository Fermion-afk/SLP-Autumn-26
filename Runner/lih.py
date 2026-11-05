from ss_compass_vqe import ss_compass_vqe
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import sys

molecule_name = "LiH"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = Path(f"{molecule_name}_{timestamp}.txt")

def main():
    # Log terminal output to one timestamped .txt file
    with log_path.open("w", buffering=1) as log_file:
        original_stdout = sys.stdout

        class Tee:
            def write(self, text):
                original_stdout.write(text)
                log_file.write(text)

            def flush(self):
                original_stdout.flush()
                log_file.flush()

        sys.stdout = Tee()
        sys.stderr = sys.stdout

        print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Logging to: {log_path.resolve()}")

        # -------------------- Molecular system --------------------

        bond_length = 1.595  # Li-H bond length in Å
        basis_set = "sto-3g"

        coordinates = "\n".join([
            "Li  0.000000  0.000000  0.000000",
            f"H   0.000000  0.000000  {bond_length:.6f}",
        ])

        print("\nGeometry passed to PySCFDriver:")
        print(coordinates)

        # -------------------- SS-Compass-VQE parameters --------------------

        num_states = 4
        weights = [10.0, 2.0, 2.0, 0.5]
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
        print(f"\nRun finished: {datetime.now().isoformat(timespec='seconds')}")

if __name__ == "__main__":
    mp.freeze_support()
    main()