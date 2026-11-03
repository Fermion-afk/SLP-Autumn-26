from ss_compass_vqe import ss_compass_vqe
import multiprocessing as mp

from datetime import datetime
from pathlib import Path
import sys

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

molecule_name = "LiH"  # Set this from your molecule input
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = Path(f"{molecule_name}_{timestamp}.txt")
log_file = open(log_path, "w", buffering=1)

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
print(f"Logging to: {log_path.resolve()}")

def main():
    # -------------------- Molecular system --------------------

    bond_length = 1.595  # Li-H bond length in Å
    basis_set = "sto-3g"

    coordinates = "\n".join([
        "Li  0.000000  0.000000  0.000000",
        f"H   0.000000  0.000000  {bond_length:.6f}",
    ])

    print("Geometry passed to PySCFDriver:")
    print(repr(coordinates))
    print(coordinates)

    # -------------------- SS-Compass-VQE parameters --------------------

    num_states = 4
    weights = [10.0, 2, 2, 0.5]

    active_space = [1, 2, 3, 4]  # 4 spatial orbitals
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