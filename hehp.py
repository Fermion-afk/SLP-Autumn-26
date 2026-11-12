from ss_compass_vqe import ss_compass_vqe
import multiprocessing as mp
import sys
from datetime import datetime
from pathlib import Path

class Tee:
    """Write output simultaneously to terminal and log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()
    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

def main():
    # -------------------- Logging --------------------
    molecule_name = "HeH+"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(f"{molecule_name}_{timestamp}.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Logging to: {log_path.resolve()}")
        # -------------------- Molecular system --------------------
        bond_length = 0.7743  # Å
        basis_set = "6-31g"
        coordinates = "\n".join([
            "He  0.000000  0.000000  0.000000",
            f"H   0.000000  0.000000  {bond_length:.6f}",
        ])
        print("\nGeometry passed to PySCFDriver:")
        print(coordinates)
        print(f"\nBasis set: {basis_set}")

        # -------------------- SS-Compass-VQE parameters --------------------
        num_states = 4
        weights = [10.0, 2.0, 2.0, 0.5]
        active_space = [1, 2, 3, 4]
        method = "single det"
        print("\nSS-Compass-VQE parameters:")
        print(f"Number of states: {num_states}")
        print(f"Weights: {weights}")
        print(f"Active space: {active_space}")
        print(f"Method: {method}")

        # -------------------- Run calculation --------------------

        print("\nStarting SS-Compass-VQE calculation...\n")
        result = ss_compass_vqe(coordinates=coordinates,
                    basis_set=basis_set,
                    weights=weights,
                    active_space=active_space,
                    charge=1,
                    method=method,
                    d_eq=bond_length,
                    no_init_states=num_states)

        print("\nSS-Compass-VQE result:")
        print(result)

    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        print(f"\nRun finished: {datetime.now().isoformat(timespec='seconds')}")
        log_file.close()

if __name__ == "__main__":
    mp.freeze_support()
    main()
