from __future__ import annotations

from pathlib import Path
import numpy as np
from revamped import ss_compass_vqe

def main() -> None:
    # Replace with the path to your .chk.npz checkpoint file
    chk_file = Path("bh_eq_nano.chk.npz")

    if not chk_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {chk_file}")

    # Restores state and appends output to the matching .log file in the same directory
    solver = ss_compass_vqe.resume(chk_file)
    results = solver.run(compute_exact=True)

    print("\nFinal Energies (Ha):", np.array2string(results["state_energies"], precision=10))
    if solver.exact_energies is not None:
        print("Exact Energies (Ha):", np.array2string(solver.exact_energies, precision=10))

if __name__ == "__main__":
    main()