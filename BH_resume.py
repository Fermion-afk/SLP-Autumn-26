from __future__ import annotations

import numpy as np
from revamped import ss_compass_vqe

def main() -> None:
    # Replace with the exact .chk.npz filename on your remote server
    chk_file = "./checkpoints/bh_eq.chk.npz"

    solver = ss_compass_vqe.resume(chk_file)
    results = solver.run(compute_exact=True)

    print("\nFinal Energies (Ha):", np.array2string(results["state_energies"], precision=10))
    if solver.exact_energies is not None:
        print("Exact Energies (Ha):", np.array2string(solver.exact_energies, precision=10))

if __name__ == "__main__":
    main()