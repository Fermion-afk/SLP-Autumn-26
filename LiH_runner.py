from __future__ import annotations

import json
import numpy as np
from revamped import ss_compass_vqe

def main() -> None:
    solver = ss_compass_vqe(
        coordinates="Li 0.0 0.0 0.0; H 0.0 0.0 1.595",
        spin=0,
        charge=0,
        basis_set="sto-3g",
        weights=[10/12, 10/12, 1/12],
        screening_weights=[1 / 101, 50 / 101, 50 / 101],
        no_init_states=3,
        threshold=[1e-6, 5e-7],
        nfc=False,
        log_file="lih_eq_nano.log",
    )

    results = solver.run(compute_exact=True)

    # Export full per-operator screening records (optimal params, cost, state energies, selected flag)
    screening_report = {
        "t2_screening": solver.t2_screening_data,
        "s_screening": solver.s_screening_data,
    }
    with open("lih_eq_screening.json", "w", encoding="utf-8") as fh:
        json.dump(screening_report, fh, indent=2)

    print("\nFinal Energies (Ha):", np.array2string(results["state_energies"], precision=10))
    if solver.exact_energies is not None:
        print("Exact Energies (Ha):", np.array2string(solver.exact_energies, precision=10))

if __name__ == "__main__":
    main()