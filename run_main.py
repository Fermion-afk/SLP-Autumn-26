"""
Entry point for the SS-Compass-VQE BeH2 calculation.

Keep run_main.py, ss_compass_vqe.py, and ss_vqe.py
in the same directory.
"""

import numpy as np

from ss_vqe import SS_VQE, SS_VQEResult
from ss_compass_vqe import ss_compass_vqe


def main():
    # Linear BeH2 geometry; distances in Angstrom.
    bond_length = 1.326
    coordinates = f"""
    Be  0.000000  0.000000  0.000000
    H   0.000000  0.000000  {bond_length:.6f}
    H   0.000000  0.000000  {-bond_length:.6f}
    """

    # Molecular and SS-Compass-VQE settings.
    basis_set = "sto-3g"
    no_init_states = 3
    weights = [3.0, 2.0, 1.0]
    method = "HartreeFock"
    threshold = [1e-5, 1e-5]  # [T2 threshold, S threshold]

    # Scattering-screening indices:
    # Spatial MOs 1 and 2, including alpha and beta spin orbitals.
    active_space = [1, 2, 7, 8]

    results = ss_compass_vqe(
        coordinates=coordinates,
        d_eq=bond_length,
        basis_set=basis_set,
        no_init_states=no_init_states,
        weights=weights,
        method=method,
        threshold=threshold,
        active_space=active_space,
        N_workers=None,
        s_weights=None,
    )

    if results is None:
        raise RuntimeError("SS-Compass-VQE did not return a result.")

    print("\n========== BeH2 SUMMARY ==========")
    print(f"Geometry: Linear BeH2, r(Be-H) = {bond_length:.6f} Angstrom")
    print(f"Basis: {basis_set}")
    print(f"Initial-state method: {method}")
    print(f"Number of reference states: {no_init_states}")
    print(f"Screening indices: {active_space}")
    print(f"Number of ansatz parameters: {results['n_params']}")

    print("\nSS-Compass-VQE energies:")
    for i, energy in enumerate(results["energies"]):
        print(f"  State {i}: {energy:.10f} Eh")

    print("\nExact (FCI) energies:")
    for i, energy in enumerate(results["exact_energies"]):
        print(f"  State {i}: {energy:.10f} Eh")

    errors = np.abs(
        np.asarray(results["energies"]) -
        np.asarray(results["exact_energies"])
    )

    print("\nAbsolute errors:")
    for i, error in enumerate(errors):
        print(f"  State {i}: {error:.3e} Eh")

    print("\nOptimal parameters:")
    print(results["optimal_params"])


if __name__ == "__main__":
    main()