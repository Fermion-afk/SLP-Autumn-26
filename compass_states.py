"""Fermionic excitation pools, determinant projections, and reference state builders."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

# ------------------------------------------------------------------
# Fermionic Excitation Pool Generators
# ------------------------------------------------------------------
def all_singles(num_spatial_orbitals: int, num_particles: tuple[int, int]) -> list[tuple]:
    n, (n_a, n_b) = num_spatial_orbitals, num_particles
    occ = list(range(n_a)) + list(range(n, n + n_b))
    vir = [i for i in range(2 * n) if i not in occ]
    return [((i,), (a,)) for i in occ for a in vir]

def all_doubles(num_spatial_orbitals: int, num_particles: tuple[int, int]) -> list[tuple]:
    n, (n_a, n_b) = num_spatial_orbitals, num_particles
    occ = list(range(n_a)) + list(range(n, n + n_b))
    vir = [i for i in range(2 * n) if i not in occ]
    return [(o, v) for o in combinations(occ, 2) for v in combinations(vir, 2)]

def scatter_hole_ex(num_spatial_orbitals: int, num_particles: tuple[int, int]) -> list[tuple]:
    n, (n_a, n_b) = num_spatial_orbitals, num_particles
    occ_a, occ_b = range(n_a), range(n, n + n_b)
    vir_a, vir_b = range(n_a, n), range(n + n_b, 2 * n)
    ex = [((i, j), (a, m)) for i in occ_a for j in occ_b for a in vir_a for m in occ_b if m != j]
    ex += [((j, i), (m, a)) for i in occ_b for j in occ_a for a in vir_b for m in occ_a if m != j]
    return ex

def scatter_particle_ex(num_spatial_orbitals: int, num_particles: tuple[int, int]) -> list[tuple]:
    n, (n_a, n_b) = num_spatial_orbitals, num_particles
    occ_a, occ_b = range(n_a), range(n, n + n_b)
    vir_a, vir_b = range(n_a, n), range(n + n_b, 2 * n)
    ex = [((i, e), (a, b)) for i in occ_a for e in vir_b for a in vir_a for b in vir_beta_iter(vir_b, e)]
    ex += [((i, e), (a, b)) for i in occ_b for e in vir_a for a in vir_b for b in vir_beta_iter(vir_a, e)]
    return ex

def vir_beta_iter(vir_range: range, excluded: int):
    return (b for b in vir_range if b != excluded)

# ------------------------------------------------------------------
# Determinant Space & Sparse Hamiltonian Projection
# ------------------------------------------------------------------
def active_hf_determinant(mf, ncore: int, n_active: int) -> int:
    occ = np.asarray(mf.mo_occ[ncore : ncore + n_active])
    hf_occ = np.concatenate((occ >= 1, occ >= 2))
    return int(sum(1 << int(p) for p in np.flatnonzero(hf_occ)))

def cis_determinants(hf_det: int, n_spin_orbitals: int) -> list[int]:
    occ = [p for p in range(n_spin_orbitals) if (hf_det >> p) & 1]
    vir = [p for p in range(n_spin_orbitals) if not ((hf_det >> p) & 1)]
    return [hf_det] + [hf_det ^ (1 << p) ^ (1 << q) for p in occ for q in vir]

def cisd_determinants(hf_det: int, n_spin_orbitals: int) -> list[int]:
    occ = [p for p in range(n_spin_orbitals) if (hf_det >> p) & 1]
    vir = [p for p in range(n_spin_orbitals) if not ((hf_det >> p) & 1)]
    doubles = [
        hf_det ^ sum(1 << p for p in o) ^ sum(1 << p for p in v)
        for o in combinations(occ, 2)
        for v in combinations(vir, 2)
    ]
    return cis_determinants(hf_det, n_spin_orbitals) + doubles

def projected_hamiltonian(fermionic_op, determinants: list[int]):
    """Project a FermionicOp into the subspace spanned by bit-encoded determinants."""
    basis = {det: i for i, det in enumerate(determinants)}
    rows, cols, data = [], [], []

    for label, coeff in fermionic_op.terms():
        ops = [(tok[0], int(tok[2:])) for tok in label.split()]
        for col, det in enumerate(determinants):
            target, sign = det, 1
            for action, p in reversed(ops):
                is_occ = (target >> p) & 1
                if (action == "+" and is_occ) or (action == "-" and not is_occ):
                    break
                sign *= (-1) ** ((target & ((1 << p) - 1)).bit_count())
                target = (target | (1 << p)) if action == "+" else (target ^ (1 << p))
            else:
                row = basis.get(target)
                if row is not None:
                    rows.append(row)
                    cols.append(col)
                    data.append(coeff * sign)

    dim = len(determinants)
    H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex).tocsc()
    diff = H - H.conj().T
    if diff.nnz > 0 and np.max(np.abs(diff.data)) > 1e-10:
        raise ValueError("Projected Hamiltonian is not Hermitian.")
    return H

def lowest_eigenvectors(H, k: int) -> tuple[np.ndarray, np.ndarray]:
    dim = H.shape[0]
    if not 1 <= k <= dim:
        raise ValueError(f"Requested {k} states, but the determinant space has dimension {dim}.")
    if k >= dim - 1:
        energies, vectors = np.linalg.eigh(H.toarray())
        return energies[:k], vectors[:, :k]
    energies, vectors = eigsh(H, k=k, which="SA")
    order = np.argsort(energies)
    return energies[order], vectors[:, order]

def determinants_to_circuits(
    determinants: list[int], coefficients: np.ndarray, n_spin_orbitals: int
) -> list[QuantumCircuit]:
    circuits = []
    for i in range(coefficients.shape[1]):
        sv = np.zeros(2**n_spin_orbitals, dtype=complex)
        sv[determinants] = coefficients[:, i]
        norm = np.linalg.norm(sv)
        if norm < 1e-12:
            raise ValueError("Initial state has zero norm.")
        qc = QuantumCircuit(n_spin_orbitals, name=f"init_state_{i}")
        qc.append(StatePreparation(Statevector(sv / norm)), range(n_spin_orbitals))
        circuits.append(qc)
    return circuits

# ------------------------------------------------------------------
# Reference State Initialization Methods (DPT1, CIS, CISD)
# ------------------------------------------------------------------
def _init_ci_states(
    fermionic_op, mf, ncore: int, n_active: int, k: int, det_fn: Callable, label: str
) -> list[QuantumCircuit]:
    if k < 1:
        raise ValueError("At least one reference state is required.")
    n_so = 2 * n_active
    dets = det_fn(active_hf_determinant(mf, ncore, n_active), n_so)
    energies, coeffs = lowest_eigenvectors(projected_hamiltonian(fermionic_op, dets), k)
    print(f"[SS-COMPASS] {label} determinant space dimension: {len(dets)}")
    print(f"[SS-COMPASS] Lowest {label} energies: {energies}")
    return determinants_to_circuits(dets, coeffs, n_so)

def _unperturbed_det_energies(
    determinants: list[int], H_proj, mf, ncore: int, n_active: int, h0_type: str = "mp"
) -> np.ndarray:
    if h0_type == "en":
        return np.real(H_proj.diagonal())
    mo_e = mf.mo_energy
    if isinstance(mo_e, tuple) or (isinstance(mo_e, np.ndarray) and mo_e.ndim == 2):
        eps_a = np.asarray(mo_e[0][ncore : ncore + n_active], dtype=float)
        eps_b = np.asarray(mo_e[1][ncore : ncore + n_active], dtype=float)
    else:
        eps_a = eps_b = np.asarray(mo_e[ncore : ncore + n_active], dtype=float)
    spin_eps = np.concatenate((eps_a, eps_b))

    energies = np.zeros(len(determinants), dtype=float)
    for idx, det in enumerate(determinants):
        e0, bits = 0.0, det
        while bits:
            lsb = bits & -bits
            e0 += spin_eps[lsb.bit_length() - 1]
            bits ^= lsb
        energies[idx] = e0
    return energies

def init_dpt1_states(
    fermionic_op, mf, ncore: int, n_active: int, k: int, deg_tol: float = 1e-5, h0_type: str = "mp"
) -> list[QuantumCircuit]:
    """Generate the lowest-k first-order degenerate perturbation theory (DPT1) states."""
    if k < 1:
        raise ValueError("At least one reference state is required.")
    n_so = 2 * n_active
    seed_dets = cisd_determinants(active_hf_determinant(mf, ncore, n_active), n_so)
    if len(seed_dets) < k:
        raise ValueError(f"Requested {k} states, but only {len(seed_dets)} seed determinants exist.")

    H_seed = projected_hamiltonian(fermionic_op, seed_dets) if h0_type == "en" else None
    e0_seed = _unperturbed_det_energies(seed_dets, H_seed, mf, ncore, n_active, h0_type)
    order = np.argsort(e0_seed)
    sorted_dets = [seed_dets[i] for i in order]
    sorted_e0 = e0_seed[order]

    p_size = int(np.searchsorted(sorted_e0, sorted_e0[k - 1] + deg_tol, side="right"))
    p_dets = sorted_dets[:p_size]
    p_set = set(p_dets)
    q_dets = list({ex for det in p_dets for ex in cisd_determinants(det, n_so) if ex not in p_set})

    all_dets = p_dets + q_dets
    H_full = projected_hamiltonian(fermionic_op, all_dets)
    e0_all = _unperturbed_det_energies(all_dets, H_full, mf, ncore, n_active, h0_type)

    e_p, v_p = np.linalg.eigh(H_full[:p_size, :p_size].toarray())
    c0 = v_p[:, :k]
    coeffs = np.zeros((len(all_dets), k), dtype=complex)
    coeffs[:p_size, :] = c0

    if q_dets:
        v_eff = H_full[p_size:, :p_size] @ c0
        e0_states = np.sum((np.abs(c0) ** 2) * e0_all[:p_size, None], axis=0)
        denom = e0_states[None, :] - e0_all[p_size:, None]
        safe = np.abs(denom) > deg_tol
        coeffs[p_size:, :] = np.where(safe, v_eff / np.where(safe, denom, 1.0), 0.0)

    # Löwdin symmetric orthonormalization
    eigvals, eigvecs = np.linalg.eigh(coeffs.conj().T @ coeffs)
    coeffs = coeffs @ (eigvecs @ np.diag(1.0 / np.sqrt(np.asarray(eigvals, dtype=float))) @ eigvecs.conj().T)

    print(f"[SS-COMPASS] DPT1 model space dim={p_size}, Q space dim={len(q_dets)}")
    print(f"[SS-COMPASS] DPT1 1st-order energies (E0 + E1): {e_p[:k]}")
    return determinants_to_circuits(all_dets, coeffs, n_so)

def build_initial_states(
    method: str | None, fermionic_op, mf, ncore: int, num_spatial_orbitals: int, no_init_states: int
) -> list[QuantumCircuit]:
    m = str(method).lower()
    if m in ("dpt1", "pt1", "dpt"):
        return init_dpt1_states(fermionic_op, mf, ncore, num_spatial_orbitals, no_init_states)
    if m == "cis":
        return _init_ci_states(fermionic_op, mf, ncore, num_spatial_orbitals, no_init_states, cis_determinants, "CIS")
    if m == "cisd":
        return _init_ci_states(fermionic_op, mf, ncore, num_spatial_orbitals, no_init_states, cisd_determinants, "CISD")
    raise ValueError(f"Unknown initialization method: {method}")