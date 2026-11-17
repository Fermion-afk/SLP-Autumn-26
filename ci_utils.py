from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

def active_hf_determinant(mf: Any, ncore: int, n_active: int) -> int:
    occ = mf.mo_occ
    if isinstance(occ, tuple):
        hf_occ = np.concatenate((np.asarray(occ[0][ncore:ncore + n_active]) >= 1,
                                 np.asarray(occ[1][ncore:ncore + n_active]) >= 1))
    else:
        arr = np.asarray(occ[ncore:ncore + n_active])
        hf_occ = np.concatenate((arr >= 1, arr >= 2))
    return sum(1 << int(p) for p in np.flatnonzero(hf_occ))

def projected_hamiltonian(fermionic_op: Any, determinants: list[int]):
    basis = {det: i for i, det in enumerate(determinants)}
    rows, cols, data = [], [], []
    for label, coeff in fermionic_op.terms():
        for col, det in enumerate(determinants):
            target, sign = det, 1
            for action, p in reversed(label):
                occupied = (target >> p) & 1
                if (action == "+" and occupied) or (action == "-" and not occupied):
                    break
                sign *= (-1) ** ((target & ((1 << p) - 1)).bit_count())
                target = target | (1 << p) if action == "+" else target ^ (1 << p)
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

def lowest_eigenvectors(H: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    dim = H.shape[0]
    if not 1 <= k <= dim:
        raise ValueError(f"Requested {k} states, but subspace dimension is {dim}.")
    if k >= dim - 1:
        energies, vectors = np.linalg.eigh(H.toarray())
        return np.real(energies[:k]), vectors[:, :k]
    energies, vectors = eigsh(H, k=k, which="SA")
    order = np.argsort(energies)
    return np.real(energies[order]), vectors[:, order]

def compute_s2_expectation(determinants: list[int], coeffs: np.ndarray, n_spatial: int) -> float:
    """Exact <Phi|S^2|Phi> = <Phi|S_z^2 + S_z|Phi> + ||S_+ |Phi>||^2."""
    alpha_mask = (1 << n_spatial) - 1
    sz_part = 0.0
    s_plus: dict[int, complex] = defaultdict(complex)
    for det, c in zip(determinants, coeffs):
        if abs(c) < 1e-15:
            continue
        ms = 0.5 * ((det & alpha_mask).bit_count() - (det >> n_spatial).bit_count())
        sz_part += (abs(c) ** 2) * (ms * ms + ms)
        for p in range(n_spatial):
            p_a, p_b = p, p + n_spatial
            if not ((det >> p_a) & 1) and ((det >> p_b) & 1):
                between = ((1 << p_b) - 1) ^ ((1 << (p_a + 1)) - 1)
                sign = -1.0 if (det & between).bit_count() % 2 else 1.0
                s_plus[det ^ (1 << p_a) ^ (1 << p_b)] += sign * c
    return float(np.real(sz_part + sum(abs(a) ** 2 for a in s_plus.values())))

def determinants_to_circuits(
    determinants: list[int], coefficients: np.ndarray, n_spin_orbitals: int
) -> list[QuantumCircuit]:
    """Prepare sparse CI superpositions via exact Givens rotations."""
    circuits = []
    for i in range(coefficients.shape[1]):
        nz = [(det, complex(c)) for det, c in zip(determinants, coefficients[:, i]) if abs(c) > 1e-14]
        if not nz:
            raise ValueError(f"Initial state {i} has zero norm.")
        dets = [d for d, _ in nz]
        amps = np.asarray([c for _, c in nz], dtype=complex)
        amps /= np.linalg.norm(amps)

        qc = QuantumCircuit(n_spin_orbitals, name=f"init_state_{i}")
        for m in range(len(dets) - 1, 0, -1):
            b_prev, b_curr = dets[m - 1], dets[m]
            c_prev, c_curr = amps[m - 1], amps[m]
            r = float(np.hypot(abs(c_prev), abs(c_curr)))
            if r < 1e-15:
                continue

            diff = b_prev ^ b_curr
            target = (diff & -diff).bit_length() - 1
            flip_target = ((b_curr >> target) & 1) == 0
            if flip_target:
                qc.x(target)
                b_prev ^= 1 << target
                b_curr ^= 1 << target

            other_diffs = [q for q in range(n_spin_orbitals) if q != target and ((diff >> q) & 1)]
            for q in other_diffs:
                qc.cx(target, q)
                b_curr ^= 1 << q

            ctrls = [q for q in range(n_spin_orbitals) if q != target]
            zero_ctrls = [q for q in ctrls if not ((b_prev >> q) & 1)]
            for q in zero_ctrls:
                qc.x(q)

            givens = np.array([[np.conj(c_prev) / r, np.conj(c_curr) / r], [-c_curr / r, c_prev / r]], dtype=complex)
            qc.append(UnitaryGate(givens).control(len(ctrls)), [*ctrls, target])

            for q in zero_ctrls:
                qc.x(q)
            for q in reversed(other_diffs):
                qc.cx(target, q)
            if flip_target:
                qc.x(target)
            amps[m - 1] = r

        for q in range(n_spin_orbitals):
            if (dets[0] >> q) & 1:
                qc.x(q)
        if abs(amps[0] - 1.0) > 1e-12:
            qc.global_phase = -float(np.angle(amps[0]))

        inv_qc = qc.inverse()
        inv_qc.name = f"init_state_{i}"
        circuits.append(inv_qc)
    return circuits

def state_energies(
    estimator: Any, ansatze: Sequence[QuantumCircuit], operator: Any, point: np.ndarray | None = None, shift: float = 0.0
) -> np.ndarray:
    pubs = [(ansatz, operator) if point is None else (ansatz, operator, point) for ansatz in ansatze]
    results = estimator.run(pubs).result()
    return np.asarray([float(np.real(r.data.evs)) + shift for r in results], dtype=float)

def build_spin_summed_1rdm(
    determinants: list[int], state_vec: np.ndarray, n_spatial: int
) -> np.ndarray:
    det_idx = {d: i for i, d in enumerate(determinants)}
    gamma = np.zeros((n_spatial, n_spatial), dtype=complex)
    for offset in (0, n_spatial):
        for col, det in enumerate(determinants):
            c_col = state_vec[col]
            if abs(c_col) < 1e-15:
                continue
            for q in range(n_spatial):
                q_so = q + offset
                if not ((det >> q_so) & 1):
                    continue
                sign_q = -1 if (det & ((1 << q_so) - 1)).bit_count() % 2 else 1
                det_ann = det ^ (1 << q_so)
                for p in range(n_spatial):
                    p_so = p + offset
                    if (det_ann >> p_so) & 1:
                        continue
                    sign_p = -1 if (det_ann & ((1 << p_so) - 1)).bit_count() % 2 else 1
                    row = det_idx.get(det_ann | (1 << p_so))
                    if row is not None:
                        gamma[p, q] += np.conj(state_vec[row]) * c_col * (sign_p * sign_q)
    return np.real(0.5 * (gamma + gamma.conj().T))


def rotate_ci_vectors_to_no(
    determinants: list[int], eigenvectors: np.ndarray, u_mat: np.ndarray, n_spatial: int
) -> tuple[list[int], np.ndarray]:
    alpha_mask = (1 << n_spatial) - 1
    mo_by_sector: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for idx, det in enumerate(determinants):
        mo_by_sector[((det & alpha_mask).bit_count(), (det >> n_spatial).bit_count())].append(
            (det, eigenvectors[idx, :])
        )

    no_amplitudes: dict[int, np.ndarray] = {}
    for (na, nb), entries in mo_by_sector.items():
        mo_a = [tuple(p for p in range(n_spatial) if (d >> p) & 1) for d, _ in entries]
        mo_b = [tuple(p for p in range(n_spatial) if (d >> (p + n_spatial)) & 1) for d, _ in entries]
        mo_vecs = np.asarray([vec for _, vec in entries], dtype=complex)
        no_a_list, no_b_list = list(combinations(range(n_spatial), na)), list(combinations(range(n_spatial), nb))

        ov_a = np.array([[np.linalg.det(u_mat[np.ix_(ia, pa)]) if na > 0 else 1.0 for pa in no_a_list] for ia in mo_a], dtype=complex)
        ov_b = np.array([[np.linalg.det(u_mat[np.ix_(ib, pb)]) if nb > 0 else 1.0 for pb in no_b_list] for ib in mo_b], dtype=complex)
        coeff_tensor = np.einsum("ma,mb,mk->abk", ov_a, ov_b, mo_vecs, optimize=True)

        for ia, pa in enumerate(no_a_list):
            det_a = sum(1 << p for p in pa)
            for ib, pb in enumerate(no_b_list):
                c_vec = coeff_tensor[ia, ib]
                if np.max(np.abs(c_vec)) > 1e-12:
                    no_amplitudes[det_a | sum(1 << (p + n_spatial) for p in pb)] = c_vec

    no_dets = sorted(no_amplitudes.keys())
    no_vecs = np.vstack([no_amplitudes[d] for d in no_dets])
    for col in range(no_vecs.shape[1]):
        no_vecs[:, col] /= np.linalg.norm(no_vecs[:, col])
    return no_dets, no_vecs