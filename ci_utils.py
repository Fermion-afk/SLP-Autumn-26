from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Any
from pyscf import fci

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

def _split_to_pyscf_sectors(dets: list[int], vec: np.ndarray, m: int) -> dict[tuple[int, int], np.ndarray]:
    """Pack bitstring CI vector into PySCF (na_str, nb_str) sector matrices."""
    alpha_mask = (1 << m) - 1
    grouped: dict[tuple[int, int], list[tuple[int, int, complex]]] = defaultdict(list)
    for det, amp in zip(dets, vec):
        if abs(amp) > 1e-16:
            oa, ob = det & alpha_mask, det >> m
            grouped[(oa.bit_count(), ob.bit_count())].append((oa, ob, amp))

    sectors: dict[tuple[int, int], np.ndarray] = {}
    for (na, nb), items in grouped.items():
        civec = np.zeros((fci.cistring.num_strings(m, na), fci.cistring.num_strings(m, nb)), dtype=vec.dtype)
        for oa, ob, amp in items:
            civec[fci.cistring.str2addr(m, na, oa), fci.cistring.str2addr(m, nb, ob)] = amp
        sectors[(na, nb)] = civec
    return sectors

def build_spin_summed_1rdm(dets: list[int], vec: np.ndarray, m: int) -> np.ndarray:
    """Spin-summed spatial 1-RDM via PySCF C-compiled fci.direct_spin1.make_rdm1."""
    gamma = np.zeros((m, m), dtype=float)
    for (na, nb), civec in _split_to_pyscf_sectors(dets, vec, m).items():
        if np.iscomplexobj(civec):
            rdm_re = fci.direct_spin1.make_rdm1(np.ascontiguousarray(np.real(civec)), m, (na, nb))
            rdm_im = fci.direct_spin1.make_rdm1(np.ascontiguousarray(np.imag(civec)), m, (na, nb))
            gamma += rdm_re + rdm_im
        else:
            gamma += fci.direct_spin1.make_rdm1(np.ascontiguousarray(civec), m, (na, nb))
    return 0.5 * (gamma + gamma.T)

def rotate_ci_vectors_to_no(
    dets: list[int], evecs: np.ndarray, u: np.ndarray, m: int, tol: float = 1e-12
) -> tuple[list[int], np.ndarray]:
    """Transform CI eigenvectors into NO basis via PySCF fci.addons.transform_ci_for_orbital_rotation."""
    n_roots = evecs.shape[1]
    u_c = np.ascontiguousarray(u, dtype=float)
    rot_maps: list[dict[int, complex]] = [defaultdict(complex) for _ in range(n_roots)]

    for k in range(n_roots):
        for (na, nb), civec in _split_to_pyscf_sectors(dets, evecs[:, k], m).items():
            if np.iscomplexobj(civec):
                rot_ci = (fci.addons.transform_ci_for_orbital_rotation(np.ascontiguousarray(np.real(civec)), m, (na, nb), u_c)
                          + 1j * fci.addons.transform_ci_for_orbital_rotation(np.ascontiguousarray(np.imag(civec)), m, (na, nb), u_c))
            else:
                rot_ci = fci.addons.transform_ci_for_orbital_rotation(np.ascontiguousarray(civec), m, (na, nb), u_c)

            strs_a = fci.cistring.gen_strings4orblist(range(m), na)
            strs_b = fci.cistring.gen_strings4orblist(range(m), nb)
            nz_a, nz_b = np.where(np.abs(rot_ci) > tol)
            for ia, ib in zip(nz_a, nz_b):
                det_no = int(strs_a[ia]) | (int(strs_b[ib]) << m)
                rot_maps[k][det_no] += complex(rot_ci[ia, ib])

    no_dets = sorted({det for rm in rot_maps for det in rm})
    det_idx = {det: i for i, det in enumerate(no_dets)}
    no_evecs = np.zeros((len(no_dets), n_roots), dtype=evecs.dtype)
    for k, rm in enumerate(rot_maps):
        for det, amp in rm.items():
            no_evecs[det_idx[det], k] = amp if np.iscomplexobj(no_evecs) else float(np.real(amp))
    return no_dets, no_evecs