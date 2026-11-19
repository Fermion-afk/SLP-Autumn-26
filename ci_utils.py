from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Any
from pyscf import fci

import numpy as np
from qiskit import transpile as qk_transpile
from qiskit.circuit import QuantumCircuit
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

def determinants_to_circuits(dets: list[int], evecs: np.ndarray, num_spin_orbitals: int) -> list[QuantumCircuit]:
    """Compile sparse determinant statevectors into purely unitary U/CX circuits (no initialize/reset)."""
    circuits: list[QuantumCircuit] = []
    for col in range(evecs.shape[1]):
        terms = [(det, complex(amp)) for det, amp in zip(dets, evecs[:, col]) if abs(amp) > 1e-12]
        qc = QuantumCircuit(num_spin_orbitals)
        if not terms:
            circuits.append(qc)
            continue

        norm = float(np.linalg.norm([a for _, a in terms]))
        terms = [(d, a / norm) for d, a in terms]

        # Build sequential unitary Givens rotations transforming |0...0> -> sum_k a_k |d_k>
        # Step k rotates remaining weight on |0...0> into |d_k> via a pivot bit where d_k has a 1
        # that distinguishes it from all previously populated determinants d_0..d_{k-1}.
        # Order determinants so every d_k (except possibly one |0...0>) has at least one 1-bit
        # not shared as a distinguishing conflict, using binary-tree decomposition on active qubits.
        active_qubits = sorted({q for d, _ in terms for q in range(num_spin_orbitals) if (d >> q) & 1})
        if not active_qubits:
            phase0 = float(np.angle(terms[0][1]))
            if abs(phase0) > 1e-12:
                qc.rz(-2.0 * phase0, 0)
                qc.x(0)
                qc.rz(2.0 * phase0, 0)
                qc.x(0)
            circuits.append(qc)
            continue

        # Recursive sparse Schmidt / binary-tree state preparation over active_qubits (exact, unitary, zero ancillas)
        sub_dim = 1 << len(active_qubits)
        sub_sv = np.zeros(sub_dim, dtype=complex)
        for d, a in terms:
            sub_idx = sum(((d >> q) & 1) << bit for bit, q in enumerate(active_qubits))
            sub_sv[sub_idx] = a

        def _prep_subspace(qubits: list[int], vec: np.ndarray, ctrls: list[int]) -> None:
            if len(qubits) == 0:
                ph = float(np.angle(vec[0]))
                if abs(ph) > 1e-12:
                    if len(ctrls) == 1:
                        qc.p(ph, ctrls[0])
                    elif len(ctrls) > 1:
                        qc.mcp(ph, ctrls[:-1], ctrls[-1])
                return

            half = 1 << (len(qubits) - 1)
            v0, v1 = vec[:half], vec[half:]
            n0, n1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
            tot = float(np.hypot(n0, n1))
            if tot < 1e-15:
                return

            target = qubits[-1]
            rest = qubits[:-1]
            theta = 2.0 * float(np.arctan2(n1, n0))
            if abs(theta) > 1e-12:
                if ctrls:
                    qc.mcry(theta, ctrls, target)
                else:
                    qc.ry(theta, target)

            if n0 > 1e-14:
                qc.x(target)
                _prep_subspace(rest, v0 / n0, ctrls + [target])
                qc.x(target)
            if n1 > 1e-14:
                _prep_subspace(rest, v1 / n1, ctrls + [target])

        _prep_subspace(active_qubits, sub_sv, [])
        circuits.append(qc)

    return qk_transpile(circuits, basis_gates=["u", "cx"], optimization_level=1)

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