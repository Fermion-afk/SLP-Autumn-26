from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np
from qiskit import transpile as qk_transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

from ci_utils import state_energies
from ss_vqe import SS_VQE

def _mr_singles(n: int, inactive_occ: list[int], active_occ: list[int],
    active_virt: list[int], inactive_virt: list[int]) -> list[tuple[tuple[int], tuple[int]]]:
    holes = sorted(set(inactive_occ) | set(active_occ))
    particles = sorted(set(active_virt) | set(inactive_virt))
    return [((i,), (a,)) for i in holes for a in particles]

def _mr_doubles(n: int, inactive_occ: list[int], active_occ: list[int],
    active_virt: list[int], inactive_virt: list[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    holes = sorted(set(inactive_occ) | set(active_occ))
    particles = sorted(set(active_virt) | set(inactive_virt))
    return [((i, j), (a, b)) for i, j in combinations(holes, 2) for a, b in combinations(particles, 2)]

def _sh_excitations(n: int, inactive_occ: list[int], active_occ: list[int],
    active_virt: list[int], inactive_virt: list[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Opposite-spin (OP) hole scatterers with contractible hole m in active_occ."""
    holes = sorted(set(inactive_occ))
    particles = sorted(set(inactive_virt))
    active_set = sorted(set(active_occ))
    occ_a, occ_b = [p for p in holes if p < n], [p for p in holes if p >= n]
    vir_a = [p for p in particles if p < n]
    cso_b = [p for p in active_set if p >= n]
    return [pair
        for i in occ_a for j in occ_b for a in vir_a for m in cso_b if m != j
        for pair in (((i, j), (a, m)), ((i + n, j - n), (a + n, m - n)))]

def _sp_excitations(n: int, inactive_occ: list[int], active_occ: list[int],
    active_virt: list[int], inactive_virt: list[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Opposite-spin (OP) particle scatterers with contractible particle e in active_virt."""
    holes = sorted(set(inactive_occ))
    particles = sorted(set(inactive_virt))
    active_set = sorted(set(active_virt))
    occ_a = [p for p in holes if p < n]
    vir_a, vir_b = [p for p in particles if p < n], [p for p in particles if p >= n]
    cso_b = [p for p in active_set if p >= n]
    return [pair
        for i in occ_a for e in cso_b for a in vir_a for b in vir_b if e != b
        for pair in (((i, e), (a, b)), ((i + n, e - n), (a + n, b - n)))]

def build_operator_pools(num_spatial_orbitals: int, num_particles: tuple[int, int], mapper: Any,
    inactive_occ: list[int], active_occ: list[int], active_virt: list[int], inactive_virt: list[int]) -> dict[str, list[Any]]:
    hf_ref = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    args = (num_spatial_orbitals, inactive_occ, active_occ, active_virt, inactive_virt)

    ucc_s = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
                excitations=lambda *_: _mr_singles(*args), initial_state=hf_ref, qubit_mapper=mapper)
    ucc_d = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
                excitations=lambda *_: _mr_doubles(*args), initial_state=hf_ref, qubit_mapper=mapper)
    ucc_sh = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
                 excitations=lambda *_: _sh_excitations(*args), initial_state=hf_ref, qubit_mapper=mapper)
    ucc_sp = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
                 excitations=lambda *_: _sp_excitations(*args), initial_state=hf_ref, qubit_mapper=mapper)

    return {"s_excitations": list(ucc_s.excitation_list or []), "s_paulis": [mapper.map(op) for op in ucc_s.excitation_ops()],
            "t2_excitations": list(ucc_d.excitation_list or []), "t2_paulis": [mapper.map(op) for op in ucc_d.excitation_ops()],
            "sh_excitations": list(ucc_sh._get_excitation_list() or []), "sh_paulis": [mapper.map(op) for op in ucc_sh.excitation_ops()],
            "sp_excitations": list(ucc_sp._get_excitation_list() or []), "sp_paulis": [mapper.map(op) for op in ucc_sp.excitation_ops()]}

def _t2_screen_worker(args):
    idx, hamiltonian, transpiled_circuit, shift, weights, s_weights, initial_energies = args
    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    solver = SS_VQE(cast(Any, estimator), transpiled_circuit, weights, optimizer, initial_point=np.array([0.0]))
    result = solver.compute_minimum_eigenvalue(hamiltonian)
    theta_t2 = float(np.asarray(result.optimal_point).reshape(-1)[0])
    energies = state_energies(estimator, transpiled_circuit, hamiltonian, np.array([theta_t2]), shift)
    delta_energies = energies - initial_energies
    score = float(np.dot(s_weights, np.abs(delta_energies)))
    return idx, score, theta_t2, energies, delta_energies

def _s_screen_worker(args):
    cand_idx, hamiltonian, transpiled_circuit, shift, weights, s_weights, baseline_energies, theta_t2 = args
    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    solver = SS_VQE(cast(Any, estimator), transpiled_circuit, weights, optimizer, initial_point=np.array([theta_t2, 0.01]))
    result = solver.compute_minimum_eigenvalue(hamiltonian)
    opt = np.asarray(result.optimal_point).reshape(-1)
    theta_t2_2p, theta_s_2p = float(opt[0]), float(opt[1])
    energies = state_energies(estimator, transpiled_circuit, hamiltonian, opt, shift)
    delta_energies = energies - baseline_energies
    score = float(np.dot(s_weights, np.abs(delta_energies)))
    return cand_idx, score, theta_t2_2p, theta_s_2p, energies, delta_energies

def screen_t2_pool(t2_paulis: list[Any], t2_excitations: list[Any], init_states: list[QuantumCircuit],
    qubit_op: Any, shift: float, weights: np.ndarray, screening_weights: np.ndarray,
    initial_energies: np.ndarray, threshold_t2: float, n_workers: int, mp_ctx: Any,
    log_fn: Callable[[str], None]) -> tuple[list[tuple[float, int, float, np.ndarray]], list[dict[str, Any]]]:
    n_t2 = len(t2_paulis)
    if n_t2 == 0:
        return [], []

    t2_ansatze = [[EvolvedOperatorAnsatz([op], initial_state=ref) for ref in init_states] for op in t2_paulis]
    t2_transpiled = [qk_transpile(cast(list[QuantumCircuit], ans), optimization_level=1) for ans in t2_ansatze]
    t2_args = [(i, qubit_op, t2_transpiled[i], shift, weights, screening_weights, initial_energies) for i in range(n_t2)]

    workers = min(n_workers, n_t2)
    log_fn(f" --- Submethod 5a: _screen_t2 (pool={n_t2}, threshold={threshold_t2:.2e}, workers={workers}) ---")

    t2_raw = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(_t2_screen_worker, arg) for arg in t2_args]
        for i, fut in enumerate(as_completed(futures), 1):
            t2_raw.append(fut.result())
            if i == 1 or i % max(1, n_t2 // 10) == 0 or i == n_t2:
                log_fn(f"   T2 progress: {i}/{n_t2} ({100.0 * i / n_t2:.1f}%)")

    t2_raw.sort(key=lambda x: x[0])
    t2_screen = sorted([(score, idx, theta, energies) for idx, score, theta, energies, _ in t2_raw if score > threshold_t2],
                       key=lambda item: item[0], reverse=True)
    t2_data = [{"operator_id": idx, "excitation": str(t2_excitations[idx]), "delta_energies": delta.tolist(),
                "weighted_sum": score, "threshold": float(threshold_t2), "selected": bool(score > threshold_t2),
                "theta_t2": theta} for idx, score, theta, energies, delta in t2_raw]
    return t2_screen, t2_data

def build_s_candidates(t2_screen: list[tuple[float, int, float, np.ndarray]], t2_excitations: list[Any],
    sh_excitations: list[Any], sp_excitations: list[Any], active_orbitals: list[int]) -> list[tuple[Any, ...]]:
    """Exact legacy CSO commutativity matching: sh_ex[1][1] == act_orb and sp_ex[0][1] == act_orb."""
    active_space_set = set(active_orbitals)
    s_candidates: list[tuple[Any, ...]] = []

    for rank, (_, t2_idx, theta_t2, e_t2) in enumerate(t2_screen):
        t2_ex = t2_excitations[t2_idx]
        t2_flat = [t2_ex[0][0], t2_ex[1][0], t2_ex[0][1], t2_ex[1][1]]
        active_in_t2 = active_space_set.intersection(t2_flat)
        if not active_in_t2:
            continue

        sh_seen: set[int] = set()
        sp_seen: set[int] = set()
        for act_orb in active_in_t2:
            for j, sh_ex in enumerate(sh_excitations):
                if j not in sh_seen and sh_ex[1][1] == act_orb:
                    s_candidates.append((len(s_candidates), rank, t2_idx, "h", j, theta_t2, e_t2))
                    sh_seen.add(j)
            for j, sp_ex in enumerate(sp_excitations):
                if j not in sp_seen and sp_ex[0][1] == act_orb:
                    s_candidates.append((len(s_candidates), rank, t2_idx, "p", j, theta_t2, e_t2))
                    sp_seen.add(j)
    return s_candidates

def screen_s_pool(s_candidates: list[tuple[Any, ...]], t2_paulis: list[Any], sh_paulis: list[Any], sp_paulis: list[Any],
    sh_excitations: list[Any], sp_excitations: list[Any], init_states: list[QuantumCircuit], qubit_op: Any,
    shift: float, weights: np.ndarray, screening_weights: np.ndarray, threshold_s: float, n_workers: int,
    mp_ctx: Any, log_fn: Callable[[str], None],
    cached_results: dict[int, tuple[float, float, float, np.ndarray, np.ndarray]] | None = None,
    checkpoint_fn: Callable[[dict[int, Any]], None] | None = None,
    checkpoint_interval: float = 60.0) -> tuple[dict[int, list[tuple[Any, ...]]], list[dict[str, Any]]]:
    n_s = len(s_candidates)
    if n_s == 0:
        log_fn(" --- Submethod 5b: _screen_s (0 CSO-matched candidates) ---")
        return defaultdict(list), []

    s_results: dict[int, tuple[float, float, float, np.ndarray, np.ndarray]] = dict(cached_results or {})
    pending = [cand for cand in s_candidates if cand[0] not in s_results]

    workers = min(n_workers, max(1, len(pending)))
    log_fn(f" --- Submethod 5b: _screen_s (candidates={n_s}, cached={len(s_results)}, "
           f"pending={len(pending)}, threshold={threshold_s:.2e}, workers={workers}) ---")

    if pending:
        s_ansatze = [[EvolvedOperatorAnsatz([t2_paulis[t2_idx], sh_paulis[s_idx] if s_type == "h" else sp_paulis[s_idx]],
                                            initial_state=ref) for ref in init_states]
                     for (_, _, t2_idx, s_type, s_idx, _, _) in pending]
        s_transpiled = [qk_transpile(cast(list[QuantumCircuit], ans), optimization_level=1) for ans in s_ansatze]
        s_args = [(cand_idx, qubit_op, s_transpiled[ci], shift, weights, screening_weights, e_t2, theta_t2)
                  for ci, (cand_idx, _, t2_idx, s_type, s_idx, theta_t2, e_t2) in enumerate(pending)]

        last_chk = perf_counter()
        done_base = len(s_results)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
            futures = [pool.submit(_s_screen_worker, arg) for arg in s_args]
            for i, fut in enumerate(as_completed(futures), 1):
                cand_idx, score, theta_t2_2p, theta_s_2p, energies, delta_energies = fut.result()
                s_results[cand_idx] = (score, theta_t2_2p, theta_s_2p, energies, delta_energies)
                done_total = done_base + i
                if done_total == 1 or done_total % max(1, n_s // 10) == 0 or done_total == n_s:
                    log_fn(f"   S progress: {done_total}/{n_s} ({100.0 * done_total / n_s:.1f}%)")
                if checkpoint_fn is not None and (perf_counter() - last_chk) >= checkpoint_interval:
                    checkpoint_fn(s_results)
                    last_chk = perf_counter()

    accepted_by_rank: dict[int, list[tuple[float, Any, Any, str, float, float]]] = defaultdict(list)
    s_data: list[dict[str, Any]] = []

    for cand_idx, rank, t2_idx, s_type, s_idx, theta_t2, _ in s_candidates:
        score, theta_t2_2p, theta_s_2p, _, delta_energies = s_results[cand_idx]
        selected = score > threshold_s
        label = "S-hole" if s_type == "h" else "S-particle"
        s_ex = sh_excitations[s_idx] if s_type == "h" else sp_excitations[s_idx]

        s_data.append({"candidate_id": cand_idx, "parent_t2_id": t2_idx, "type": label, "operator_id": s_idx,
                       "excitation": str(s_ex), "delta_energies": delta_energies.tolist(), "weighted_sum": score,
                       "threshold": float(threshold_s), "selected": bool(selected), "theta_t2_1p": theta_t2,
                       "theta_t2_2p": theta_t2_2p, "theta_s_2p": theta_s_2p})
        if selected:
            s_pauli = sh_paulis[s_idx] if s_type == "h" else sp_paulis[s_idx]
            accepted_by_rank[rank].append((score, s_pauli, s_ex, label, theta_t2_2p, theta_s_2p))

    s_by_rank: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for rank, items in accepted_by_rank.items():
        items.sort(key=lambda item: item[0], reverse=True)
        s_by_rank[rank] = [(p, ex, lbl, th_t2, th_s) for _, p, ex, lbl, th_t2, th_s in items]
    return s_by_rank, s_data