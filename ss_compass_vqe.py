"""SS-COMPASS-VQE: Subspace-Search COMPASS Variational Quantum Eigensolver."""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from math import comb
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np
from scipy.sparse.linalg import eigsh
from qiskit import transpile as qk_transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

from compass_states import (
    all_doubles,
    all_singles,
    build_initial_states,
    projected_hamiltonian,
    scatter_hole_ex,
    scatter_particle_ex,
)
from ss_vqe import SS_VQE

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Screening Workers & Evaluation Utilities
# ------------------------------------------------------------------
def _state_energies(estimator, ansatze, operator, point=None, shift: float = 0.0) -> np.ndarray:
    pubs = [(a, operator) if point is None else (a, operator, point) for a in ansatze]
    return np.asarray([float(np.real(r.data.evs)) + shift for r in estimator.run(pubs).result()], dtype=float)

def _run_screen_job(
    hamiltonian, circuits, shift: float, weights, s_weights, baseline_energies, init_pt: np.ndarray, param_idx: int
):
    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    res = SS_VQE(cast(Any, estimator), circuits, weights, optimizer, initial_point=init_pt).compute_minimum_eigenvalue(
        hamiltonian
    )
    opt_pt = np.asarray(res.optimal_point).reshape(-1)
    energies = _state_energies(estimator, circuits, hamiltonian, opt_pt, shift)
    score = float(np.dot(s_weights, energies - baseline_energies))
    return score, float(opt_pt[param_idx]), energies

def _t2_screen_worker(args):
    idx, hamiltonian, circuits, shift, weights, s_weights, initial_energies = args
    score, theta_t2, energies = _run_screen_job(
        hamiltonian, circuits, shift, weights, s_weights, initial_energies, np.array([0.0]), 0
    )
    return idx, score, theta_t2, energies

def _s_screen_worker(args):
    idx, hamiltonian, circuits, shift, weights, s_weights, baseline_energies, theta_t2 = args
    score, theta_s, energies = _run_screen_job(
        hamiltonian, circuits, shift, weights, s_weights, baseline_energies, np.array([theta_t2, 0.01]), 1
    )
    return idx, score, theta_s, energies

def _run_parallel_screening(worker_fn: Callable, task_args: list, label: str, max_workers: int, ctx, run_start: float):
    raw_results = []
    total = len(task_args)
    step = max(1, total // 20)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(worker_fn, arg) for arg in task_args]
        for i, future in enumerate(as_completed(futures), 1):
            raw_results.append(future.result())
            if i == 1 or i % step == 0 or i == total:
                _progress(f"{label}: {i}/{total} ({100 * i / total:.1f}%)", run_start)
    return raw_results

def _print_ansatz(op_info: list[dict[str, Any]]) -> None:
    print(f"\n{'='*75}\nSS_COMPASS-VQE Final Ansatz  —  {len(op_info)} operator(s)\n{'='*75}")
    print(f"  {'#':>4}  {'Type':<14}  {'Excitation':<38}  {'θ_init':>10}")
    print(f"  {'—'*4}  {'—'*14}  {'—'*38}  {'—'*10}")
    for i, info in enumerate(op_info, 1):
        print(f"  {i:>4}  {info['type']:<14}  {str(info['excitation']):<38}  {info['theta']:>10.6f}")
    print(f"{'='*75}\n")

def _progress(message: str, start_time: float | None = None) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    elapsed = "" if start_time is None else f" | elapsed {perf_counter() - start_time:,.1f} s"
    print(f"[{stamp}{elapsed}] [SS-COMPASS] {message}", flush=True)

# ------------------------------------------------------------------
# Main SS-COMPASS-VQE Driver
# ------------------------------------------------------------------
def ss_compass_vqe(
    coordinates: str,
    d_eq: float,
    basis_set: str,
    no_init_states: int,
    weights: list,
    charge: int = 0,
    spin: int = 0,
    init_states: list[QuantumCircuit] | None = None,
    method: str | None = "dpt1",
    threshold: list = [1e-5, 5e-6],
    active_space: list | None = None,
    s_weights: list | None = None,
) -> dict | None:
    run_start = perf_counter()
    _progress(f"Run started | geometry={d_eq} Å | basis={basis_set} | charge={charge} | spin={spin} | states={no_init_states}")

    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    n_cpu = len(cast(Any, sched_getaffinity(0))) if callable(sched_getaffinity) else (os.cpu_count() or 1)

    if s_weights is None:
        sum_w = sum(weights)
        s_weights = [w / sum_w for w in weights[::-1]]
    if threshold is None or len(threshold) != 2:
        raise ValueError("threshold must be a length-2 list: [T2_threshold, S_threshold].")

    weights_arr = np.asarray(weights, dtype=float)
    s_weights_arr = np.asarray(s_weights, dtype=float)
    thresh_arr = np.asarray(threshold, dtype=float)
    if len(weights_arr) != no_init_states or len(s_weights_arr) != no_init_states:
        raise ValueError("weights and s_weights must have length no_init_states.")
    if np.any(thresh_arr < 0):
        raise ValueError("Screening thresholds must be non-negative.")

    ctx = mp.get_context("spawn")
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    estimator = StatevectorEstimator()

    # 1. Molecular system & reference states
    t0 = perf_counter()
    _progress("Stage 1/8: running PySCF driver and applying frozen-core transformation", run_start)
    driver = PySCFDriver(atom=coordinates, charge=charge, spin=spin, basis=basis_set)
    es_problem = cast(ElectronicStructureProblem, FreezeCoreTransformer(freeze_core=True).transform(driver.run()))
    mf = driver._calc
    mapper = JordanWignerMapper()
    fermionic_op = es_problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)

    num_particles = es_problem.num_particles
    if num_particles is None:
        raise ValueError("The number of particles can't be zero")
    num_spatial_orbitals = int(cast(Any, es_problem.num_spatial_orbitals))
    num_spin_orbitals = 2 * num_spatial_orbitals
    ncore = (mf.mol.nelectron - sum(num_particles)) // 2

    energy_constants = es_problem.hamiltonian.constants
    shift = float(np.real(sum(energy_constants.values())))
    if es_problem.nuclear_repulsion_energy is None:
        raise ValueError("Nuclear repulsion energy is unavailable")
    print(f"[Energy check] Nuclear repulsion: {float(es_problem.nuclear_repulsion_energy):.12f} Ha")
    print(f"[Energy check] Hamiltonian constants: {energy_constants}")
    print(f"[Energy check] Total constant shift: {shift:.12f} Ha")

    active_space_set = set(range(num_spatial_orbitals) if active_space is None else active_space)
    _progress(f"Molecular Hamiltonian ready | spatial orbitals={num_spatial_orbitals} | particles={num_particles} | setup took {perf_counter()-t0:.1f} s", run_start)

    if init_states is None:
        init_states = build_initial_states(method, fermionic_op, mf, ncore, num_spatial_orbitals, no_init_states)
    initial_energies = _state_energies(estimator, init_states, qubit_op, shift=shift)
    print(f"[SS-COMPASS] Initial reference-state energies:\n{initial_energies}", flush=True)
    _progress(f"Reference states ready | took {perf_counter()-t0:.1f} s", run_start)
    print(f"Number of qubits:{init_states[0].num_qubits}")

    # 2. Hartree-Fock state for UCC operator pools
    hf_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)

    # 3. Exact fixed-N spectrum
    t0 = perf_counter()
    _progress("Stage 3/8: computing exact fixed-N eigenvalues", run_start)
    n_electrons = sum(num_particles)
    n_dets = int(comb(num_spin_orbitals, n_electrons))
    if not 1 <= no_init_states <= n_dets:
        raise ValueError(f"Requested {no_init_states} states, but the fixed-N space has dimension {n_dets}.")

    dets = [sum(1 << p for p in occ) for occ in combinations(range(num_spin_orbitals), n_electrons)]
    H_N = projected_hamiltonian(fermionic_op, dets)
    if no_init_states >= n_dets - 1:
        exact_elec = np.linalg.eigvalsh(np.asarray(H_N.toarray()))[:no_init_states]
    else:
        exact_elec = np.sort(np.asarray(eigsh(H_N, k=no_init_states, which="SA", return_eigenvectors=False)))
    exact_energies = np.real(exact_elec) + shift

    print(f"[SS-COMPASS] Total electrons: {n_electrons}\n[SS-COMPASS] Spin orbitals: {num_spin_orbitals}")
    print(f"[SS-COMPASS] Fixed-N determinant space dimension: {n_dets}")
    print(f"First {len(exact_energies)} exact fixed-N total energies:\n{exact_energies}", flush=True)
    _progress(f"Exact fixed-N spectrum ready | took {perf_counter()-t0:.1f} s", run_start)

    # 4. Build S, T2, Sh, and Sp operator pools
    t0 = perf_counter()
    _progress("Stage 4/8: constructing S/T2 operator pools", run_start)

    def _build_pool(ex_fn, label: str, use_private_getter: bool = False):
        ucc = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
                  excitations=ex_fn, initial_state=hf_state, qubit_mapper=mapper)
        ex_list = ucc._get_excitation_list() if use_private_getter else ucc.excitation_list
        if ex_list is None:
            raise ValueError(f"The UCC {label}-excitation pool is empty")
        return ex_list, [mapper.map(cast(Any, op)) for op in ucc.excitation_ops()]

    s_excitations, s_paulis = _build_pool(all_singles, "single")
    t2_excitations, t2_paulis = _build_pool(all_doubles, "double")
    sh_excitations, sh_paulis = _build_pool(scatter_hole_ex, "hole", use_private_getter=True)
    sp_excitations, sp_paulis = _build_pool(scatter_particle_ex, "particle", use_private_getter=True)
    _progress(f"Operator pools ready | S={len(s_paulis)} T2={len(t2_paulis)} Sh={len(sh_paulis)} Sp={len(sp_paulis)} | took {perf_counter()-t0:.1f} s", run_start)

    # 5. Parallel T2 screening
    n_t2, n_states = len(t2_paulis), len(init_states)
    print(f"[SS-COMPASS] Pre-compiling {n_t2} T2 operators for {n_states} reference states ...")
    t2_transpiled = [
        qk_transpile(cast(list[QuantumCircuit], [EvolvedOperatorAnsatz([op], initial_state=r) for r in init_states]), optimization_level=1)
        for op in t2_paulis
    ]
    t2_args = [(i, qubit_op, t2_transpiled[i], shift, weights_arr, s_weights_arr, initial_energies) for i in range(n_t2)]

    print(f"[SS-COMPASS] Screening {n_t2} T2 operators for {n_states} reference states ...")
    t0 = perf_counter()
    n_workers = min(n_cpu, len(t2_args))
    _progress(f"T2 screening started | workers={n_workers}", run_start)
    t2_raw = _run_parallel_screening(_t2_screen_worker, t2_args, "T2 screening", n_workers, ctx, run_start)

    t2_screen = sorted(
        [(score, idx, theta, energies) for idx, score, theta, energies in t2_raw if abs(score) > thresh_arr[0]],
        key=lambda item: abs(item[0]),
        reverse=True,
    )
    _progress(f"T2 screening complete | accepted={len(t2_screen)}/{n_t2} | stage took {perf_counter()-t0:.1f} s", run_start)

    # 6. Identify contractible S-hole / S-particle candidates per accepted T2
    s_candidates = []
    n_a, n_b = num_particles
    occ_spin = set(range(n_a)) | set(range(num_spatial_orbitals, num_spatial_orbitals + n_b))

    for rank, (_, t2_idx, theta_t2, t2_energies) in enumerate(t2_screen):
        t2_ex = t2_excitations[t2_idx]
        active_holes = active_space_set.intersection(p % num_spatial_orbitals for p in t2_ex[0])
        active_particles = active_space_set.intersection(p % num_spatial_orbitals for p in t2_ex[1])

        sh_seen, sp_seen = set(), set()
        for j, sh_ex in enumerate(sh_excitations):
            m = next(p for p in sh_ex[1] if p in occ_spin) % num_spatial_orbitals
            if m in active_holes and j not in sh_seen:
                s_candidates.append((len(s_candidates), rank, t2_idx, "h", j, theta_t2, t2_energies))
                sh_seen.add(j)
        for j, sp_ex in enumerate(sp_excitations):
            e = next(p for p in sp_ex[0] if p not in occ_spin) % num_spatial_orbitals
            if e in active_particles and j not in sp_seen:
                s_candidates.append((len(s_candidates), rank, t2_idx, "p", j, theta_t2, t2_energies))
                sp_seen.add(j)

    # 7. Parallel S-operator screening
    s_results = {}
    if s_candidates:
        n_s = len(s_candidates)
        print(f"[SS-COMPASS] Pre-compiling {n_s} S circuits for {n_states} reference states ...")
        s_transpiled = [
            qk_transpile(
                cast(
                    list[QuantumCircuit],
                    [EvolvedOperatorAnsatz([t2_paulis[t2_i], sh_paulis[s_i] if s_t == "h" else sp_paulis[s_i]], initial_state=r) for r in init_states],
                ),
                optimization_level=1,
            )
            for (_, _, t2_i, s_t, s_i, _, _) in s_candidates
        ]
        s_args = [
            (ci, qubit_op, s_transpiled[i], shift, weights_arr, s_weights_arr, t2_e, th_t2)
            for i, (ci, _, _, _, _, th_t2, t2_e) in enumerate(s_candidates)
        ]
        n_workers = min(n_cpu, len(s_args))
        print(f"[SS-COMPASS] Screening {n_s} S candidates across all T2 operators using {n_workers} workers ...")
        t0 = perf_counter()
        s_raw = _run_parallel_screening(_s_screen_worker, s_args, "S screening", n_workers, ctx, run_start)
        _progress(f"S screening complete | screened={n_s} | workers={n_workers} | stage took {perf_counter()-t0:.1f} s", run_start)
        s_results = {ci: (score, th_s, energies) for ci, score, th_s, energies in s_raw}

    s_by_rank = defaultdict(list)
    for ci, rank, _, s_t, s_i, _, _ in s_candidates:
        score, opt_s_theta, _ = s_results[ci]
        if abs(score) > thresh_arr[1]:
            s_by_rank[rank].append((
                sh_paulis[s_i] if s_t == "h" else sp_paulis[s_i],
                sh_excitations[s_i] if s_t == "h" else sp_excitations[s_i],
                "S-hole" if s_t == "h" else "S-particle",
                opt_s_theta,
            ))
    print(f"[SS-COMPASS] {sum(len(v) for v in s_by_rank.values())} S operators passed screening (threshold = {thresh_arr[1]}).\n")

    # 8. Assemble final ansatz & run global SS-VQE optimization
    final_paulis, final_opt_t, final_op_info = [], [], []
    for rank, (_, t2_idx, theta_t2, _) in enumerate(t2_screen):
        final_paulis.append(t2_paulis[t2_idx])
        final_opt_t.append(theta_t2)
        final_op_info.append({"type": "T2", "excitation": t2_excitations[t2_idx], "theta": theta_t2})
        for s_pauli, s_ex, label, opt_s_theta in s_by_rank.get(rank, []):
            final_paulis.append(s_pauli)
            final_opt_t.append(opt_s_theta)
            final_op_info.append({"type": label, "excitation": s_ex, "theta": opt_s_theta})

    for s_pauli, s_ex in zip(s_paulis, s_excitations):
        final_paulis.append(s_pauli)
        final_opt_t.append(0.01)
        final_op_info.append({"type": "S-standard", "excitation": s_ex, "theta": 0.01})

    ini_pt = np.asarray(final_opt_t, dtype=np.float64)
    print(f"[SS-COMPASS] Final ansatz contains {len(final_paulis)} operators and {len(ini_pt)} parameters across {n_states} reference states.")
    _print_ansatz(final_op_info)

    t0 = perf_counter()
    _progress("Stage 8/8: building final ansatz and starting global SS-VQE optimization", run_start)
    final_ansatze = [EvolvedOperatorAnsatz(final_paulis, initial_state=r) for r in init_states]
    final_transpiled = qk_transpile(cast(list[QuantumCircuit], final_ansatze), optimization_level=1)

    final_result = SS_VQE(
        estimator,
        final_transpiled,
        weights_arr,
        optimizer,
        initial_point=ini_pt,
        callback=lambda c, _, cost, __: _progress(f"Optimizer evaluation {c} | weighted cost={float(np.real(cost)):.10f} Ha", run_start),
    ).compute_minimum_eigenvalue(cast(Any, qubit_op))
    _progress(f"Global optimization finished | evaluations={final_result.cost_function_evals} | stage took {perf_counter()-t0:.1f} s", run_start)

    state_energies = _state_energies(estimator, final_result.optimal_circuits, qubit_op, shift=shift)
    print(f"geom- {d_eq}  parameters- {len(final_opt_t)}")
    for i, (e, ex) in enumerate(zip(state_energies, exact_energies)):
        print(f"  state {i}: SS-COMPASS {e:.8f}   FCI {ex:.8f}   err {e - ex:.2e} Ha")

    _progress(f"Run complete | total elapsed {perf_counter()-run_start:.1f} s", run_start)
    return {
        "geometry": d_eq,
        "energies": state_energies,
        "n_params": len(final_opt_t),
        "exact_energies": exact_energies,
        "initial_energies": initial_energies,
        "ansatz": final_ansatze,
        "optimal_params": final_result.optimal_point,
        "op_info": final_op_info,
    }