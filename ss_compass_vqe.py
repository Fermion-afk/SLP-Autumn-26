#-------------------
# Imports
#-------------------
from ss_vqe import SS_VQE

import warnings
warnings.filterwarnings("ignore")
from typing import Any, cast

#Math Related Functionalities 
from math import comb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations

# Parallel processing and Logging
import os
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter


# Qiskit and Qiskit_algorithms (circuits and primitives)
from qiskit import transpile as qk_transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.circuit.library import StatePreparation
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import L_BFGS_B

# Qiskit Nature: electronic structure
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

def _all_singles(num_spatial_orbitals, num_particles):
    n = num_spatial_orbitals
    n_alpha, n_beta = num_particles
    occupied = list(range(n_alpha)) + list(range(n, n + n_beta))
    virtual = [i for i in range(2 * n) if i not in occupied]
    return [((i,), (a,)) for i in occupied for a in virtual]

def _all_doubles(num_spatial_orbitals, num_particles):
    n = num_spatial_orbitals
    n_alpha, n_beta = num_particles
    occupied = list(range(n_alpha)) + list(range(n, n + n_beta))
    virtual = [i for i in range(2 * n) if i not in occupied]
    return [(occ, virt) for occ in combinations(occupied, 2)
                       for virt in combinations(virtual, 2)]

def _Sh_ex(num_spatial_orbitals, num_particles):
    n = num_spatial_orbitals
    n_alpha, n_beta = num_particles
    occ_alpha = range(n_alpha)
    occ_beta = range(n, n + n_beta)
    vir_alpha = range(n_alpha, n)
    vir_beta = range(n + n_beta, 2*n)
    excitations = []
    # OP: (i_alpha, j_beta) -> (a_alpha, m_beta)
    for i in occ_alpha:
        for j in occ_beta:
            for a in vir_alpha:
                for m in occ_beta:
                    if m != j:
                        excitations.append(((i, j), (a, m)))
    # OP: (j_alpha, i_beta) -> (m_alpha, a_beta)
    for i in occ_beta:
        for j in occ_alpha:
            for a in vir_beta:
                for m in occ_alpha:
                    if m != j:
                        excitations.append(((j, i), (m, a)))
    return excitations

def _Sp_ex(num_spatial_orbitals, num_particles):
    n = num_spatial_orbitals
    n_alpha, n_beta = num_particles
    occ_alpha = range(n_alpha)
    occ_beta = range(n, n + n_beta)
    vir_alpha = range(n_alpha, n)
    vir_beta = range(n + n_beta, 2*n)
    excitations = []
    # OP: (i_alpha, e_beta) -> (a_alpha, b_beta)
    for i in occ_alpha:
        for e in vir_beta:
            for a in vir_alpha:
                for b in vir_beta:
                    if e != b:
                        excitations.append(((i, e), (a, b)))
    # OP: (i_beta, e_alpha) -> (a_beta, b_alpha)
    for i in occ_beta:
        for e in vir_alpha:
            for a in vir_beta:
                for b in vir_alpha:
                    if e != b:
                        excitations.append(((i, e), (a, b)))
    return excitations

# ------------------
# Helper functions
# ------------------
def _t2_screen_worker(args):
    idx, hamiltonian, transpiled_circuit, shift, weights, s_weights, initial_energies = args

    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    ss_vqe = SS_VQE(estimator, transpiled_circuit, weights, optimizer,
                    initial_point=np.array([0.0]))
    result = ss_vqe.compute_minimum_eigenvalue(hamiltonian)
    optimal_point = np.asarray(result.optimal_point).reshape(-1)
    theta_t2 = float(optimal_point[0])
    state_energies = _state_energies(estimator, transpiled_circuit, hamiltonian, optimal_point, shift)
    delta_energies = state_energies - initial_energies
    screening_score = float(np.dot(s_weights, delta_energies))

    return idx, screening_score, theta_t2, state_energies

def _s_screen_worker(args):
    idx, hamiltonian, transpiled_circuit, shift, weights, s_weights, baseline_energies, theta_t2 = args

    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    ss_vqe = SS_VQE(estimator, transpiled_circuit, weights, optimizer,
                    initial_point=np.array([theta_t2, 0.01]))
    result = ss_vqe.compute_minimum_eigenvalue(hamiltonian)
    optimal_point = np.asarray(result.optimal_point).reshape(-1)
    theta_s = float(optimal_point[1])
    state_energies = _state_energies(estimator, transpiled_circuit, hamiltonian, optimal_point, shift)
    delta_energies = state_energies - baseline_energies
    screening_score = float(np.dot(s_weights, delta_energies))

    return idx, screening_score, theta_s, state_energies

def _print_ansatz(op_info):
    n = len(op_info)
    print(f"\n{'='*75}")
    print(f"SS_COMPASS-VQE Final Ansatz  —  {n} operator(s)")
    print(f"{'='*75}")
    print(f"  {'#':>4}  {'Type':<14}  {'Excitation':<38}  {'θ_init':>10}")
    print(f"  {'—'*4}  {'—'*14}  {'—'*38}  {'—'*10}")
    for i, info in enumerate(op_info, 1):
        print(f"  {i:>4}  {info['type']:<14}  {str(info['excitation']):<38}  {info['theta']:>10.6f}")
    print(f"{'='*75}\n")

def _build_pauli_ops(ucc_form, mapper):
    return [mapper.map(op) for op in ucc_form.excitation_ops()]

def _state_energies(estimator, ansatze, operator, point=None, shift=0.0):
    """Return individual state energies <psi_i|H|psi_i> + shift."""
    pubs = [(ansatz, operator) if point is None else (ansatz, operator, point) for ansatz in ansatze]
    results = estimator.run(pubs).result()
    return np.asarray([float(np.real(r.data.evs)) + shift for r in results], dtype=float)

#------------------------------------
# Helper Functions for _init_states()
#------------------------------------

def _init_single_det_states(fermionic_op, mf, ncore, n_active, k):
    """Generate, rank and prepare the lowest-k single-determinant reference states."""
    n_spin_orbitals = 2 * n_active
    hf_det = _active_hf_determinant(mf, ncore, n_active)
    determinants = _cis_determinants(hf_det, n_spin_orbitals)

    # Diagonal elements of the projected Hamiltonian give the exact electronic energies
    H_single = _projected_hamiltonian(fermionic_op, determinants)
    diag_energies = np.real(H_single.diagonal())
    ranked = sorted(zip(diag_energies, determinants), key=lambda x: x[0])

    if not 1 <= k <= len(ranked):
        raise ValueError(f"Requested {k} states, but only {len(ranked)} determinants exist.")

    print("[SS-COMPASS] Ranked single determinants:")
    circuits = []
    for rank, (energy, det) in enumerate(ranked[:k], 1):
        print(f"  {rank}: determinant={det:0{n_spin_orbitals}b} | E_elec={energy:.10f} Ha")
        qc = QuantumCircuit(n_spin_orbitals, name=f"init_state_{rank-1}")
        for p in range(n_spin_orbitals):
            if (det >> p) & 1:
                qc.x(p)
        circuits.append(qc)

    return circuits

def _active_hf_determinant(mf, ncore, n_active):
    occ = np.asarray(mf.mo_occ[ncore:ncore + n_active])
    hf_occ = np.concatenate((occ >= 1, occ >= 2))
    hf_det = sum(1 << p for p in np.flatnonzero(hf_occ))
    return hf_det

def _cis_determinants(hf_det, n_spin_orbitals):
    occupied = [p for p in range(n_spin_orbitals) if (hf_det >> p) & 1]
    virtual = [p for p in range(n_spin_orbitals) if not (hf_det >> p) & 1]
    determinants = [hf_det]
    determinants.extend(hf_det ^ (1 << p) ^ (1 << q)
                    for p in occupied for q in virtual)
    return determinants

def _cisd_determinants(hf_det, n_spin_orbitals):
    occupied = [p for p in range(n_spin_orbitals) if (hf_det >> p) & 1]
    virtual = [p for p in range(n_spin_orbitals) if not (hf_det >> p) & 1]
    determinants = _cis_determinants(hf_det, n_spin_orbitals)
    determinants.extend(hf_det ^ sum(1 << p for p in occ) ^ sum(1 << p for p in virt)
                    for occ in combinations(occupied, 2)
                    for virt in combinations(virtual, 2))
    return determinants

def _projected_hamiltonian(fermionic_op, determinants):
    basis = {det: i for i, det in enumerate(determinants)}
    rows, cols, data = [], [], []

    for label, coeff in fermionic_op.terms():
        operators = [(token[0], int(token[2:])) for token in label.split()]
        for col, det in enumerate(determinants):
            target = det
            sign = 1
            for action, p in reversed(operators):
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

def _lowest_eigenvectors(H, k):
    dim = H.shape[0]
    if not 1 <= k <= dim:
        raise ValueError(f"Requested {k} states, but the determinant space has dimension {dim}.")

    if k >= dim - 1:
        energies, vectors = np.linalg.eigh(H.toarray())
        return energies[:k], vectors[:, :k]

    energies, vectors = eigsh(H, k=k, which="SA")
    order = np.argsort(energies)
    return energies[order], vectors[:, order]

def _determinants_to_circuits(determinants, coefficients, n_spin_orbitals):
    circuits = []
    for i in range(coefficients.shape[1]):
        statevector = np.zeros(2**n_spin_orbitals, dtype=complex)
        statevector[determinants] = coefficients[:, i]

        norm = np.linalg.norm(statevector)
        if norm < 1e-12:
            raise ValueError("Initial state has zero norm.")

        qc = QuantumCircuit(n_spin_orbitals, name=f"init_state_{i}")
        qc.append(StatePreparation(Statevector(statevector / norm)), range(n_spin_orbitals))
        circuits.append(qc)
    return circuits

def _init_cis_states(fermionic_op, mf, ncore, n_active, k):
    """Generate the lowest-k eigenstates in the complete single-excitation space."""
    if k < 1:
        raise ValueError("At least one reference state is required.")

    n_spin_orbitals = 2 * n_active
    hf_det = _active_hf_determinant(mf, ncore, n_active)
    determinants = _cis_determinants(hf_det, n_spin_orbitals)

    H_cis = _projected_hamiltonian(fermionic_op, determinants)
    energies, coefficients = _lowest_eigenvectors(H_cis, k)

    print(f"[SS-COMPASS] CIS determinant space dimension: {len(determinants)}")
    print(f"[SS-COMPASS] Lowest CIS energies: {energies}")

    return _determinants_to_circuits(determinants, coefficients, n_spin_orbitals)

def _init_cisd_states(fermionic_op, mf, ncore, n_active, k):
    """Generate the lowest-k eigenstates in the complete single/double-excitation space."""
    if k < 1:
        raise ValueError("At least one reference state is required.")

    n_spin_orbitals = 2 * n_active
    hf_det = _active_hf_determinant(mf, ncore, n_active)
    determinants = _cisd_determinants(hf_det, n_spin_orbitals)

    H_cisd = _projected_hamiltonian(fermionic_op, determinants)
    energies, coefficients = _lowest_eigenvectors(H_cisd, k)

    print(f"[SS-COMPASS] CISD determinant space dimension: {len(determinants)}")
    print(f"[SS-COMPASS] Lowest CISD energies: {energies}")

    return _determinants_to_circuits(determinants, coefficients, n_spin_orbitals)

# ----------------
# Timestamped progress logging
# ----------------
def _progress(message: str, start_time: float | None = None) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    elapsed = "" if start_time is None else f" | elapsed {perf_counter() - start_time:,.1f} s"
    print(f"[{stamp}{elapsed}] [SS-COMPASS] {message}", flush=True)
# ----------------
# Main Callable
# ----------------
def ss_compass_vqe(coordinates: str, d_eq: float, basis_set: str, no_init_states: int, weights: list,
                   charge: int = 0, spin: int = 0, init_states: list[QuantumCircuit] | None = None, method: str | None = "Single det", 
                   threshold: list = [1e-5,5e-6], active_space: list | None = None, s_weights: list | None = None) -> dict|None:

    run_start = perf_counter()
    _progress(f"Run started | geometry={d_eq} Å | basis={basis_set} | charge={charge} | spin={spin} | states={no_init_states}")

    N_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    
    if s_weights is None:
        sum_w = sum(weights)
        s_weights = [weight / sum_w for weight in weights[::-1]]

    if threshold is None or len(threshold) != 2:
        raise ValueError("threshold must be a length-2 list: [T2_threshold, S_threshold].")

    # Preserve the declared list types; use array views for numerical work.
    weights_array = np.asarray(weights, dtype=float)
    s_weights_array = np.asarray(s_weights, dtype=float)
    threshold_array = np.asarray(threshold, dtype=float)

    if len(weights_array) != no_init_states or len(s_weights_array) != no_init_states:
        raise ValueError("weights and s_weights must have length no_init_states.")

    if np.any(threshold_array < 0):
        raise ValueError("Screening thresholds must be non-negative.")
    
    # spawn creates fully fresh worker processes — avoids deadlocks from BLAS/Aer
    # C++ threads that are already running in the parent when the pool is created.
    # (fork inherits the parent's thread lock state without the actual threads,
    # which causes silent hangs inside Aer's statevector simulator.)
    _ctx = mp.get_context('spawn')
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    estimator = StatevectorEstimator()

    # ------------------------------------------------------------------
    # 1. Molecular system setup
    # ------------------------------------------------------------------
    stage_start = perf_counter()
    _progress("Stage 1/8: running PySCF driver and applying frozen-core transformation", run_start)
    driver = PySCFDriver(atom=coordinates, charge=charge, spin=spin, basis=basis_set)
    es_problem = cast(ElectronicStructureProblem, driver.run())
    es_problem = cast(ElectronicStructureProblem,
        FreezeCoreTransformer(freeze_core=True).transform(es_problem))
    mf = driver._calc   
    mapper = JordanWignerMapper()
    fermionic_op = es_problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)
    num_particles = es_problem.num_particles
    num_spin_orbitals = 2*cast(Any,es_problem.num_spatial_orbitals)
    if num_particles is None:
        raise ValueError("The number of particles can't be zero")
    # Generalize core spatial orbital calculation for both RHF and ROHF
    ncore = (mf.mol.nelectron - sum(num_particles)) // 2

    # Energy contributions omitted from the mapped active-space Hamiltonian.
    energy_constants = es_problem.hamiltonian.constants
    shift = float(np.real(sum(energy_constants.values())))

    nuclear_repulsion_energy = es_problem.nuclear_repulsion_energy
    if nuclear_repulsion_energy is None:
        raise ValueError("Nuclear repulsion energy is unavailable")
    nuclear_repulsion = float(nuclear_repulsion_energy)
    print(f"[Energy check] Nuclear repulsion: {nuclear_repulsion:.12f} Ha")
    print(f"[Energy check] Hamiltonian constants: {energy_constants}")
    print(f"[Energy check] Total constant shift: {shift:.12f} Ha")

    num_spatial_orbitals = int(cast(Any, es_problem.num_spatial_orbitals))
    if active_space is None:
        active_space_set = set(range(num_spatial_orbitals))
    else:
        active_space_set = set(active_space)
    _progress(f"Molecular Hamiltonian ready | spatial orbitals={num_spatial_orbitals} | particles={num_particles} | setup took {perf_counter()-stage_start:.1f} s", run_start)

    def _init_states(method, no_init_states, num_spatial_orbitals, num_particles):
        method = str(method).lower()
        if method in ("single det", "singledet"):
            return _init_single_det_states(fermionic_op, mf, ncore, num_spatial_orbitals, no_init_states)
        elif method == "cis":
            return _init_cis_states(fermionic_op, mf, ncore, num_spatial_orbitals, no_init_states)
        elif method == "cisd":
            return _init_cisd_states(fermionic_op, mf, ncore, num_spatial_orbitals, no_init_states)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    if init_states is None:
        init_states = _init_states(method, no_init_states, num_spatial_orbitals, num_particles)
    # Initial reference-state energies E_i^(0)
    initial_energies = _state_energies(estimator, init_states, qubit_op, shift=shift)
    print("[SS-COMPASS] Initial reference-state energies:")
    print(initial_energies, flush=True)
    _progress(f"Reference states ready | took {perf_counter()-stage_start:.1f} s", run_start)

    # Force circuit construction before sending it to worker processes
    _ = init_states[0].num_qubits
    print(f"Number of qubits:{init_states[0].num_qubits}")
    # ------------------------------------------------------------------
    # 2. Hartree-Fock Reference state for constructing Operator pools
    # ------------------------------------------------------------------
    # num_spatial_orbitals is already the correct int from Stage 1; pass it directly.
    init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)    # ------------------------------------------------------------------
    # 3. Exact spectrum in the complete fixed-N Hilbert space
    # ------------------------------------------------------------------
    stage_start = perf_counter()
    _progress("Stage 3/8: computing exact fixed-N eigenvalues", run_start)

    n_electrons = sum(num_particles)
    n_determinants = int(comb(num_spin_orbitals, n_electrons))

    if not 1 <= no_init_states <= n_determinants:
        raise ValueError(f"Requested {no_init_states} states, but the fixed-N space has dimension {n_determinants}.")

    # Construct Hamiltonian directly in the fixed-N determinant basis
    determinants = [sum(1 << p for p in occ)
                    for occ in combinations(range(num_spin_orbitals), n_electrons)]
    H_N = _projected_hamiltonian(fermionic_op, determinants)

    # Diagonalize the complete fixed-N Hamiltonian
    if no_init_states >= n_determinants - 1:
        if hasattr(H_N, "toarray"):
            H_N_dense: np.ndarray = np.asarray(H_N.toarray())
        else:
            H_N_dense = np.asarray(H_N)
        exact_electronic_energies = np.linalg.eigvalsh(H_N_dense)[:no_init_states]
    else:
        exact_electronic_energies = eigsh(H_N, k=no_init_states, which="SA", return_eigenvectors=False)
        exact_electronic_energies = np.sort(np.asarray(exact_electronic_energies))
    exact_energies = np.real(exact_electronic_energies) + shift

    print(f"[SS-COMPASS] Total electrons: {n_electrons}")
    print(f"[SS-COMPASS] Spin orbitals: {num_spin_orbitals}")
    print(f"[SS-COMPASS] Fixed-N determinant space dimension: {n_determinants}")
    print(f"First {len(exact_energies)} exact fixed-N total energies:")
    print(exact_energies, flush=True)

    _progress(f"Exact fixed-N spectrum ready | took {perf_counter()-stage_start:.1f} s", run_start)
    # ------------------------------------------------------------------
    # 4. Build operator pools
    # ------------------------------------------------------------------
    stage_start = perf_counter()
    _progress("Stage 4/8: constructing S/T2 operator pools", run_start)

    # Single excitations
    ucc_s = UCC(num_particles=num_particles,
        num_spatial_orbitals=num_spatial_orbitals,
        excitations=_all_singles, initial_state=init_state, qubit_mapper=mapper)
    s_excitations = ucc_s.excitation_list
    if s_excitations is None:
        raise ValueError("The UCC single-excitation pool is empty")
    s_paulis = _build_pauli_ops(ucc_s, mapper)

    # Double excitations
    ucc_d = UCC(num_particles=num_particles,
        num_spatial_orbitals=num_spatial_orbitals,
        excitations=_all_doubles, initial_state=init_state, qubit_mapper=mapper)
    t2_excitations = ucc_d.excitation_list
    if t2_excitations is None:
        raise ValueError("The UCC double-excitation pool is empty")
    t2_paulis = _build_pauli_ops(ucc_d, mapper)

    # Scatters (Hole)
    ucc_sh = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
            excitations=_Sh_ex, initial_state=init_state, qubit_mapper=mapper)
    sh_excitations = ucc_sh._get_excitation_list()
    if sh_excitations is None:
        raise ValueError("The UCC hole-excitation pool is empty")
    sh_paulis = _build_pauli_ops(ucc_sh,mapper)

    # Scatters (Particle)
    ucc_sp = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
            excitations=_Sp_ex, initial_state=init_state, qubit_mapper=mapper)
    sp_excitations = ucc_sp._get_excitation_list()
    if sp_excitations is None:
        raise ValueError("The UCC particle-excitation pool is empty")
    sp_paulis = _build_pauli_ops(ucc_sp, mapper)
    _progress(f"Operator pools ready | S={len(s_paulis)} T2={len(t2_paulis)} Sh={len(sh_paulis)} Sp={len(sp_paulis)} | took {perf_counter()-stage_start:.1f} s", run_start)

    #----------------------------------
    # 5. T2 screening
    #----------------------------------
    n_t2 = len(t2_paulis)
    n_states = len(init_states)

    print(f"[SS-COMPASS] Pre-compiling {n_t2} T2 operators for {n_states} reference states ...")
    t2_ansatze = [[EvolvedOperatorAnsatz([ops], initial_state=ref) for ref in init_states]
            for ops in t2_paulis]
    t2_ansatze_transpiled = [qk_transpile(cast(list[QuantumCircuit], ansatze), optimization_level=1)
            for ansatze in t2_ansatze]
    t2_args = [(i, qubit_op, t2_ansatze_transpiled[i], shift, weights_array, s_weights_array, initial_energies)
            for i in range(n_t2)]
    print(f"[SS-COMPASS] Screening {n_t2} T2 operators for {n_states} reference states ...")

    stage_start = perf_counter()
    t2_raw = []
    n_workers = min(N_workers, len(t2_args)) # Checks for no of parallel processors(workers) available 

    _progress(f"T2 screening started | workers={n_workers}", run_start)

    # Runs T2 screening parallely across several T2 operators
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as executor:
        futures = [executor.submit(_t2_screen_worker, args) for args in t2_args]

        for i, future in enumerate(as_completed(futures), 1):
            t2_raw.append(future.result())

            if i == 1 or i % max(1, n_t2 // 20) == 0 or i == n_t2:
                _progress(f"T2 screening: {i}/{n_t2} ({100*i/n_t2:.1f}%)", run_start)

    t2_screen = sorted([(score, idx, theta, energies) #Sorts and screens the operator based on the their Post optimization energy difference 
            for idx, score, theta, energies in t2_raw
            if abs(score) > threshold_array[0]],
            key=lambda item: abs(item[0]),
            reverse=True)
 
    _progress(f"T2 screening complete | accepted={len(t2_screen)}/{n_t2} "
            f"| stage took {perf_counter()-stage_start:.1f} s",
            run_start)

    # ------------------------------------------------------------------
    # 6. Determine S-operator candidates (pure Python — no VQE needed)
    #    For each sorted T2, collect unique S-hole/particle indices whose
    #    contractible orbital matches an active orbital in that T2.
    # ------------------------------------------------------------------
    # s_candidates: list of (cand_idx, rank, t2_idx, s_type, s_idx, theta_t2, E_t2)
    s_candidates = []

    # Occupied spin-orbitals set for robust contractible orbital identification
    n_alpha, n_beta = num_particles
    occupied_spin_orbitals = set(range(n_alpha)) | set(range(num_spatial_orbitals, num_spatial_orbitals + n_beta))

    for rank, (t2_score, t2_idx, theta_t2, t2_energies) in enumerate(t2_screen):
        t2_ex = t2_excitations[t2_idx]
        # Spatial orbitals involved in the T2 excitation
        t2_holes = {p % num_spatial_orbitals for p in t2_ex[0]}
        t2_particles = {p % num_spatial_orbitals for p in t2_ex[1]}

        # Active-space contractible orbitals
        active_holes = active_space_set.intersection(t2_holes)
        active_particles = active_space_set.intersection(t2_particles)

        sh_seen, sp_seen = set(), set()
        # S_h: match its contractible hole orbital (the occupied orbital in sh_ex[1]) with an active T2 hole
        for j, sh_ex in enumerate(sh_excitations):
            m_spin = next(p for p in sh_ex[1] if p in occupied_spin_orbitals)
            m = m_spin % num_spatial_orbitals
            if m in active_holes and j not in sh_seen:
                s_candidates.append((len(s_candidates), rank, t2_idx, 'h', j, theta_t2, t2_energies))
                sh_seen.add(j)
        # S_p: match its contractible particle orbital (the virtual orbital in sp_ex[0]) with an active T2 particle
        for j, sp_ex in enumerate(sp_excitations):
            e_spin = next(p for p in sp_ex[0] if p not in occupied_spin_orbitals)
            e = e_spin % num_spatial_orbitals
            if e in active_particles and j not in sp_seen:
                s_candidates.append((len(s_candidates), rank, t2_idx, 'p', j, theta_t2, t2_energies))
                sp_seen.add(j)
    # ------------------------------------------------------------------
    # 7. Build and screen S-hole / S-particle candidates
    #    Global dynamic scheduling across all T2 operators:
    #    workers pull independent (T2, S) screening tasks from one queue.
    # ------------------------------------------------------------------
    if s_candidates:
        n_s = len(s_candidates)

        print(f"[SS-COMPASS] Pre-compiling {n_s} S circuits for {n_states} reference states ...")

        s_ansatze = [[EvolvedOperatorAnsatz(
                          operators=[t2_paulis[t2_idx],
                                     sh_paulis[s_idx] if s_type == "h" else sp_paulis[s_idx]],
                          initial_state=ref)
                      for ref in init_states]
                     for (_, rank, t2_idx, s_type, s_idx, theta_t2, t2_energies) in s_candidates]

        s_transpiled_list = [qk_transpile(cast(list[QuantumCircuit], ansatze), optimization_level=1)
                             for ansatze in s_ansatze]

        s_args = [(cand_idx, qubit_op, s_transpiled_list[ci], shift, weights_array, s_weights_array,
                   t2_energies, theta_t2)
                  for ci, (cand_idx, rank, t2_idx, s_type, s_idx, theta_t2, t2_energies)
                  in enumerate(s_candidates)]

        n_workers = min(N_workers, len(s_args))
        print(f"[SS-COMPASS] Screening {n_s} S candidates across all T2 operators using {n_workers} workers ...")

        stage_start = perf_counter()
        s_raw = []

        # Submit all independent (T2, S) tasks to one shared worker pool.
        # Workers dynamically take the next available task as they finish.
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as executor:
            futures = [executor.submit(_s_screen_worker, args) for args in s_args]

            for i, future in enumerate(as_completed(futures), 1):
                s_raw.append(future.result())

                if i == 1 or i % max(1, n_s // 20) == 0 or i == n_s:
                    _progress(f"S screening: {i}/{n_s} ({100 * i / n_s:.1f}%)", run_start)

        _progress(f"S screening complete | screened={n_s} | workers={n_workers} "
                  f"| stage took {perf_counter() - stage_start:.1f} s", run_start)

        s_results = {cand_idx: (score, theta_s, state_energies)
                     for cand_idx, score, theta_s, state_energies in s_raw}

    else:
        s_results = {}

    s_by_rank = defaultdict(list)

    for cand_idx, rank, t2_idx, s_type, s_idx, theta_t2, t2_energies in s_candidates:
        screening_score, opt_s_theta, state_energies = s_results[cand_idx]

        if abs(screening_score) > threshold_array[1]:
            s_pauli = sh_paulis[s_idx] if s_type == "h" else sp_paulis[s_idx]
            s_ex = sh_excitations[s_idx] if s_type == "h" else sp_excitations[s_idx]
            label = "S-hole" if s_type == "h" else "S-particle"
            s_by_rank[rank].append((s_pauli, s_ex, label, opt_s_theta))

    n_s_accepted = sum(len(operators) for operators in s_by_rank.values())
    print(f"[SS-COMPASS] {n_s_accepted} S operators passed screening (threshold = {threshold_array[1]}).\n")

    # ------------------------------------------------------------------
    # 8. Assemble final ansatz: sorted T2 -> its S operators -> standard singles
    # ------------------------------------------------------------------
    final_paulis = []
    final_opt_t = []
    final_op_info = []

    for rank, (t2_score, t2_idx, theta_t2, t2_energies) in enumerate(t2_screen):
        final_paulis.append(t2_paulis[t2_idx])
        final_opt_t.append(theta_t2)
        final_op_info.append({'type': 'T2', 'excitation': t2_excitations[t2_idx], 'theta': theta_t2})

        for s_pauli, s_ex, label, opt_s_theta in s_by_rank.get(rank, []):
            final_paulis.append(s_pauli)
            final_opt_t.append(opt_s_theta)
            final_op_info.append({'type': label, 'excitation': s_ex, 'theta': opt_s_theta})

    for i, s_ex in enumerate(s_excitations):
        final_paulis.append(s_paulis[i])
        final_opt_t.append(0.01)
        final_op_info.append({'type': 'S-standard', 'excitation': s_ex, 'theta': 0.01})

    # Initial parameter vector shared across all reference-state ansatze
    ini_pt = np.array(final_opt_t, dtype=np.float64)

    print(f"[SS-COMPASS] Final ansatz contains {len(final_paulis)} operators "
        f"and {len(ini_pt)} parameters across {n_states} reference states.")
    
    _print_ansatz(final_op_info)

    stage_start = perf_counter()
    _progress("Stage 8/8: building final ansatz and starting global SS-VQE optimization", run_start)
    final_ansatze = [EvolvedOperatorAnsatz(final_paulis, initial_state=ref) for ref in init_states]
    final_transpiled = qk_transpile(cast(list[QuantumCircuit], final_ansatze), optimization_level=1)
    def _opt_callback(eval_count, params, cost, metadata):
        _progress(f"Optimizer evaluation {eval_count} | weighted cost={float(np.real(cost)):.10f} Ha", run_start)

    final_ss = SS_VQE(estimator, final_transpiled, weights_array, optimizer,
                      initial_point=ini_pt, callback=_opt_callback)
    final_result = final_ss.compute_minimum_eigenvalue(cast(Any,qubit_op))
    _progress(f"Global optimization finished | evaluations={final_result.cost_function_evals} | stage took {perf_counter()-stage_start:.1f} s", run_start)
    # per-reference-state energies at the optimum (weighted cost alone hides these)
    state_energies = _state_energies(estimator, final_result.optimal_circuits, qubit_op, shift=shift)
    print(f"geom- {d_eq}  parameters- {len(final_opt_t)}")
    for i, (e, ex) in enumerate(zip(state_energies, exact_energies)):
        print(f"  state {i}: SS-COMPASS {e:.8f}   FCI {ex:.8f}   err {e - ex:.2e} Ha")

    _progress(f"Run complete | total elapsed {perf_counter()-run_start:.1f} s", run_start)
    return {'geometry': d_eq,
            'energies': state_energies,
            'n_params': len(final_opt_t),
            'exact_energies': exact_energies,
            'initial_energies': initial_energies,
            'ansatz': final_ansatze,
            'optimal_params': final_result.optimal_point,
            'op_info': final_op_info}