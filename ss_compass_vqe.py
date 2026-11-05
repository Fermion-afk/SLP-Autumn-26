from ss_vqe import SS_VQE
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from time import perf_counter
from typing import Any, cast

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import os

from pyscf import ci, mcscf, cc
from pyscf.fci import cistring, direct_spin1
from qiskit.circuit.library import StatePreparation

import numpy as np
from qiskit import transpile as qk_transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.quantum_info import Statevector
from qiskit_algorithms import NumPyEigensolver

# Qiskit Nature: electronic structure
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

# Second quantization and fermion-to-qubit mapping
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Reference state and ansatz
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

# Ground-state solver interface
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver

def _Sh_ex(num_spatial_orbitals=None, num_particles=None, **kwargs):
    if num_spatial_orbitals is None:
        num_spatial_orbitals = kwargs["num_spin_orbitals"]

    n = num_spatial_orbitals
    if num_particles is None:
        raise ValueError("num_particles must be provided for _Sh_ex.")

    n_alpha, n_beta = num_particles

    occupied = list(range(n_alpha)) + list(range(n, n + n_beta))
    virtual = [i for i in range(2 * n) if i not in occupied]

    excitations = []

    for occ in combinations(occupied, 2):
        for virt in combinations(virtual, 2):
            if sum(i < n for i in occ) != sum(a < n for a in virt):
                continue
            if len(set(occ + virt)) == 4:
                excitations.append((occ, virt))

    return excitations

def _Sp_ex(num_spatial_orbitals=None, num_particles=None, **kwargs):
    if num_spatial_orbitals is None:
        num_spatial_orbitals = kwargs["num_spin_orbitals"]

    n = num_spatial_orbitals
    if num_particles is None:
        raise ValueError("num_particles must be provided for _Sp_ex.")

    n_alpha, n_beta = num_particles

    occupied = list(range(n_alpha)) + list(range(n, n + n_beta))
    virtual = [i for i in range(2 * n) if i not in occupied]

    excitations = []

    for occ in combinations(occupied, 2):
        for virt in combinations(virtual, 2):
            if sum(i < n for i in occ) != sum(a < n for a in virt):
                continue
            if len(set(occ + virt)) == 4:
                excitations.append((occ, virt))

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
    energies = []
    for ansatz in ansatze:
        pub = (ansatz, operator) if point is None else (ansatz, operator, point)
        energy = estimator.run([pub]).result()[0].data.evs
        energies.append(float(np.real(energy)) + shift)
    return np.asarray(energies, dtype=float)

#------------------------------------
# Helper Functions for _init_states()
#------------------------------------
def _number_operator_encoding(qubit_mapper, n_so):
    """Per spin orbital: n_i = c0 + c1 * Z_support under the given mapper (JW / Parity / BK)."""
    rows, n_qubits = [], None
    for i in range(n_so):
        q = qubit_mapper.map(FermionicOp({f"+_{i} -_{i}": 1.0}, num_spin_orbitals=n_so)).simplify()
        n_qubits = q.num_qubits
        c0, c1, supp = 0.0, 0.0, []
        for p, c in zip(q.paulis, q.coeffs):
            if p.x.any():
                raise ValueError("Number operator is not diagonal under the supplied mapper.")
            if not p.z.any():
                c0 += c.real
            else:
                c1 += c.real
                supp = list(np.flatnonzero(p.z))
        rows.append((supp, c0, c1))
    return rows, n_qubits

def _solve_gf2(A, y):
    m, n = A.shape
    M = np.concatenate([A % 2, (y % 2)[:, None]], axis=1).astype(np.uint8)
    pivots, r = [], 0
    for c in range(n):
        piv = next((i for i in range(r, m) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(m):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
    if any(M[i, :n].sum() == 0 and M[i, n] for i in range(m)):
        raise ValueError("Occupation is inconsistent with the mapper's particle sector.")
    b = np.zeros(n, dtype=np.uint8)
    for row, c in enumerate(pivots):
        b[c] = M[row, n]
    return b

def _mapped_bitstring(occ, enc, n_qubits):
    A, y = [], []
    for (supp, c0, c1), n_i in zip(enc, occ):
        if abs(c1) < 1e-12:
            continue
        row = np.zeros(n_qubits, dtype=np.uint8)
        row[supp] = 1
        A.append(row)
        y.append(0 if (n_i - c0) / c1 > 0 else 1)
    return _solve_gf2(np.array(A), np.array(y, dtype=np.uint8))
  
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
                   method: str = "HartreeFock",  threshold: list = [1e-5,5e-6],active_space: list | None = None,
                    s_weights: list | None = None) -> dict|None:

    run_start = perf_counter()
    _progress(f"Run started | geometry={d_eq} Å | basis={basis_set} | states={no_init_states}")

    N_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    
    if s_weights is None:
        sum_w = sum(w for w in weights)
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

    # ------------------------------------------------------------------
    # 1. Molecular system setup
    # ------------------------------------------------------------------
    stage_start = perf_counter()
    _progress("Stage 1/8: running PySCF driver and applying frozen-core transformation", run_start)
    driver = PySCFDriver(atom= coordinates,charge= 0,spin =0,basis = basis_set)
    es_problem = cast(ElectronicStructureProblem, driver.run())
    es_problem = cast(ElectronicStructureProblem,
        FreezeCoreTransformer(freeze_core=True).transform(es_problem))
    mf = driver._calc   
    mapper = JordanWignerMapper()
    fermionic_op = es_problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)
    num_particles = es_problem.num_particles
    num_spin_orbitals = 2*cast(Any,es_problem.num_spatial_orbitals)
    if num_particles is not None:
        num_pa = num_particles
    else:
        raise ValueError("The number of particles can't be zero")
        num_pa = 0
    ncore = mf.mol.nelectron // 2 - num_pa[0]

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

    def _init_states(method, no_init_states, num_spatial_orbitals, num_particles, qubit_mapper):

        n, (na, nb), k = num_spatial_orbitals, num_particles, no_init_states
        enc, nq = _number_operator_encoding(qubit_mapper, 2*n)
        sa = cistring.make_strings(range(n), na)
        sb = cistring.make_strings(range(n), nb)
        if k > len(sa) * len(sb):
            raise ValueError(f"Requested {k} states, but only {len(sa) * len(sb)} determinants exist.")

        m = str(method).lower()
        if m == "fci":                                   # exact lowest k roots in the active space
            mc = mcscf.CASCI(mf, n, (na, nb))
            mc_solver = cast(Any, mc)
            mc_solver.fcisolver = direct_spin1.FCI()
            mc_solver.fcisolver.nroots = k
            mc_solver.kernel()
            vecs = list(mc_solver.ci) if k > 1 else [mc_solver.ci]
        elif m == "cisd":                                # CISD roots, mapped to the FCI space
            c = ci.CISD(mf, frozen=ncore)
            c.nroots = k
            c.kernel()
            cs = c.ci if k > 1 else [c.ci]
            vecs = [ci.cisd.to_fcivec(v, mf.mo_coeff.shape[1], mf.mol.nelec, frozen=ncore) for v in cast(Any,cs)]

        elif m == "ccsd":                                # CCSD ground state (+ CISD roots for the rest)
            g = cc.CCSD(mf, frozen=ncore)
            g.kernel()
            t1, t2 = g.t1, g.t2
            c2 = t2 + np.einsum("ia,jb->ijab", cast(Any, t1), cast(Any, t1))   # exp(T)|HF> truncated at doubles
            v0 = ci.cisd.to_fcivec(ci.cisd.amplitudes_to_cisdvec(1.0, t1, c2),
                                   mf.mo_coeff.shape[1], mf.mol.nelec, frozen=ncore)
            vecs = [v0]
            if k > 1:                                    # CCSD has no excited roots here, so borrow CISD's
                c = ci.CISD(mf, frozen=ncore)
                c.nroots = k
                c.kernel()
                vecs += [ci.cisd.to_fcivec(v, mf.mo_coeff.shape[1], mf.mol.nelec, frozen=ncore) for v in cast(Any,c.ci)[1:]]
        
        else:                                            # default: HF, then determinants by excitation rank                                            # HF: lowest-k determinants by orbital-energy sum
            eps = mf.mo_energy[ncore:ncore + n]
            e_a = [sum(eps[p] for p in range(n) if (int(s) >> p) & 1) for s in sa]
            e_b = [sum(eps[p] for p in range(n) if (int(s) >> p) & 1) for s in sb]
            ranked = sorted((x + y, i, j) for i, x in enumerate(e_a) for j, y in enumerate(e_b))[:k]
            vecs = []
            for _, i, j in ranked:
                v = np.zeros((len(sa), len(sb)))
                v[i, j] = 1.0
                vecs.append(v)

        states = []
        for v in vecs:
            v = np.asarray(v).reshape(len(sa), len(sb))
            amp = np.zeros(2**cast(Any,nq), dtype=complex)
            for (i, j), a in np.ndenumerate(v):          # CI amplitude -> qubit basis state
                if abs(a) < 1e-10:
                    continue
                occ = [(int(sa[i]) >> p) & 1 for p in range(n)] + [(int(sb[j]) >> p) & 1 for p in range(n)]
                bits = _mapped_bitstring(occ, enc, nq)
                amp[sum(int(b) << q for q, b in enumerate(bits))] += a
            states.append(amp)

        Q, R = np.linalg.qr(np.array(states).T)          # enforce exact orthonormality
        Q = Q * np.sign(np.diag(R))
        circuits = []
        for i in range(k):
            qc = QuantumCircuit(nq, name=f"init_state_{i}")
            qc.append(StatePreparation(Statevector(Q[:, i])), range(cast(Any, nq)))
            circuits.append(qc)
        return circuits

    stage_start = perf_counter()
    _progress("Stage 2/8: constructing reference states", run_start)
    init_states = _init_states(method, no_init_states,num_spatial_orbitals,num_pa, mapper)
        # Initial reference-state energies E_i^(0)
    initial_energies = _state_energies(StatevectorEstimator(), init_states, qubit_op, shift=shift)
    print("[SS-COMPASS] Initial reference-state energies:")
    print(initial_energies, flush=True)
    _progress(f"Reference states ready | took {perf_counter()-stage_start:.1f} s", run_start)

    # Force circuit construction before sending it to worker processes
    _ = init_states[0].num_qubits
    print(f"Number of qubits:{init_states[0].num_qubits}")
    # ------------------------------------------------------------------
    # 2. Hartree-Fock energy
    # ------------------------------------------------------------------
    num_spatial_orbitals = es_problem.num_spatial_orbitals
    init_state = HartreeFock(num_spatial_orbitals,num_particles,mapper)
    # ------------------------------------------------------------------
    # 3. Exact (NumPy / FCI) energy spectrum
    # ------------------------------------------------------------------
    stage_start = perf_counter()
    _progress("Stage 3/8: computing exact active-space eigenvalues", run_start)

    numpy_solver = NumPyEigensolver(
        k=no_init_states,
        filter_criterion=cast(Any, es_problem.get_default_filter_criterion())
    )
    calc = ExcitedStatesEigensolver(mapper, numpy_solver)
    exact_result = calc.solve(es_problem)

    if exact_result.eigenvalues is not None:
        exact_electronic_energies = np.real(exact_result.eigenvalues)
        exact_energies = exact_electronic_energies + shift
    else:
        raise RuntimeError("The FCI calculation didn't return any eigenvalues.")

    print(f"First {len(exact_energies)} FCI total energies:")
    print(exact_energies, flush=True)

    _progress(f"Exact spectrum ready | took {perf_counter()-stage_start:.1f} s", run_start)
    # ------------------------------------------------------------------
    # 4. Build operator pools
    # ------------------------------------------------------------------
    stage_start = perf_counter()
    _progress("Stage 4/8: constructing S/T2 operator pools", run_start)
    ucc_s = UCC(num_particles=num_particles,
        num_spatial_orbitals=es_problem.num_spatial_orbitals,
        excitations="s",initial_state=init_state,qubit_mapper=mapper)
    s_excitations = ucc_s.excitation_list
    if s_excitations is None:
        raise ValueError("The UCC single-excitation pool is empty")
    s_paulis = _build_pauli_ops(ucc_s, mapper)

    ucc_d = UCC(num_particles=num_particles,
        num_spatial_orbitals=es_problem.num_spatial_orbitals,
        excitations="d",initial_state=init_state,qubit_mapper=mapper)
    t2_excitations = ucc_d.excitation_list
    if t2_excitations is None:
        raise ValueError("The UCC double-excitation pool is empty")
    t2_paulis = _build_pauli_ops(ucc_d, mapper)

    ucc_sh = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
            excitations=_Sh_ex, initial_state=init_state, qubit_mapper=mapper)
    sh_excitations = ucc_sh._get_excitation_list()
    if sh_excitations is None:
        raise ValueError("The UCC hole-excitation pool is empty")
    sh_paulis = _build_pauli_ops(ucc_sh,mapper)

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
    n_workers = min(N_workers, len(t2_args))

    _progress(f"T2 screening started | workers={n_workers}", run_start)

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as executor:
        futures = [executor.submit(_t2_screen_worker, args) for args in t2_args]

        for i, future in enumerate(as_completed(futures), 1):
            t2_raw.append(future.result())

            if i == 1 or i % max(1, n_t2 // 20) == 0 or i == n_t2:
                _progress(f"T2 screening: {i}/{n_t2} ({100*i/n_t2:.1f}%)", run_start)

    t2_screen = sorted([(score, idx, theta, energies)
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

    for rank, (t2_score, t2_idx, theta_t2, t2_energies) in enumerate(t2_screen):
        if t2_excitations is not None:
            t2_ex = t2_excitations[t2_idx]
            t2_flat = [t2_ex[0][0], t2_ex[1][0], t2_ex[0][1], t2_ex[1][1]]
            active_in_t2 = active_space_set.intersection(t2_flat)
            if not active_in_t2:
                continue
        else:
            raise ValueError("the pool of t2 excitations are empty")

        sh_seen = set()
        sp_seen = set()
        for act_orb in active_in_t2:
            for j, sh_ex in enumerate(sh_excitations):
                if j not in sh_seen and act_orb in sh_ex[1]:
                    s_candidates.append((len(s_candidates), rank, t2_idx, 'h', j, theta_t2, t2_energies))
                    sh_seen.add(j)
            for j, sp_ex in enumerate(sp_excitations):
                if j not in sp_seen and act_orb in sp_ex[0]:
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

        selected_s_paulis = [sh_paulis[s_idx] if s_type == "h" else sp_paulis[s_idx]
                             for _, rank, t2_idx, s_type, s_idx, theta_t2, t2_energies in s_candidates]

        s_ansatze = [[EvolvedOperatorAnsatz(operators=[t2_paulis[t2_idx], s_pauli], initial_state=ref)
                      for ref in init_states]
                     for (_, rank, t2_idx, s_type, s_idx, theta_t2, t2_energies), s_pauli
                     in zip(s_candidates, selected_s_paulis)]

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

    final_ss = SS_VQE(StatevectorEstimator(), final_transpiled, weights_array, optimizer,
                      initial_point=ini_pt, callback=_opt_callback)
    final_result = final_ss.compute_minimum_eigenvalue(cast(Any,qubit_op))
    _progress(f"Global optimization finished | evaluations={final_result.cost_function_evals} | stage took {perf_counter()-stage_start:.1f} s", run_start)
    # per-reference-state energies at the optimum (weighted cost alone hides these)
    state_energies = _state_energies(StatevectorEstimator(), final_result.optimal_circuits, qubit_op, shift=shift)
    print(f"geom- {d_eq}  parameters- {len(final_opt_t)}")
    for i, (e, ex) in enumerate(zip(state_energies, exact_energies)):
        print(f"  state {i}: SS-COMPASS {e:.8f}   FCI {ex:.8f}   err {e - ex:.2e} Ha")

    _progress(f"Run complete | total elapsed {perf_counter()-run_start:.1f} s", run_start)
    return {
        'geometry': d_eq,
        'energies': state_energies,
        'n_params': len(final_opt_t),
        'exact_energies': exact_energies,
        'initial_energies': initial_energies,
        'ansatz': final_ansatze,
        'optimal_params': final_result.optimal_point,
        'op_info': final_op_info,
    }