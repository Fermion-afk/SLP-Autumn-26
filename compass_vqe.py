import warnings
warnings.filterwarnings("ignore")

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from qiskit import transpile as qk_transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.transpiler import PassManager

# Qiskit Nature: electronic structure
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

# Second quantization and fermion-to-qubit mapping
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Reference state and ansatz
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

# Ground-state solver interface
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# ---------------------------------------------------------------------------
# S-type excitation definitions (fixed for BeH2 STO-3G frozen-core)
# ---------------------------------------------------------------------------

def _Sh_ex(num_spin_orbitals, num_particles):
    return [
        ((6,0),(8,1)),  ((6,1),(8,1)),  ((6,0),(9,1)),  ((6,1),(9,1)),
        ((6,0),(10,1)), ((6,1),(10,1)), ((6,0),(11,1)), ((6,1),(11,1)),
        ((7,0),(8,1)),  ((7,1),(8,1)),  ((7,0),(9,1)),  ((7,1),(9,1)),
        ((7,0),(10,1)), ((7,1),(10,1)), ((7,0),(11,1)), ((7,1),(11,1)),
        ((0,6),(2,7)),  ((0,7),(2,7)),  ((0,6),(3,7)),  ((0,7),(3,7)),
        ((0,6),(4,7)),  ((0,7),(4,7)),  ((0,6),(5,7)),  ((0,7),(5,7)),
        ((1,6),(2,7)),  ((1,7),(2,7)),  ((1,6),(3,7)),  ((1,7),(3,7)),
        ((1,6),(4,7)),  ((1,7),(4,7)),  ((1,6),(5,7)),  ((1,7),(5,7))]

def _Sp_ex(num_spin_orbitals, num_particles):
    return [
        ((6,2),(8,2)),  ((6,2),(8,3)),  ((6,2),(8,4)),  ((6,2),(8,5)),
        ((6,2),(9,2)),  ((6,2),(9,3)),  ((6,2),(9,4)),  ((6,2),(9,5)),
        ((6,2),(10,2)), ((6,2),(10,3)), ((6,2),(10,4)), ((6,2),(10,5)),
        ((6,2),(11,2)), ((6,2),(11,3)), ((6,2),(11,4)), ((6,2),(11,5)),
        ((7,2),(8,2)),  ((7,2),(8,3)),  ((7,2),(8,4)),  ((7,2),(8,5)),
        ((7,2),(9,2)),  ((7,2),(9,3)),  ((7,2),(9,4)),  ((7,2),(9,5)),
        ((7,2),(10,2)), ((7,2),(10,3)), ((7,2),(10,4)), ((7,2),(10,5)),
        ((7,2),(11,2)), ((7,2),(11,3)), ((7,2),(11,4)), ((7,2),(11,5)),
        ((0,8),(2,8)),  ((0,8),(2,9)),  ((0,8),(2,10)), ((0,8),(2,11)),
        ((0,8),(3,8)),  ((0,8),(3,9)),  ((0,8),(3,10)), ((0,8),(3,11)),
        ((0,8),(4,8)),  ((0,8),(4,9)),  ((0,8),(4,10)), ((0,8),(4,11)),
        ((0,8),(5,8)),  ((0,8),(5,9)),  ((0,8),(5,10)), ((0,8),(5,11)),
        ((1,8),(2,8)),  ((1,8),(2,9)),  ((1,8),(2,10)), ((1,8),(2,11)),
        ((1,8),(3,8)),  ((1,8),(3,9)),  ((1,8),(3,10)), ((1,8),(3,11)),
        ((1,8),(4,8)),  ((1,8),(4,9)),  ((1,8),(4,10)), ((1,8),(4,11)),
        ((1,8),(5,8)),  ((1,8),(5,9)),  ((1,8),(5,10)), ((1,8),(5,11))]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pauli_ops(ucc_form, mapper):
    """New: mapper.map(ex) instead of converter.convert(ex)"""
    return [mapper.map(ex) for ex in ucc_form.excitation_ops()]

# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for multiprocessing)
# ---------------------------------------------------------------------------

def _t2_screen_worker(args):
    cand_idx, transpiled_circ, qubit_op, shift = args
    estimator = StatevectorEstimator()
    vqe = VQE(estimator, transpiled_circ, L_BFGS_B(maxfun=200_000, maxiter=10_000), initial_point=np.array([0.0]))
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    eigenvalue = result.eigenvalue
    if eigenvalue is None:
        raise RuntimeError("VQE did not return an eigenvalue")
    optimal_point = result.optimal_point
    if optimal_point is None:
        raise RuntimeError("VQE did not return an optimal point")
    return cand_idx, float(np.real(eigenvalue)) + shift, float(np.asarray(optimal_point).reshape(-1)[0])

def _s_screen_worker(args):
    cand_idx, transpiled_circ, theta_t2, qubit_op, shift = args
    estimator = StatevectorEstimator()
    vqe = VQE(estimator, transpiled_circ, L_BFGS_B(maxfun=200_000, maxiter=10_000), initial_point=np.array([theta_t2, 0.01]))
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    eigenvalue = result.eigenvalue
    if eigenvalue is None:
        raise RuntimeError("VQE did not return an Eigenvalue")
    optimal_point = result.optimal_point
    if optimal_point is None:
        raise RuntimeError("VQE did not return an optimal point")
    return cand_idx, float(np.real(eigenvalue)) + shift, float(np.asarray(optimal_point).reshape(-1)[1])

# ---------------------------------------------------------------------------
# Ansatz printer
# ---------------------------------------------------------------------------

def _print_ansatz(op_info):
    n = len(op_info)
    print(f"\n{'='*75}")
    print(f"  COMPASS-VQE Final Ansatz  —  {n} operator(s)")
    print(f"{'='*75}")
    print(f"  {'#':>4}  {'Type':<14}  {'Excitation':<38}  {'θ_init':>10}")
    print(f"  {'—'*4}  {'—'*14}  {'—'*38}  {'—'*10}")
    for i, info in enumerate(op_info, 1):
        print(f"  {i:>4}  {info['type']:<14}  {str(info['excitation']):<38}  {info['theta']:>10.6f}")
    print(f"{'='*75}\n")

# ---------------------------------------------------------------------------
# Main callable
# ---------------------------------------------------------------------------

def compass_vqe(coord, d_eq, basis='sto3g', active_space=None, n_workers=None):
    """
    COMPASS-VQE ansatz builder and energy solver with parallel operator screening.
    T2 screening and S-operator screening are both parallelised across CPU cores.
    Parameters
    ----------
    coord : str
        Molecular geometry in PySCF format, e.g. "Be 0 0 0; H 0 0 1.3; H 0 0 -1.3"
    d_eq : float
        Geometry label used in printed output.
    basis : str
        PySCF basis set string. Default 'sto3g'.
    active_space : list of int, optional
        Orbital indices for S-operator contraction. Default [1, 2, 7, 8].
    n_workers : int, optional
        Number of parallel worker processes. Default: all available CPUs.
    Returns
    -------
    dict
        'geometry'      : d_eq
        'energy'        : COMPASS-VQE total energy (Hartree)
        'n_params'      : number of variational parameters
        'exact_energy'  : NumPy (FCI) total energy (Hartree)
        'hf_energy'     : Hartree-Fock energy (Hartree)
        'ansatz'        : final EvolvedOperatorAnsatz
        'optimal_params': optimised parameter vector
        'op_info'       : list of dicts describing each operator in the ansatz
    """
    if active_space is None:
        active_space = [1, 2, 7, 8]
    active_space_set = set(active_space)

    # spawn creates fully fresh worker processes — avoids deadlocks from BLAS/Aer
    # C++ threads that are already running in the parent when the pool is created.
    # (fork inherits the parent's thread lock state without the actual threads,
    # which causes silent hangs inside Aer's statevector simulator.)
    _ctx = mp.get_context('spawn')
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    # ------------------------------------------------------------------
    # 1. Molecular system setup
    # ------------------------------------------------------------------
    # 1. Molecule
    driver = PySCFDriver(atom=coord, charge=0, spin=0, basis=basis)
    es_problem_raw = ElectronicStructureProblem(driver)
    es_problem = FreezeCoreTransformer(freeze_core=True).transform(es_problem_raw)
    mapper = JordanWignerMapper()
    second_q_ops = es_problem.second_q_ops()
    fermionic_op = second_q_ops["ElectronicEnergy"] if isinstance(second_q_ops, dict) else second_q_ops[0]
    qubit_op = mapper.map(fermionic_op)
    num_particles = es_problem.num_particles
    num_spin_orbitals = es_problem.num_spin_orbitals
    try:
        shift = es_problem.nuclear_repulsion_energy or 0.0
    except AttributeError:
        shift = es_problem_raw.nuclear_repulsion_energy or 0.0

    init_state = HartreeFock(num_spin_orbitals, num_particles, qubit_mapper=mapper)
    _ = init_state.num_qubits  # force build for pickling

    # ------------------------------------------------------------------
    # 2. Hartree-Fock energy
    # ------------------------------------------------------------------
    theta = Parameter('a')
    qc    = QuantumCircuit(qubit_op.num_qubits)
    qc.rz(theta * 0, 0)
    qc.compose(init_state, front=True, inplace=True)
    hf_energy = np.real(cast(Any,VQE(qc, quantum_instance=backend).compute_minimum_eigenvalue(qubit_op).eigenvalue)) + shift

    # ------------------------------------------------------------------
    # 3. Exact (NumPy / FCI) energy
    # ------------------------------------------------------------------
    calc         = GroundStateEigensolver(converter, NumPyMinimumEigensolverFactory())
    exact_energy = np.real(calc.solve(es_problem).eigenenergies[0]) + shift

    # ------------------------------------------------------------------
    # 4. Build operator pools
    # ------------------------------------------------------------------
    ucc_d = UCC(num_particles=num_particles, num_spin_orbitals=num_spin_orbitals,
                excitations='d', initial_state=init_state, qubit_converter=converter)
    t2_excitations = ucc_d._get_excitation_list()
    t2_paulis      = _build_pauli_ops(ucc_d, converter)

    ucc_sh = UCC(num_particles=num_particles, num_spin_orbitals=num_spin_orbitals,
                 excitations=_Sh_ex, initial_state=init_state, qubit_converter=converter)
    sh_excitations = ucc_sh._get_excitation_list()
    sh_paulis      = _build_pauli_ops(ucc_sh, converter)

    ucc_sp = UCC(num_particles=num_particles, num_spin_orbitals=num_spin_orbitals,
                 excitations=_Sp_ex, initial_state=init_state, qubit_converter=converter)
    sp_excitations = ucc_sp._get_excitation_list()
    sp_paulis      = _build_pauli_ops(ucc_sp, converter)

    ucc_s = UCC(num_particles=num_particles, num_spin_orbitals=num_spin_orbitals,
                excitations='s', initial_state=init_state, qubit_converter=converter)
    s_excitations = ucc_s._get_excitation_list()
    s_paulis      = _build_pauli_ops(ucc_s, converter)

    # ------------------------------------------------------------------
    # 5. Pre-compile T2 circuits, then screen in parallel.
    #
    #    Transpilation is done ONCE in the main process for all T2 circuits
    #    together (qk_transpile parallelises internally).  Workers receive
    #    already-compiled circuits and only bind a single float parameter,
    #    avoiding repeated circuit build + transpile inside each process.
    # ------------------------------------------------------------------
    n_t2 = len(t2_paulis)
    print(f"[COMPASS] Pre-compiling {n_t2} T2 circuits ...")
    t2_ansatze    = [EvolvedOperatorAnsatz([op], initial_state=init_state) for op in t2_paulis]
    t2_transpiled = qk_transpile(t2_ansatze, backend=backend, optimization_level=1)
    # VQE reads circuit.parameters (sorted alphabetically) and maps initial_point
    # positionally, so no explicit parameter extraction is needed here.

    n_w_t2  = min(n_workers or mp.cpu_count(), n_t2)
    t2_args = [(i, t2_transpiled[i], qubit_op, shift) for i in range(n_t2)]

    print(f"[COMPASS] Screening {n_t2} T2 operators using {n_w_t2} workers ...")
    with ProcessPoolExecutor(max_workers=n_w_t2, mp_context=_ctx) as pool:
        t2_raw = list(pool.map(_t2_screen_worker, t2_args))

    # Filter T2s that lower the energy below HF, sort by energy (ascending)
    t2_screen = sorted(
        [(E, idx, theta) for idx, E, theta in t2_raw if abs(hf_energy - E) > 1e-5],
        key=lambda x: x[0]
    )
    print(f"[COMPASS] {len(t2_screen)} T2 operators passed screening.\n")

    # ------------------------------------------------------------------
    # 6. Determine S-operator candidates (pure Python — no VQE needed)
    #    For each sorted T2, collect unique S-hole/particle indices whose
    #    contractible orbital matches an active orbital in that T2.
    # ------------------------------------------------------------------
    # s_candidates: list of (cand_idx, rank, t2_idx, s_type, s_idx, theta_t2, E_t2)
    s_candidates = []

    for rank, (E_t2, t2_idx, theta_t2) in enumerate(t2_screen):
        t2_ex   = t2_excitations[t2_idx]
        t2_flat = [t2_ex[0][0], t2_ex[1][0], t2_ex[0][1], t2_ex[1][1]]
        active_in_t2 = active_space_set.intersection(t2_flat)
        if not active_in_t2:
            continue

        sh_seen = set()
        sp_seen = set()
        for act_orb in active_in_t2:
            for j, sh_ex in enumerate(sh_excitations):
                if j not in sh_seen and sh_ex[1][1] == act_orb:
                    s_candidates.append((len(s_candidates), rank, t2_idx, 'h', j, theta_t2, E_t2))
                    sh_seen.add(j)
            for j, sp_ex in enumerate(sp_excitations):
                if j not in sp_seen and sp_ex[0][1] == act_orb:
                    s_candidates.append((len(s_candidates), rank, t2_idx, 'p', j, theta_t2, E_t2))
                    sp_seen.add(j)

    # ------------------------------------------------------------------
    # 7. Pre-compile S circuits, then screen in parallel.
    #
    #    Same principle as step 5: transpile all (T2, S) circuits once in
    #    the main process.  Workers fix theta_t2 and use minimize_scalar
    #    to search only over theta_s (1-D), avoiding a 2-D VQE call.
    # ------------------------------------------------------------------
    if s_candidates:
        n_s = len(s_candidates)
        print(f"[COMPASS] Pre-compiling {n_s} S circuits ...")
        s_ansatze = [
            EvolvedOperatorAnsatz(
                [t2_paulis[t2_idx],
                 sh_paulis[s_idx] if s_type == 'h' else sp_paulis[s_idx]],
                initial_state=init_state
            )
            for _, rank, t2_idx, s_type, s_idx, theta_t2, E_t2 in s_candidates
        ]
        s_transpiled_list = qk_transpile(s_ansatze, backend=backend, optimization_level=1)

        # Each S circuit has 2 parameters: t[0] = T2 angle, t[1] = S angle.
        # EvolvedOperatorAnsatz([t2_op, s_op]) names its params t[0] (T2) and t[1] (S).
        # VQE maps initial_point positionally to circuit.parameters (sorted alphabetically),
        # so initial_point=[theta_t2, 0.01] correctly seeds t[0] and t[1] respectively.
        s_args = [
            (cand_idx, s_transpiled_list[ci], theta_t2, qubit_op, shift)
            for ci, (cand_idx, rank, t2_idx, s_type, s_idx, theta_t2, E_t2)
            in enumerate(s_candidates)
        ]

        n_w_s = min(n_workers or mp.cpu_count(), n_s)
        print(f"[COMPASS] Screening {n_s} S operators using {n_w_s} workers ...")
        with ProcessPoolExecutor(max_workers=n_w_s, mp_context=_ctx) as pool:
            s_raw = list(pool.map(_s_screen_worker, s_args))
        s_results = {ci: (E, opt_theta) for ci, E, opt_theta in s_raw}
    else:
        s_results = {}

    # Group accepted S operators by their T2 rank (preserves insertion order)
    s_by_rank = defaultdict(list)
    for cand_idx, rank, t2_idx, s_type, s_idx, theta_t2, E_t2 in s_candidates:
        E1, opt_s_theta = s_results[cand_idx]
        if abs(E_t2 - E1) > 1e-6:
            s_pauli = sh_paulis[s_idx] if s_type == 'h' else sp_paulis[s_idx]
            s_ex    = sh_excitations[s_idx] if s_type == 'h' else sp_excitations[s_idx]
            label   = 'S-hole' if s_type == 'h' else 'S-particle'
            s_by_rank[rank].append((s_pauli, s_ex, label, opt_s_theta))

    n_s_accepted = sum(len(v) for v in s_by_rank.values())
    print(f"[COMPASS] {n_s_accepted} S operator(s) accepted.\n")

    # ------------------------------------------------------------------
    # 8. Assemble final ansatz: sorted T2 -> its S operators -> standard singles
    # ------------------------------------------------------------------
    final_paulis  = []
    final_opt_t   = []
    final_op_info = []   # metadata for printing

    for rank, (E_t2, t2_idx, theta_t2) in enumerate(t2_screen):
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

    # ------------------------------------------------------------------
    # 9. Print final ansatz operator list
    # ------------------------------------------------------------------
    _print_ansatz(final_op_info)

    # ------------------------------------------------------------------
    # 10. Final VQE with complete COMPASS ansatz
    # ------------------------------------------------------------------
    ini_pt       = np.array(final_opt_t, dtype=np.float64)
    final_ansatz     = EvolvedOperatorAnsatz(final_paulis, initial_state=init_state)
    final_transpiled = qk_transpile(final_ansatz, backend=backend, optimization_level=1)
    qi_final = QuantumInstance(backend,
                               pass_manager=PassManager(),
                               bound_pass_manager=PassManager())
    final_vqe    = VQE(final_transpiled, optimizer=optimizer, quantum_instance=qi_final,
                       include_custom=True, initial_point=ini_pt)
    final_result = final_vqe.compute_minimum_eigenvalue(qubit_op)
    final_energy = np.real(final_result.eigenvalue) + shift

    print(f"geom- {d_eq}  Op_spin energy- {final_energy}  "
          f"parameters- {len(final_opt_t)}  fci energy- {exact_energy}")

    return {
        'geometry':       d_eq,
        'energy':         final_energy,
        'n_params':       len(final_opt_t),
        'exact_energy':   exact_energy,
        'hf_energy':      hf_energy,
        'ansatz':         final_ansatz,
        'optimal_params': final_result.optimal_point,
        'op_info':        final_op_info,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    raw   = input()
    parts = raw.split(':')
    compass_vqe(coord=parts[1], d_eq=float(parts[0]))
