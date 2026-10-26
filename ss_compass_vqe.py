from ss_vqe import SS_VQE,SS_VQEResult
import warnings
warnings.filterwarnings("ignore")
from typing import Any, cast

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from math import comb

from pyscf import ci, mcscf, cc
from pyscf.fci import cistring, direct_spin1
from qiskit.circuit.library import StatePreparation

import numpy as np
from qiskit import transpile as qk_transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver

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
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# ------------------
# Helper functions
# ------------------

def _t2_screen_worker(args):
    idx,operator,transpiled_circuit,shift,weights = args
    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    ss_vqe = SS_VQE(estimator,transpiled_circuit,weights,optimizer,
                    initial_point = np.array([0.0]))
    result = ss_vqe.compute_minimum_eigenvalue(operator)
    eigenvalue = result.eigenvalue
    optimal_point = result.optimal_point
    return idx, float(np.real(cast(Any, eigenvalue))) + shift,float(np.asarray(optimal_point).reshape(-1)[0])

def _s_screen_worker(args):
    idx,operator,transpiled_circuit,shift,weights,theta_t2 = args
    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
    ss_vqe = SS_VQE(estimator,transpiled_circuit,weights,optimizer,
                    initial_point = np.array([theta_t2,0.01]))
    result = ss_vqe.compute_minimum_eigenvalue(operator)
    eigenvalue = result.eigenvalue
    optimal_point = result.optimal_point
    return idx,float(np.real(cast(Any,eigenvalue))) + shift,float(np.asarray(optimal_point).reshape(-1)[1])

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
# Main Callable
# ----------------
def ss_compass_vqe(coordinates: str,d_eq: float,basis_set: str,no_init_states: int,weights:list,
                   method: str = "HartreeFock",active_space: list|None = None,N_workers: int|None= None) -> dict|None:
    """ Returns
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

    # Constant energy contributions excluded from the qubit Hamiltonian
    shift = float(np.real(sum(es_problem.hamiltonian.constants.values())))

    num_spatial_orbitals = int(cast(Any, es_problem.num_spatial_orbitals))  

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

    init_states = _init_states(method, no_init_states,num_spatial_orbitals,num_particles, mapper)

    # Force circuit construction before sending it to worker processes
    _ = init_states[0].num_qubits
    # ------------------------------------------------------------------
    # 2. Hartree-Fock energy
    # ------------------------------------------------------------------
    num_spatial_orbitals = es_problem.num_spatial_orbitals
    init_state = HartreeFock(num_spatial_orbitals,num_particles,mapper)
    # ------------------------------------------------------------------
    # 3. Exact (NumPy / FCI) energy
    # ------------------------------------------------------------------
    numpy_solver = NumPyMinimumEigensolver(filter_criterion=cast(Any,es_problem.get_default_filter_criterion()))
    calc = GroundStateEigensolver(mapper, numpy_solver)
    exact_result = calc.solve(es_problem)
    result = exact_result.eigenvalues
    if result is not None:
        exact_energy = float(np.real(result[0]))
    else : 
        raise RuntimeError("The FCI calculation didn't return any output")
    # ------------------------------------------------------------------
    # 4. Build operator pools
    # ------------------------------------------------------------------
    ucc_s = UCC(num_particles=num_particles,
        num_spatial_orbitals=es_problem.num_spatial_orbitals,
        excitations="s",initial_state=init_state,qubit_mapper=mapper)
    s_excitations = ucc_s.excitation_list
    s_paulis = _build_pauli_ops(ucc_s, mapper)

    ucc_d = UCC(num_particles=num_particles,
        num_spatial_orbitals=es_problem.num_spatial_orbitals,
        excitations="d",initial_state=init_state,qubit_mapper=mapper)
    t2_excitations = ucc_d.excitation_list
    t2_paulis = _build_pauli_ops(ucc_d, mapper)

    ucc_sh = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
            excitations=_Sh_ex, initial_state=init_state, qubit_mapper=mapper)
    sh_excitations = ucc_sh._get_excitation_list()
    sh_paulis = _build_pauli_ops(ucc_sh,mapper)

    ucc_sp = UCC(num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals,
            excitations=_Sp_ex, initial_state=init_state, qubit_mapper=mapper)
    sp_excitations = ucc_sp._get_excitation_list()
    sp_paulis = _build_pauli_ops(ucc_sp, mapper)

