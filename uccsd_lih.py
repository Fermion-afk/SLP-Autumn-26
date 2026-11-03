# uccsd_ss_vqe.py
# Run in the same directory as ss_vqe.py

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from time import perf_counter
from typing import cast,Any

import numpy as np
from pyscf.fci import cistring

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, EvolvedOperatorAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

from ss_vqe import SS_VQE


# -------------------- User input --------------------

bond_length = 1.595
basis_set = "sto-3g"
coordinates = f"Li 0.0 0.0 0.0\nH 0.0 0.0 {bond_length:.6f}"

num_states = 4
weights = np.array([10.0, 2.0, 2.0, 0.5])

maxiter = 10_000
maxfun = 200_000


# -------------------- Logging --------------------

run_start = perf_counter()


def log(message):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    elapsed = perf_counter() - run_start
    print(f"[{stamp} | elapsed {elapsed:,.1f} s] {message}", flush=True)


def callback(eval_count, parameters, cost, metadata):
    log(f"Evaluation {eval_count:>5} | weighted cost = {cost:.12f} Ha")


# -------------------- Reference states --------------------

def make_reference_states(mf, num_spatial_orbitals, num_particles, mapper, count):
    """Create determinant references ranked by orbital-energy sum."""
    n = num_spatial_orbitals
    n_alpha, n_beta = num_particles

    alpha_strings = cistring.make_strings(range(n), n_alpha)
    beta_strings = cistring.make_strings(range(n), n_beta)

    determinants = []
    orbital_energies = np.asarray(mf.mo_energy)

    for alpha in alpha_strings:
        for beta in beta_strings:
            occ_alpha = [i for i in range(n) if (alpha >> i) & 1]
            occ_beta = [i for i in range(n) if (beta >> i) & 1]

            energy = sum(orbital_energies[occ_alpha])
            energy += sum(orbital_energies[occ_beta])

            # JW ordering: alpha spin orbitals followed by beta spin orbitals.
            occupied = occ_alpha + [n + i for i in occ_beta]
            index = sum(1 << i for i in occupied)
            determinants.append((energy, index))

    determinants.sort(key=lambda item: item[0])

    if count > len(determinants):
        raise ValueError(
            f"Requested {count} references, but only {len(determinants)} exist."
        )

    n_qubits = 2 * n
    circuits = []

    for _, index in determinants[:count]:
        state = np.zeros(2**n_qubits, dtype=complex)
        state[index] = 1.0

        circuit = QuantumCircuit(n_qubits)
        circuit.append(StatePreparation(Statevector(state)), range(n_qubits))
        circuits.append(circuit)

    return circuits


# -------------------- Main calculation --------------------

def uccsd_ss_vqe():
    log(
        f"Run started | LiH | bond length={bond_length:.6f} Å "
        f"| basis={basis_set}"
    )

    driver = PySCFDriver(
        atom=coordinates,
        charge=0,
        spin=0,
        basis=basis_set,
    )
    es_problem = driver.run()

    es_problem = cast(
        ElectronicStructureProblem,
        FreezeCoreTransformer(freeze_core=True).transform(es_problem),
    )
    mapper = JordanWignerMapper()

    second_q_operator = cast(FermionicOp, es_problem.hamiltonian.second_q_op())
    hamiltonian = cast(SparsePauliOp, mapper.map(second_q_operator))
    # ``constants`` is not declared on the base Hamiltonian type.
    constants = getattr(es_problem.hamiltonian, "constants", {})
    shift = float(np.real(sum(constants.values())))

    num_particles = es_problem.num_particles
    num_spatial_orbitals = es_problem.num_spatial_orbitals
    num_qubits = hamiltonian.num_qubits
    mf = driver._calc

    log(
        f"Hamiltonian ready | orbitals={num_spatial_orbitals} "
        f"| particles={num_particles} | qubits={num_qubits} "
        f"| shift={shift:.12f} Ha"
    )

    # Prepare distinct determinant reference states.
    references = make_reference_states(
        mf,
        num_spatial_orbitals,
        num_particles,
        mapper,
        num_states,
    )
    log(f"Prepared {len(references)} determinant reference states")

    # Build the shared UCCSD excitation pool.
    hf_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)

    uccsd = UCC(
        num_particles=num_particles,
        num_spatial_orbitals=num_spatial_orbitals,
        excitations="sd",
        initial_state=hf_state,
        qubit_mapper=mapper,
    )

    uccsd_ops = [mapper.map(cast(Any,op)) for op in uccsd.excitation_ops()]
    num_parameters = len(uccsd_ops)

    excitation_list = uccsd.excitation_list
    if excitation_list is None:
        raise RuntimeError("UCCSD excitation list was not generated")
    num_singles = sum(len(exc[0]) == 1 for exc in excitation_list)
    num_doubles = sum(len(exc[0]) == 2 for exc in excitation_list)

    log(
        f"UCCSD pool ready | singles={num_singles} "
        f"| doubles={num_doubles} | parameters={num_parameters}"
    )

    # Same UCCSD pool applied to every reference state.
    ansatze: list[QuantumCircuit] = [
        EvolvedOperatorAnsatz(uccsd_ops, initial_state=reference)
        for reference in references
    ]
    ansatze = transpile(ansatze, optimization_level=1)

    parameter_counts = [ansatz.num_parameters for ansatz in ansatze]
    if len(set(parameter_counts)) != 1:
        raise ValueError(f"Ansatz parameter counts differ: {parameter_counts}")

    initial_point = np.zeros(num_parameters)
    estimator = StatevectorEstimator()
    optimizer = L_BFGS_B(maxfun=maxfun, maxiter=maxiter)

    log("Starting SS-VQE optimization")

    ss_vqe = SS_VQE(
        estimator=estimator,
        ansatze=ansatze,
        weights=weights,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=callback,
    )

    result = ss_vqe.compute_minimum_eigenvalue(hamiltonian)

    # Individual optimized state energies, including the constant shift.
    final_energies = []

    for circuit in result.optimal_circuits:
        bindings = dict(zip(circuit.parameters, np.asarray(result.optimal_point)))
        pub = (circuit, hamiltonian, bindings)
        energy = getattr(estimator.run([pub]).result()[0].data, "evs")
        final_energies.append(float(np.real(energy)) + shift)

    final_energies = np.asarray(final_energies)
    gaps = final_energies - final_energies[0]
    weighted_total = result.optimal_value + shift * np.sum(weights)

    log("Optimization complete")
    log(
        f"Evaluations={result.cost_function_evals} "
        f"| optimizer time={result.optimizer_time:.2f} s"
    )
    log(f"Weighted objective (with shift) = {weighted_total:.12f} Ha")

    print("\n" + "=" * 65, flush=True)
    print("                 UCCSD-SS-VQE RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"{'State':>8} {'Energy (Ha)':>22} {'Gap (Ha)':>22}", flush=True)

    for i, (energy, gap) in enumerate(zip(final_energies, gaps)):
        print(f"{i:>8} {energy:>22.12f} {gap:>22.12f}", flush=True)

    print("=" * 65, flush=True)
    print(f"Optimal parameters: {np.asarray(result.optimal_point)}", flush=True)
    log("Run finished")

    return {
        "result": result,
        "energies": final_energies,
        "gaps": gaps,
        "optimal_point": np.asarray(result.optimal_point),
        "ansatze": result.optimal_circuits,
    }


if __name__ == "__main__":
    output = uccsd_ss_vqe()