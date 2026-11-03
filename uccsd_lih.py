from typing import cast
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import StatevectorEstimator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.operators import FermionicOp

# -------------------- LiH input: same as lih.py --------------------

bond_length = 1.595
basis_set = "sto-3g"
num_states = 4
active_space = [1, 2, 3, 4]  # Four active spatial orbitals

coordinates = f"""
Li  0.000000  0.000000  0.000000
H   0.000000  0.000000  {bond_length:.6f}
"""

print("\nGeometry:")
print(coordinates)
print(f"Basis: {basis_set}")
print(f"Active orbitals: {active_space}")


# -------------------- Build electronic structure problem --------------------

driver = PySCFDriver(
    atom=coordinates,
    basis=basis_set,
    charge=0,
    spin=0,
)

problem = driver.run()

# Freeze the core and retain spatial orbitals [1, 2, 3, 4].
transformer = ActiveSpaceTransformer(
    num_electrons=2,
    num_spatial_orbitals=len(active_space),
    active_orbitals=active_space,
)
problem = cast(ElectronicStructureProblem, transformer.transform(problem))

mapper = JordanWignerMapper()
hamiltonian = cast(
    SparsePauliOp,
    mapper.map(cast(FermionicOp, problem.hamiltonian.second_q_op())),
)

num_particles = problem.num_particles
num_spatial_orbitals = problem.num_spatial_orbitals

print("\nActive-space information:")
print(f"Particles: {num_particles}")
print(f"Spatial orbitals: {num_spatial_orbitals}")
print(f"Qubits: {hamiltonian.num_qubits}")


# -------------------- Obtain first four active-space states --------------------
# CASCI/FCI is used here to identify the four states and their energies.

mol = gto.M(
    atom=coordinates,
    basis=basis_set,
    charge=0,
    spin=0,
    unit="Angstrom",
    verbose=0,
)

mf = scf.RHF(mol).run()

# LiH STO-3G has 6 electrons. Freeze the Li 1s core and use 4 active orbitals.
mc = mcscf.CASCI(mf, 4, 2)
setattr(getattr(mc, "fcisolver"), "nroots", num_states)
# PySCF's type stubs do not expose ``fcisolver`` on the CASCI variants,
# although it is provided at runtime by the CASCI object.
fcisolver = getattr(mc, "fcisolver")

# The PySCF type stubs omit CASCI attributes that are available at runtime.
ncas = int(getattr(mc, "ncas"))
nelecas = getattr(mc, "nelecas")
# ``get_h1eff`` is available on CASCI at runtime, but is missing from
# PySCF's type stubs (which may infer a DF-CASCI variant here).
get_h1eff = getattr(mc, "get_h1eff")
# ``get_eri`` is available on CASCI at runtime, but is missing from
# PySCF's type stubs (which may infer a DF-CASCI variant here).
get_eri = getattr(mc, "get_eri")
fcisolver.kernel(get_h1eff(), get_eri(), ncas, nelecas)

print("\nCASCI/FCI reference energies:")
for state, energy in enumerate(fcisolver.e_tot):
    print(f"State {state}: {energy:.12f} Ha")


# -------------------- UCCSD optimization for each state --------------------

optimizer = SLSQP(maxiter=500, ftol=1e-9)
estimator = StatevectorEstimator()

uccsd_energies = []

for state in range(num_states):
    print(f"\n{'=' * 60}")
    print(f"UCCSD optimization for state {state}")
    print(f"{'=' * 60}")

    # Build the state-specific reference determinant.
    # State 0 uses the HF determinant; excited states use excited
    # determinants with the same electron count.
    reference = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    # UCCSD ansatz; excitation pool is generated for this active space.
    ansatz = UCC(
        num_particles=num_particles,
        num_spatial_orbitals=num_spatial_orbitals,
        excitations="sd",
        qubit_mapper=mapper,
        initial_state=reference,
    )

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=np.zeros(ansatz.num_parameters),
    )

    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    eigenvalue = result.eigenvalue
    if eigenvalue is None:
        raise RuntimeError("VQE did not return an eigenvalue")
    energy = float(np.real(eigenvalue))

    uccsd_energies.append(energy)
    print(f"State {state} UCCSD energy: {energy:.12f} Ha")
    print(f"Function evaluations: {result.cost_function_evals}")


# -------------------- Summary --------------------

print("\n\nUCCSD results:")
for state, energy in enumerate(uccsd_energies):
    print(f"State {state}: {energy:.12f} Ha")

print("\nExcitation energies relative to UCCSD state 0:")
for state, energy in enumerate(uccsd_energies):
    gap_hartree = energy - uccsd_energies[0]
    gap_ev = gap_hartree * 27.211386
    print(f"State {state}: {gap_hartree:.8f} Ha  ({gap_ev:.6f} eV)")

print("\nRun finished.")