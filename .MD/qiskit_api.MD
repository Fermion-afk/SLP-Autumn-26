# PySCF(Python based Simulation of chemical framework)

## Building a Molecule
```python
import pyscf
gto = pyscf.gto()
mol = gto.Mole() # Defining a class mol
mol.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
mol.basis = 'sto-3g'
mol.build()
```
there are several shortcuts to build the molecule
```python
mol = gto.Mole()
mol.build(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g',
)
```
above is a alternative simpler way to build a molecule  
and below is a even simpler way 
```python
mol = gto.M(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g',
)
```
### Other attributes
```python
import pyscf
gto = pyscf.gto
mol = gto.Mole()
mol.charge = 1
mol.spin = -0.5
mol.symmetry = "d2h"
mol.max_memory = 1000 # in MB
```

## Mean Field Theory
### Hartree - Fock

```python
import pyscf 
scf = pyscf.scf
gto = pyscf.gto
mol = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz')
rhf_mol = scf.RHF(mol) #Hartree fock Object
e_mol = rhf_mol.kernel() #Ground state Hartree fock energy
```
Open shell system can be treated with the following methods
```python
mol_o2 = gto.M(atom='O 0 0 0; O 0 0 1.2', spin=2) # (n+2 alpha, n beta) electrons
uhf_o2 = scf.UHF(mol_o2) # UHF object
e_uhf = uhf_o2.kernel()
rohf_o2 = scf.ROHF(mol_o2) # ROHF object
e_rohf = rohf_o2.kernel()
```
## Correlated Wavefunction methods
### MP-2 
```python
from pyscf import mp
mp2_c2 = mp.MP2(rhf_c2)
o_c2 = mp2_c2.kernel()
# o_c2 is object with the energy stored in [0] index and the [1] index stores the excitation amplitude in a 4D array
e_c2 = o_c2[0]
```
### CC
```python
from pyscf import cc
ccsd_h2o = cc.CCSD(rhf_h2o)
ccsd_h2o.direct = True # AO-direct algorithm to reduce I/O overhead
ccsd_h2o.frozen = 1 # frozen core
e_ccsd = ccsd_h2o.kernel()[1]
e_ccsd_t = e_ccsd + ccsd_h2o.ccsd_t()
```
### FCI
```python
import pyscf
fci = pyscf.fci
h2o_fci = fci.FCI(rhf_h2o)
e_h2o_fci = h2o_fci.kernel()[0]
```

# Qiskit transpiler
## Circuit converter
``` python
from qiskit.converters import circuit_to_dag, dag_to_circuit

#consider a already defined quantum circuit object
# lets call it qc

dag = circuit_to_dag(qc)

#some methods of the created class
print(type(dag))
print(dag.num_qubits())
print(dag.size())

print(dag.draw())

qc_o = dag_to_circuit(dag)

print(qc_o.draw())
```
## Pass Manager
This create a custom pass pipeline whereas the transpile takes care of the pass management itself, This method can used when extra customization is needed.
```python
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates,
    CommutativeCancellation,
)

qc = QuantumCircuit(2)

qc.h(0)
qc.h(0)
qc.x(1)
qc.x(1)
qc.cx(0, 1)

print("Before:")
print(qc.draw())

pm = PassManager([
    Optimize1qGates(),
    CommutativeCancellation(),
])
optimized = pm.run(qc)
print("After:")
print(optimized.draw())
``` 
## Circuit synthesis
Circuit synthesis converts a higher-level operation into a sequence of elementary quantum gates.

```python
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.synthesis import synthesize_clifford_full

# A two-qubit controlled-NOT matrix
U = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
], dtype=complex)

qc = QuantumCircuit(2)
qc.append(UnitaryGate(U), [0, 1])

print(qc.draw())
```