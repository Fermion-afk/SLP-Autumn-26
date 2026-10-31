from ss_compass_vqe_timestamped import ss_compass_vqe

# -------------------- Molecular system --------------------

bond_length = 1.326
basis_set = "sto-3g"

coordinates = "\n".join([
    "Be  0.000000  0.000000  0.000000",
    f"H   0.000000  0.000000  {bond_length:.6f}",
    f"H   0.000000  0.000000  {-bond_length:.6f}",
])

# Check the exact geometry string passed to PySCF
print("Geometry passed to PySCFDriver:")
print(repr(coordinates))
print(coordinates)

# -------------------- SS-Compass-VQE parameters --------------------

num_states = 4
weights = [10.0, 1.5, 1.0, 0.5]

active_space = [1, 2, 7, 8]
method = "HartreeFock"

# -------------------- Run SS-Compass-VQE --------------------

result = ss_compass_vqe(
    coordinates=coordinates,
    basis_set=basis_set,
    weights=weights,
    active_space=active_space,
    method=method,
    d_eq=bond_length,
    no_init_states=num_states,
)

print("\nSS-Compass-VQE result:")
print(result)