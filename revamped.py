from __future__ import annotations

import multiprocessing as mp
import os
import signal
import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from itertools import combinations, product
from math import comb
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from pyscf import ao2mo
from qiskit import transpile as qk_transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

from ci_utils import (active_hf_determinant,
    build_spin_summed_1rdm,
    compute_s2_expectation,
    determinants_to_circuits,
    lowest_eigenvectors,
    projected_hamiltonian,
    rotate_ci_vectors_to_no,
    state_energies)
from operators import build_operator_pools, build_s_candidates, screen_s_pool, screen_t2_pool
from ss_vqe import SS_VQE

warnings.filterwarnings("ignore")

_STAGE_ORDER = ["HF_SOLVER", "initialize_states", "compute_1rdm", "compute_exact_spectrum",
                "pool_creator", "screen_t2", "screener", "ansatz_assembler", "final_vqe_runner"]

class ss_compass_vqe:
    def __init__(
        self,
        coordinates: str,
        spin: int,
        charge: int,
        basis_set: str,
        weights: Sequence[float],
        no_init_states: int,
        threshold: Sequence[float],
        screening_weights: Sequence[float] | None = None,
        nfc: bool = True,
        log_file: str | Path | None = None,
        occ_threshold: float = 1.90,
        virt_threshold: float = 0.1,
    ) -> None:
        self.coordinates = coordinates
        self.spin = int(spin)
        self.charge = int(charge)
        self.basis_set = basis_set
        self.no_init_states = int(no_init_states)
        self.nfc = bool(nfc)
        self.occ_threshold = float(occ_threshold)
        self.virt_threshold = float(virt_threshold)

        if screening_weights is None:
            sum_w = float(sum(weights))
            screening_weights = [float(w) / sum_w for w in weights[::-1]]

        self.weights = np.asarray(weights, dtype=float)
        self.screening_weights = np.asarray(screening_weights, dtype=float)
        self.threshold = np.asarray(threshold, dtype=float)

        if self.no_init_states < 1:
            raise ValueError("no_init_states must be >= 1.")
        if len(self.weights) != self.no_init_states or len(self.screening_weights) != self.no_init_states:
            raise ValueError("weights and screening_weights must have length no_init_states.")
        if self.threshold.shape != (2,) or np.any(self.threshold < 0):
            raise ValueError("threshold must be a 2-element non-negative sequence [T2_thresh, S_thresh].")

        sched_getaffinity = getattr(os, "sched_getaffinity", None)
        self.n_workers = len(cast(Any, sched_getaffinity(0))) if callable(sched_getaffinity) else (os.cpu_count() or 1)
        self._mp_ctx = mp.get_context("spawn")
        self.estimator = StatevectorEstimator()
        self.optimizer = L_BFGS_B(maxfun=200_000, maxiter=10_000)
        self.mapper = JordanWignerMapper()

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(log_file) if log_file is not None else Path(f"ss_compass_vqe_{stamp}.log")
        self.chk_path = self.log_path.with_suffix(".chk.npz")
        self._log_fh = None
        self._current_stage = "INIT"
        self._completed_stage: str | None = None
        self._run_start: float | None = None

        # Stage 1: HF_SOLVER state
        self.es_problem: ElectronicStructureProblem | None = None
        self.mf: Any = None
        self.hf_energy: float | None = None
        self.nuclear_repulsion: float | None = None
        self.shift: float | None = None
        self.num_particles: tuple[int, int] | None = None
        self.n_electrons: int | None = None
        self.num_spatial_orbitals: int | None = None
        self.num_spin_orbitals: int | None = None
        self.ncore: int = 0
        self.mo_coeff: np.ndarray | None = None
        self.mo_energy: np.ndarray | None = None
        self.mo_occ: np.ndarray | None = None
        self.h1e: np.ndarray | None = None
        self.h2e: np.ndarray | None = None
        self.fermionic_op: Any = None
        self.qubit_op: Any = None

        # Stage 2: initialize_states (Fixed-N spin-complete CIS)
        self.sci_determinants: list[int] | None = None
        self.sci_eigenvectors: np.ndarray | None = None
        self.sci_energies: np.ndarray | None = None
        self.sci_s2: np.ndarray | None = None

        # Stage 3: compute_1rdm (NO basis + 4-fold classification + Givens circuits)
        self.rdm1_spatial: np.ndarray | None = None
        self.noons: np.ndarray | None = None
        self.no_transform: np.ndarray | None = None
        self.no_coeff: np.ndarray | None = None
        self.inactive_occ: list[int] = []
        self.active_occ: list[int] = []
        self.active_virt: list[int] = []
        self.inactive_virt: list[int] = []
        self.active_orbitals: list[int] = []
        self.h1e_no: np.ndarray | None = None
        self.h2e_no: np.ndarray | None = None
        self.no_determinants: list[int] | None = None
        self.sci_eigenvectors_no: np.ndarray | None = None
        self.init_states: list[QuantumCircuit] | None = None
        self.initial_energies: np.ndarray | None = None

        # Stage 3b: Exact fixed-N reference spectrum
        self.exact_energies: np.ndarray | None = None

        # Stage 4: pool_creator
        self.s_excitations: list[Any] = []
        self.t2_excitations: list[Any] = []
        self.sh_excitations: list[Any] = []
        self.sp_excitations: list[Any] = []
        self.s_paulis: list[Any] = []
        self.t2_paulis: list[Any] = []
        self.sh_paulis: list[Any] = []
        self.sp_paulis: list[Any] = []

        # Stages 5-7: screener, ansatz_assembler, final_vqe_runner (pending final design)
        self.t2_screen: list[tuple[float, int, float, np.ndarray]] = []
        self.t2_screening_data: list[dict[str, Any]] = []
        self.s_candidates: list[tuple[Any, ...]] = []
        self.s_by_rank: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
        self.s_screening_data: list[dict[str, Any]] = []
        self._s_partial_results: dict[int, Any] = {}
        self.final_paulis: list[Any] = []
        self.final_op_info: list[dict[str, Any]] = []
        self.initial_point: np.ndarray | None = None
        self.final_ansatze: list[QuantumCircuit] | None = None
        self.final_result: Any = None
        self.state_energies: np.ndarray | None = None

        # VQE recovery state
        self.vqe_eval_count: int = 0
        self.latest_params: np.ndarray | None = None
        self.latest_state_energies: np.ndarray | None = None
        self.best_cost: float = float("inf")
        self.best_params: np.ndarray | None = None
        self.best_state_energies: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Gaussian-Style Logging & Atomic Checkpointing
    # ------------------------------------------------------------------
    def _ensure_log(self, append: bool = False) -> None:
        if self._log_fh is not None and not self._log_fh.closed:
            return
        mode = "a" if append and self.log_path.exists() else "w"
        self._log_fh = open(self.log_path, mode, encoding="utf-8", buffering=1)
        if mode == "w":
            now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            scf_tag = "RHF" if self.spin == 0 else "ROHF"
            self._log("*" * 74)
            self._log(" SS-COMPASS-VQE  |  Subspace-Search Commutativity-Screened VQE")
            self._log(f" Run started: {now}  |  Workers: {self.n_workers}")
            self._log("*" * 74)
            self._log("-" * 74)
            self._log(f" # {scf_tag}/{self.basis_set}  NFC={self.nfc}  NIS={self.no_init_states}  "
                      f"Thresh=({self.threshold[0]:.2e}, {self.threshold[1]:.2e})")
            self._log(f"   NOON_Cutoffs = ({self.occ_threshold:.2f}, {self.virt_threshold:.2f})")
            self._log(f"   Weights      = {np.array2string(self.weights, precision=6)}")
            self._log(f"   Scr_Weights  = {np.array2string(self.screening_weights, precision=6)}")
            self._log("-" * 74)
            self._log(f" Charge = {self.charge:2d}   Multiplicity = {self.spin + 1} (2S = {self.spin})")
            self._log(f" Checkpoint file: {self.chk_path}")
            self._log("-" * 74)

    def _log(self, line: str = "", also_stdout: bool = True) -> None:
        self._ensure_log(append=True)
        assert self._log_fh is not None
        self._log_fh.write(line + "\n")
        self._log_fh.flush()
        if also_stdout:
            print(line, flush=True)

    def _log_stage_header(self, stage_name: str) -> None:
        self._current_stage = stage_name
        idx = _STAGE_ORDER.index(stage_name) + 1 if stage_name in _STAGE_ORDER else 0
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        self._log(f"\n{'='*74}\n [Stage {idx}/{len(_STAGE_ORDER)}] Entering {stage_name}  ({now})\n{'='*74}")

    def _require(self, *attrs: str) -> None:
        missing = [a for a in attrs if getattr(self, a, None) is None]
        if missing:
            raise RuntimeError(f"Cannot run {self._current_stage}: missing prerequisite attribute(s) {missing}.")

    def save_checkpoint(self, completed_stage: str | None = None) -> None:
        if completed_stage is not None:
            self._completed_stage = completed_stage

        payload: dict[str, Any] = {
            "coordinates": np.array(self.coordinates), "spin": np.array(self.spin),
            "charge": np.array(self.charge), "basis_set": np.array(self.basis_set),
            "weights": self.weights, "screening_weights": self.screening_weights,
            "no_init_states": np.array(self.no_init_states), "threshold": self.threshold,
            "nfc": np.array(self.nfc), "occ_threshold": np.array(self.occ_threshold),
            "virt_threshold": np.array(self.virt_threshold),
            "completed_stage": np.array(self._completed_stage or ""),
            "current_stage": np.array(self._current_stage),
            "vqe_eval_count": np.array(self.vqe_eval_count), "best_cost": np.array(self.best_cost),
        }

        optional = {
            "hf_energy": self.hf_energy, "nuclear_repulsion": self.nuclear_repulsion, "shift": self.shift,
            "num_particles": self.num_particles, "n_electrons": self.n_electrons,
            "num_spatial_orbitals": self.num_spatial_orbitals, "num_spin_orbitals": self.num_spin_orbitals,
            "ncore": self.ncore, "mo_coeff": self.mo_coeff, "mo_energy": self.mo_energy, "mo_occ": self.mo_occ,
            "h1e": self.h1e, "h2e": self.h2e, "sci_determinants": self.sci_determinants,
            "sci_eigenvectors": self.sci_eigenvectors, "sci_energies": self.sci_energies, "sci_s2": self.sci_s2,
            "rdm1_spatial": self.rdm1_spatial, "noons": self.noons, "no_transform": self.no_transform,
            "no_coeff": self.no_coeff, "inactive_occ": self.inactive_occ, "active_occ": self.active_occ,
            "active_virt": self.active_virt, "inactive_virt": self.inactive_virt,
            "active_orbitals": self.active_orbitals, "h1e_no": self.h1e_no, "h2e_no": self.h2e_no,
            "no_determinants": self.no_determinants, "sci_eigenvectors_no": self.sci_eigenvectors_no,
            "initial_energies": self.initial_energies, "exact_energies": self.exact_energies,
            "t2_screen": np.array(self.t2_screen, dtype=object) if self.t2_screen else None,
            "s_by_rank": np.array(dict(self.s_by_rank), dtype=object) if self.s_by_rank else None,
            "_s_partial_results": np.array(self._s_partial_results, dtype=object) if self._s_partial_results else None,
            "final_op_info": np.array(self.final_op_info, dtype=object) if self.final_op_info else None,
            "initial_point": self.initial_point, "state_energies": self.state_energies,
            "latest_params": self.latest_params, "latest_state_energies": self.latest_state_energies,
            "best_params": self.best_params, "best_state_energies": self.best_state_energies,
        }
        for key, val in optional.items():
            if val is not None:
                payload[key] = val if isinstance(val, np.ndarray) else np.asarray(val)

        tmp_path = self.chk_path.with_name(self.chk_path.name + ".tmp.npz")
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, self.chk_path)

    @classmethod
    def resume(cls, checkpoint_path: str | Path, log_file: str | Path | None = None) -> ss_compass_vqe:
        """Restore solver state from a .chk.npz checkpoint file."""
        chk = np.load(checkpoint_path, allow_pickle=True)
        resolved_log = Path(log_file) if log_file is not None else Path(checkpoint_path).with_suffix("").with_suffix(".log")
        obj = cls(
            coordinates=str(chk["coordinates"].item()), spin=int(chk["spin"].item()),
            charge=int(chk["charge"].item()), basis_set=str(chk["basis_set"].item()),
            weights=chk["weights"].tolist(), no_init_states=int(chk["no_init_states"].item()),
            threshold=chk["threshold"].tolist(), screening_weights=chk["screening_weights"].tolist(),
            nfc=bool(chk["nfc"].item()), log_file=resolved_log,
            occ_threshold=float(chk["occ_threshold"].item()) if "occ_threshold" in chk else 1.8,
            virt_threshold=float(chk["virt_threshold"].item()) if "virt_threshold" in chk else 0.2,
        )

        scalar_keys = {"hf_energy", "nuclear_repulsion", "shift", "best_cost"}
        int_keys = {"n_electrons", "num_spatial_orbitals", "num_spin_orbitals", "ncore", "vqe_eval_count"}
        list_keys = {"sci_determinants", "no_determinants", "inactive_occ", "active_occ", "active_virt", "inactive_virt", "active_orbitals"}
        skip_keys = {"coordinates", "spin", "charge", "basis_set", "weights", "screening_weights",
                     "no_init_states", "threshold", "nfc", "occ_threshold", "virt_threshold", "completed_stage", "current_stage"}

        for key in chk.files:
            if key in skip_keys:
                continue
            if key in scalar_keys:
                setattr(obj, key, float(chk[key].item()))
            elif key in int_keys:
                setattr(obj, key, int(chk[key].item()))
            elif key in list_keys:
                setattr(obj, key, chk[key].tolist())
            elif key == "num_particles":
                obj.num_particles = tuple(int(x) for x in chk[key].tolist())  # type: ignore[assignment]
            elif key == "t2_screen":
                obj.t2_screen = list(chk[key].tolist())
            elif key == "s_by_rank":
                obj.s_by_rank = defaultdict(list, chk[key].item())
            elif key == "_s_partial_results":
                obj._s_partial_results = chk[key].item()
            elif key == "final_op_info":
                obj.final_op_info = list(chk[key].tolist())
            elif hasattr(obj, key):
                setattr(obj, key, chk[key])

        obj._completed_stage = str(chk["completed_stage"].item()) or None

        h1, h2 = (obj.h1e_no, obj.h2e_no) if (obj.h1e_no is not None and obj.h2e_no is not None) else (obj.h1e, obj.h2e)
        if h1 is not None and h2 is not None and obj.shift is not None:
            elec = ElectronicEnergy.from_raw_integrals(h1, h2)
            elec.constants = {"shift": obj.shift}
            obj.fermionic_op = elec.second_q_op()
            obj.qubit_op = obj.mapper.map(obj.fermionic_op)

        if obj.no_determinants is not None and obj.sci_eigenvectors_no is not None and obj.num_spin_orbitals is not None:
            obj.init_states = determinants_to_circuits(obj.no_determinants, obj.sci_eigenvectors_no, obj.num_spin_orbitals)

        if (obj._completed_stage in {"pool_creator", "screen_t2", "screener", "ansatz_assembler", "final_vqe_runner"}
                and obj.num_spatial_orbitals is not None and obj.num_particles is not None):
            pools = build_operator_pools(obj.num_spatial_orbitals, obj.num_particles, obj.mapper,
                                         obj.inactive_occ, obj.active_occ, obj.active_virt, obj.inactive_virt)
            for k, v in pools.items():
                setattr(obj, k, v)

        if (obj._completed_stage in {"ansatz_assembler", "final_vqe_runner"}
                and obj.final_op_info and obj.init_states is not None):
            ex_map = {**{("T2", ex): p for ex, p in zip(obj.t2_excitations, obj.t2_paulis)},
                      **{("S-hole", ex): p for ex, p in zip(obj.sh_excitations, obj.sh_paulis)},
                      **{("S-particle", ex): p for ex, p in zip(obj.sp_excitations, obj.sp_paulis)},
                      **{("S-standard", ex): p for ex, p in zip(obj.s_excitations, obj.s_paulis)}}
            obj.final_paulis = [ex_map[(info["type"], info["excitation"])] for info in obj.final_op_info]
            obj.final_ansatze = [EvolvedOperatorAnsatz(obj.final_paulis, initial_state=ref) for ref in obj.init_states]

        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        obj._log(f"\n{'*'*74}\n RESTORED FROM CHECKPOINT at {now} | Last stage: {obj._completed_stage or 'NONE'}\n{'*'*74}")
        return obj
    # ------------------------------------------------------------------
    # Stage 1: HF_SOLVER
    # ------------------------------------------------------------------
    def HF_SOLVER(self) -> dict[str, Any]:
        stage_start = perf_counter()
        self._log_stage_header("HF_SOLVER")

        scf_method = MethodType.RHF if self.spin == 0 else MethodType.ROHF
        driver = PySCFDriver(atom=self.coordinates,charge=self.charge,
                    spin=self.spin,basis=self.basis_set,
                    method=scf_method)    
        es_problem = cast(ElectronicStructureProblem, driver.run())
        self.mf = driver._calc
        self.hf_energy = float(self.mf.e_tot)

        if not self.mf.converged:
            raise RuntimeError(f"SCF failed to converge after {self.mf.max_cycle} cycles. "
                            f"Last energy: {self.mf.e_tot:.12f} Ha")
        
        if self.nfc:
            es_problem = cast(ElectronicStructureProblem, FreezeCoreTransformer(freeze_core=True).transform(es_problem))

        self.es_problem = es_problem
        if es_problem.num_particles is None or es_problem.nuclear_repulsion_energy is None:
            raise ValueError("PySCFDriver returned incomplete particle or nuclear repulsion data.")

        self.num_particles = es_problem.num_particles
        self.n_electrons = sum(self.num_particles)
        self.num_spatial_orbitals = int(cast(Any, es_problem.num_spatial_orbitals))
        self.num_spin_orbitals = 2 * self.num_spatial_orbitals
        self.ncore = (self.mf.mol.nelectron - self.n_electrons) // 2 if self.nfc else 0

        m = self.num_spatial_orbitals
        self.mo_coeff = np.asarray(self.mf.mo_coeff[:, self.ncore:self.ncore + m], dtype=float)
        self.mo_energy = np.asarray(self.mf.mo_energy[self.ncore:self.ncore + m], dtype=float)
        self.mo_occ = np.asarray(self.mf.mo_occ[self.ncore:self.ncore + m], dtype=float)

        c_act = self.mo_coeff
        h1_ao = self.mf.get_hcore()
        if self.ncore > 0:
            c_core = np.asarray(self.mf.mo_coeff[:, :self.ncore], dtype=float)
            dm_core = 2.0 * (c_core @ c_core.T)
            vj, vk = self.mf.get_jk(dm=dm_core)
            h1_ao = h1_ao + vj - 0.5 * vk

        self.h1e = np.asarray(c_act.T @ h1_ao @ c_act, dtype=float)
        eri_chem = ao2mo.restore(1, ao2mo.kernel(self.mf.mol, c_act), m)
        # Convert chemist's notation (pr|qs) -> physicist's notation <pq|rs> expected by ElectronicEnergy
        self.h2e = np.asarray(np.swapaxes(eri_chem, 1, 2), dtype=float)
        self.nuclear_repulsion = float(es_problem.nuclear_repulsion_energy)
        self.shift = float(np.real(sum(es_problem.hamiltonian.constants.values())))
        self.fermionic_op = es_problem.hamiltonian.second_q_op()
        self.qubit_op = self.mapper.map(self.fermionic_op)

        self._log(f" SCF Done:  E({scf_method.value.upper()}) = {self.hf_energy:18.12f} Ha  |  Active MOs = {m} (Core = {self.ncore})")
        self._log(f" Active Electrons: N_alpha = {self.num_particles[0]}, N_beta = {self.num_particles[1]} (Total N = {self.n_electrons})")
        self._log(f" Nuclear Repulsion = {self.nuclear_repulsion:16.12f} Ha  |  Shift = {self.shift:16.12f} Ha")
        self._log(f" HF_SOLVER completed in {perf_counter() - stage_start:.2f} s.")

        self.save_checkpoint("HF_SOLVER")
        return {
            "hf_energy": self.hf_energy, "num_particles": self.num_particles, "n_electrons": self.n_electrons,
            "num_spatial_orbitals": self.num_spatial_orbitals, "mo_coeff": self.mo_coeff, "mo_energy": self.mo_energy,
            "mo_occ": self.mo_occ, "h1e": self.h1e, "h2e": self.h2e, "shift": self.shift,
        }
    # ------------------------------------------------------------------
    # Stage 2: initialize_states (Fixed-N Spin-Complete CIS)
    # ------------------------------------------------------------------
    def initialize_states(self) -> dict[str, Any]:
        stage_start = perf_counter()
        self._log_stage_header("initialize_states")
        self._require("mf", "fermionic_op", "num_spatial_orbitals", "num_spin_orbitals", "n_electrons", "shift")

        m = cast(int, self.num_spatial_orbitals)
        n_spin = cast(int, self.num_spin_orbitals)
        n_elec = cast(int, self.n_electrons)
        e_shift = cast(float, self.shift)

        hf_det = active_hf_determinant(self.mf, self.ncore, m)
        occ = [p for p in range(n_spin) if (hf_det >> p) & 1]
        virt = [p for p in range(n_spin) if not ((hf_det >> p) & 1)]

        # Fixed-N CIS across all spin channels + CSF spin closure for exact S^2
        raw_dets = {hf_det} | {hf_det ^ (1 << i) ^ (1 << a) for i in occ for a in virt}
        closed_dets: set[int] = set()
        for det in raw_dets:
            doubly = [p for p in range(m) if ((det >> p) & 1) + ((det >> (p + m)) & 1) == 2]
            singly = [p for p in range(m) if ((det >> p) & 1) + ((det >> (p + m)) & 1) == 1]
            base = sum((1 << p) | (1 << (p + m)) for p in doubly)
            for spins in product((0, m), repeat=len(singly)):
                closed_dets.add(base | sum(1 << (p + s) for p, s in zip(singly, spins)))

        alpha_mask = (1 << m) - 1
        sectors: dict[tuple[int, int], list[int]] = defaultdict(list)
        for det in sorted(closed_dets):
            na, nb = (det & alpha_mask).bit_count(), (det >> m).bit_count()
            if na + nb == n_elec:
                sectors[(na, nb)].append(det)

        all_dets = [det for sec in sorted(sectors) for det in sectors[sec]]
        det_to_idx = {det: idx for idx, det in enumerate(all_dets)}

        candidates: list[tuple[float, np.ndarray]] = []
        for sec in sorted(sectors):
            sec_dets = sectors[sec]
            k_sec = min(self.no_init_states, len(sec_dets))
            evals, evecs = lowest_eigenvectors(projected_hamiltonian(self.fermionic_op, sec_dets), k_sec)
            for col in range(k_sec):
                full_vec = np.zeros(len(all_dets), dtype=complex)
                for row, det in enumerate(sec_dets):
                    full_vec[det_to_idx[det]] = evecs[row, col]
                candidates.append((float(evals[col]) + e_shift, full_vec))

        candidates.sort(key=lambda item: item[0])
        selected = candidates[:self.no_init_states]

        self.sci_determinants = all_dets
        self.sci_energies = np.asarray([e for e, _ in selected], dtype=float)
        self.sci_eigenvectors = np.column_stack([v for _, v in selected])
        self.sci_s2 = np.asarray([compute_s2_expectation(all_dets, self.sci_eigenvectors[:, i], m)
                                  for i in range(self.no_init_states)], dtype=float)

        self._log(f" Spin sectors explored (N={n_elec}): {sorted(sectors.keys())}")
        self._log(f" CSF-closed fixed-N CIS determinant space dimension: {len(all_dets)}")
        self._log(f"   {'Root':>4}   {'E_SCI (Ha)':>18}   {'Delta_E (Ha)':>14}   {'<S^2>':>8}   {'Mult':>4}")
        self._log(f"   {'-'*4}   {'-'*18}   {'-'*14}   {'-'*8}   {'-'*4}")
        e0 = self.sci_energies[0]
        for i, (en, s2) in enumerate(zip(self.sci_energies, self.sci_s2)):
            mult = int(round(np.sqrt(max(0.0, 1.0 + 4.0 * s2))))
            self._log(f"   {i:>4}   {en:18.12f}   {en - e0:14.8f}   {s2:8.4f}   {mult:>4}")
        self._log(f" initialize_states completed in {perf_counter() - stage_start:.2f} s.")

        self.save_checkpoint("initialize_states")
        return {"sci_determinants": self.sci_determinants, "sci_eigenvectors": self.sci_eigenvectors,
                "sci_energies": self.sci_energies, "sci_s2": self.sci_s2}
    # ------------------------------------------------------------------
    # Stage 3: compute_1rdm (NOs, 4-Fold Classification, NO-Basis Rotation)
    # ------------------------------------------------------------------
    def compute_1rdm(self, state_index: int | None = None) -> dict[str, Any]:
        stage_start = perf_counter()
        self._log_stage_header("compute_1rdm")
        self._require("sci_determinants", "sci_eigenvectors", "sci_energies", "h1e", "h2e", "mo_coeff",
                      "es_problem", "num_spatial_orbitals", "num_spin_orbitals", "shift")

        m = cast(int, self.num_spatial_orbitals)
        n_spin = cast(int, self.num_spin_orbitals)
        sci_energies = cast(np.ndarray, self.sci_energies)
        sci_vecs = cast(np.ndarray, self.sci_eigenvectors)
        e_shift = cast(float, self.shift)

        target_idx = (self.no_init_states - 1) if state_index is None else int(state_index)
        if not 0 <= target_idx < len(sci_energies):
            raise ValueError(f"state_index {target_idx} out of range for {len(sci_energies)} states.")
        dets = cast(list[int], self.sci_determinants)

        t_sub = perf_counter()
        gamma = np.asarray(build_spin_summed_1rdm(dets, sci_vecs[:, target_idx], m), dtype=float)
        evals, evecs = np.linalg.eigh(gamma)
        order = np.argsort(evals)[::-1]
        self.noons = np.clip(evals[order], 0.0, 2.0)
        self.no_transform = evecs[:, order]
        self.rdm1_spatial = gamma
        self.no_coeff = cast(np.ndarray, self.mo_coeff) @ self.no_transform
        self._log(f"   [timer] 1-RDM build & diagonalization: {perf_counter() - t_sub:.3f} s")

        oi_sp = [p for p, n in enumerate(self.noons) if n > self.occ_threshold]
        oa_sp = [p for p, n in enumerate(self.noons) if 1.0 <= n <= self.occ_threshold]
        va_sp = [p for p, n in enumerate(self.noons) if self.virt_threshold <= n < 1.0]
        vi_sp = [p for p, n in enumerate(self.noons) if n < self.virt_threshold]

        self.inactive_occ = oi_sp + [p + m for p in oi_sp]
        self.active_occ = oa_sp + [p + m for p in oa_sp]
        self.active_virt = va_sp + [p + m for p in va_sp]
        self.inactive_virt = vi_sp + [p + m for p in vi_sp]
        self.active_orbitals = self.active_occ + self.active_virt

        t_sub = perf_counter()
        u = self.no_transform
        self.h1e_no = u.T @ cast(np.ndarray, self.h1e) @ u
        self.h2e_no = np.einsum("pi,qj,pqrs,rk,sl->ijkl", u, u, cast(np.ndarray, self.h2e), u, u, optimize=True)
        elec_energy = ElectronicEnergy.from_raw_integrals(self.h1e_no, cast(np.ndarray, self.h2e_no))
        elec_energy.constants = dict(cast(ElectronicStructureProblem, self.es_problem).hamiltonian.constants)
        cast(Any, self.es_problem)._hamiltonian = elec_energy
        self.fermionic_op = elec_energy.second_q_op()
        self.qubit_op = self.mapper.map(self.fermionic_op)
        self._log(f"   [timer] NO integral transform & JW map: {perf_counter() - t_sub:.3f} s")

        t_sub = perf_counter()
        self.no_determinants, self.sci_eigenvectors_no = rotate_ci_vectors_to_no(dets, sci_vecs, u, m)
        self._log(f"   [timer] rotate_ci_vectors_to_no ({len(self.no_determinants)} dets): {perf_counter() - t_sub:.3f} s")

        t_sub = perf_counter()
        raw_circuits = determinants_to_circuits(self.no_determinants, self.sci_eigenvectors_no, n_spin)
        self._log(f"   [timer] determinants_to_circuits: {perf_counter() - t_sub:.3f} s")

        t_sub = perf_counter()
        self.init_states = qk_transpile(raw_circuits, optimization_level=1)
        depths = [qc.depth() for qc in self.init_states]
        self._log(f"   [timer] qk_transpile init_states (depths={depths}): {perf_counter() - t_sub:.3f} s")

        t_sub = perf_counter()
        self.initial_energies = state_energies(self.estimator, self.init_states, self.qubit_op, shift=e_shift)
        self._log(f"   [timer] initial state_energies: {perf_counter() - t_sub:.3f} s")
        max_rot_err = float(np.max(np.abs(self.initial_energies - sci_energies)))

        self._log(f" 1-RDM constructed using SCI root {target_idx}")
        self._log(f"   {'NO':>3}   {'NOON':>10}   {'Class':<16}   {'Spin-Orbitals (a, b)'}")
        self._log(f"   {'-'*3}   {'-'*10}   {'-'*16}   {'-'*20}")
        for p, occ_num in enumerate(self.noons):
            cls_tag = ("O_I (Inact-Occ)" if p in oi_sp else
                       "O_A (Act-Occ)" if p in oa_sp else
                       "V_A (Act-Virt)" if p in va_sp else "V_I (Inact-Virt)")
            self._log(f"   {p:>3}   {occ_num:10.6f}   {cls_tag:<16}   ({p}, {p + m})")

        self._log(f" Unitary NO-basis invariance check: max |E_NO - E_SCI| = {max_rot_err:.2e} Ha")
        self._log(f" Compiled {len(self.init_states)} sparse Givens circuits on {n_spin} NO qubits.")
        self._log(f" compute_1rdm completed in {perf_counter() - stage_start:.2f} s.")

        self.save_checkpoint("compute_1rdm")
        return {
            "noons": self.noons, "no_transform": self.no_transform, "inactive_occ": self.inactive_occ,
            "active_occ": self.active_occ, "active_virt": self.active_virt, "inactive_virt": self.inactive_virt,
            "init_states": self.init_states, "initial_energies": self.initial_energies,
        }
    # ------------------------------------------------------------------
    # Stage 3b: Optional Exact Fixed-N Spectrum
    # ------------------------------------------------------------------
    def compute_exact_spectrum(self) -> np.ndarray:
        stage_start = perf_counter()
        self._log_stage_header("compute_exact_spectrum")
        self._require("fermionic_op", "num_spatial_orbitals", "num_spin_orbitals", "n_electrons", "shift")

        m = cast(int, self.num_spatial_orbitals)
        n_spin = cast(int, self.num_spin_orbitals)
        n_elec = cast(int, self.n_electrons)
        e_shift = cast(float, self.shift)

        total_dim = int(comb(n_spin, n_elec))
        if not 1 <= self.no_init_states <= total_dim:
            raise ValueError(f"Requested {self.no_init_states} states, but fixed-N space dim is {total_dim}.")

        all_evals: list[float] = []
        for na in range(max(0, n_elec - m), min(m, n_elec) + 1):
            nb = n_elec - na
            sec_dets = [sum(1 << p for p in occ_a) | sum(1 << (q + m) for q in occ_b)
                        for occ_a in combinations(range(m), na) for occ_b in combinations(range(m), nb)]
            if sec_dets:
                evals, _ = lowest_eigenvectors(projected_hamiltonian(self.fermionic_op, sec_dets),
                                               min(self.no_init_states, len(sec_dets)))
                all_evals.extend(evals.tolist())

        all_evals.sort()
        self.exact_energies = np.asarray(all_evals[:self.no_init_states], dtype=float) + e_shift

        self._log(f" Fixed-N Hilbert space dimension (N={n_elec}, 2M={n_spin}): {total_dim}")
        for i, ex in enumerate(self.exact_energies):
            self._log(f"   Exact Root {i:>2}: {ex:18.12f} Ha")
        self._log(f" compute_exact_spectrum completed in {perf_counter() - stage_start:.2f} s.")

        self.save_checkpoint("compute_exact_spectrum")
        return self.exact_energies
    # ------------------------------------------------------------------
    # Stage 4: pool_creator
    # ------------------------------------------------------------------
    def pool_creator(self) -> dict[str, Any]:
            stage_start = perf_counter()
            self._log_stage_header("pool_creator")
            self._require("num_spatial_orbitals", "num_particles")
    
            pools = build_operator_pools(
                cast(int, self.num_spatial_orbitals), cast(tuple[int, int], self.num_particles), self.mapper,
                self.inactive_occ, self.active_occ, self.active_virt, self.inactive_virt,
            )
            for key, val in pools.items():
                setattr(self, key, val)
    
            self._log(f" NO-Basis Operator Pools (OP Scatterers, Active CSO = {self.active_orbitals}):")
            self._log(f"   S (Singles) = {len(self.s_paulis)}  |  T2 (MR-Doubles) = {len(self.t2_paulis)}  |  "
                      f"Sh = {len(self.sh_paulis)}  |  Sp = {len(self.sp_paulis)}")
            self._log(f" pool_creator completed in {perf_counter() - stage_start:.2f} s.")
    
            self.save_checkpoint("pool_creator")
            return pools
 # ------------------------------------------------------------------
    # Stage 5a: T2 screening 
    # ------------------------------------------------------------------
    def _screen_t2(self) -> list[tuple[float, int, float, np.ndarray]]:
        t0 = perf_counter()
        self.t2_screen, self.t2_screening_data = screen_t2_pool(
            self.t2_paulis, self.t2_excitations, cast(list[QuantumCircuit], self.init_states),
            self.qubit_op, cast(float, self.shift), self.weights, self.screening_weights,
            cast(np.ndarray, self.initial_energies), float(self.threshold[0]),
            self.n_workers, self._mp_ctx, self._log)

        t2_json_path = self.log_path.with_name(self.log_path.stem + "_t2_screening.json")
        import json
        with open(t2_json_path, "w", encoding="utf-8") as fh:
            json.dump(self.t2_screening_data, fh, indent=2)
        self._log(f" Saved T2 screening data to {t2_json_path}")

        self._log(f" _screen_t2 accepted {len(self.t2_screen)}/{len(self.t2_paulis)} in {perf_counter() - t0:.2f} s.")
        self.save_checkpoint("screen_t2")
        return self.t2_screen
    # ------------------------------------------------------------------
    # Stage 5b: Scatters screening 
    # ------------------------------------------------------------------
    def _screen_s(self) -> dict[int, list[tuple[Any, ...]]]:
        t0 = perf_counter()
        self.s_candidates = build_s_candidates(
            self.t2_screen, self.t2_excitations, self.sh_excitations, self.sp_excitations, self.active_orbitals)

        def _periodic_chk(partial: dict[int, Any]) -> None:
            self._s_partial_results = dict(partial)
            self.save_checkpoint()

        self.s_by_rank, self.s_screening_data = screen_s_pool(
            self.s_candidates, self.t2_paulis, self.sh_paulis, self.sp_paulis,
            self.sh_excitations, self.sp_excitations, cast(list[QuantumCircuit], self.init_states),
            self.qubit_op, cast(float, self.shift), self.weights, self.screening_weights,
            float(self.threshold[1]), self.n_workers, self._mp_ctx, self._log,
            cached_results=self._s_partial_results, checkpoint_fn=_periodic_chk)
        self._s_partial_results = {}

        s_json_path = self.log_path.with_name(self.log_path.stem + "_s_screening.json")
        import json
        with open(s_json_path, "w", encoding="utf-8") as fh:
            json.dump(self.s_screening_data, fh, indent=2)
        self._log(f" Saved S-scatterer screening data to {s_json_path}")

        n_acc = sum(len(v) for v in self.s_by_rank.values())
        self._log(f" _screen_s accepted {n_acc}/{len(self.s_candidates)} in {perf_counter() - t0:.2f} s.")
        return self.s_by_rank
    # ------------------------------------------------------------------
    # Stage 5c: main screener (Orchestrates _screen_t2 and _screen_s)
    # ------------------------------------------------------------------
    def screener(self) -> dict[str, Any]:
        stage_start = perf_counter()
        self._log_stage_header("screener")
        self._require("qubit_op", "init_states", "initial_energies", "shift", "t2_paulis")
        if self._completed_stage != "screen_t2":
            self._screen_t2()
        self._screen_s()
        self._log(f" screener completed in {perf_counter() - stage_start:.2f} s.")
        self.save_checkpoint("screener")
        return {"t2_screen": self.t2_screen, "s_by_rank": dict(self.s_by_rank),
                "t2_screening": self.t2_screening_data, "s_screening": self.s_screening_data}
    # ------------------------------------------------------------------
    # Stage 6: ansatz_assembler (Adaptive 1P / 2P Parameter Initialization)
    # ------------------------------------------------------------------
    def ansatz_assembler(self, init_states: list[QuantumCircuit] | None = None) -> list[QuantumCircuit]:
        stage_start = perf_counter()
        self._log_stage_header("ansatz_assembler")
        if init_states is not None:
            self.init_states = init_states
        self._require("init_states")

        self.final_paulis, final_opt_t, self.final_op_info = [], [], []
        for rank, (_, t2_idx, theta_t2_1p, _) in enumerate(self.t2_screen):
            scatterers = self.s_by_rank.get(rank, [])
            theta_t2_init = scatterers[0][3] if len(scatterers) == 1 else theta_t2_1p

            self.final_paulis.append(self.t2_paulis[t2_idx])
            final_opt_t.append(theta_t2_init)
            self.final_op_info.append({"type": "T2", "excitation": self.t2_excitations[t2_idx], "theta": theta_t2_init})

            for s_pauli, s_ex, label, _, opt_s_theta in scatterers:
                self.final_paulis.append(s_pauli)
                final_opt_t.append(opt_s_theta)
                self.final_op_info.append({"type": label, "excitation": s_ex, "theta": opt_s_theta})

        for i, s_ex in enumerate(self.s_excitations):
            self.final_paulis.append(self.s_paulis[i])
            final_opt_t.append(0.01)
            self.final_op_info.append({"type": "S-standard", "excitation": s_ex, "theta": 0.01})

        if not self.final_paulis:
            raise RuntimeError("ansatz_assembler produced 0 operators; lower threshold or check NOON cutoffs.")    

        self.initial_point = np.asarray(final_opt_t, dtype=np.float64)
        self.final_ansatze = [EvolvedOperatorAnsatz(self.final_paulis, initial_state=ref)
                              for ref in cast(list[QuantumCircuit], self.init_states)]

        self._log(f" Assembled Ansatz: {len(self.final_paulis)} operator(s), {len(self.initial_point)} parameter(s).")
        self._log(f"   {'#':>4}   {'Type':<12}   {'Excitation':<30}   {'Theta_Init':>10}")
        self._log(f"   {'-'*4}   {'-'*12}   {'-'*30}   {'-'*10}")
        for i, info in enumerate(self.final_op_info, 1):
            self._log(f"   {i:>4}   {info['type']:<12}   {str(info['excitation']):<30}   {info['theta']:10.6f}")
        self._log(f" ansatz_assembler completed in {perf_counter() - stage_start:.2f} s.")
        self.save_checkpoint("ansatz_assembler")
        return self.final_ansatze
    # ------------------------------------------------------------------
    # Stage 7: final_vqe_runner (Global SS-VQE Optimization & Recovery)
    # ------------------------------------------------------------------
    def final_vqe_runner(self, log_interval: int = 1, checkpoint_interval: int = 50) -> dict[str, Any]:
        stage_start = perf_counter()
        self._log_stage_header("final_vqe_runner")
        self._require("qubit_op", "final_ansatze", "initial_point", "shift")

        ansatze = cast(list[QuantumCircuit], self.final_ansatze)
        transpiled = qk_transpile(ansatze, optimization_level=1)
        e_shift = cast(float, self.shift)

        start_pt = (np.asarray(self.best_params, dtype=np.float64)
                    if (self.best_params is not None and len(self.best_params) == len(cast(np.ndarray, self.initial_point)))
                    else np.asarray(self.initial_point, dtype=np.float64))

        self._log(f" Starting Global SS-VQE: {len(start_pt)} parameters | {len(transpiled)} states | "
                  f"Resumed evals = {self.vqe_eval_count}")

        def _vqe_callback(eval_count: int, params: np.ndarray, cost: float, metadata: dict[str, Any]) -> None:
            self.vqe_eval_count += 1
            step = self.vqe_eval_count
            p_copy = np.asarray(params, dtype=float).copy()
            self.latest_params = p_copy
            total_cost = float(cost) + float(np.sum(self.weights)) * e_shift

            raw_e = metadata.get("state_energies")
            cur_e = (np.asarray(raw_e, dtype=float) + e_shift) if raw_e is not None else None

            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_params = p_copy.copy()
                if cur_e is not None:
                    self.best_state_energies = cur_e.copy()

            if step == 1 or step % log_interval == 0:
                e_str = (
                    "[" + ", ".join(f"{val:.8f}" for val in cur_e) + "]"
                    if cur_e is not None else "N/A"
                )
                self._log(
                    f"   Eval {step:>5d} | Weighted E = {total_cost:16.10f} Ha | "
                    f"States = {e_str} | Best = {self.best_cost:16.10f} Ha"
                )

            if step % checkpoint_interval == 0:
                self.latest_state_energies = (
                    cur_e.copy() if cur_e is not None
                    else state_energies(self.estimator, transpiled, self.qubit_op, p_copy, shift=e_shift)
                )
                self.save_checkpoint()

        interrupted = {"flag": False, "sig": None}
        prev_int, prev_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

        def _handle_signal(signum: int, _frame: Any) -> None:
            interrupted["flag"] = True
            interrupted["sig"] = signum
            raise KeyboardInterrupt(f"Signal {signum} received during final_vqe_runner.")

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            solver = SS_VQE(cast(Any, self.estimator), transpiled, self.weights, self.optimizer,
                            initial_point=start_pt, callback=_vqe_callback)
            self.final_result = solver.compute_minimum_eigenvalue(self.qubit_op)
            opt_params = np.asarray(self.final_result.optimal_point, dtype=float)
            self.state_energies = state_energies(self.estimator, transpiled, self.qubit_op, opt_params, shift=e_shift)
            self.best_params = opt_params.copy()
            self.best_state_energies = self.state_energies.copy()
            self.best_cost = float(np.dot(self.weights, self.state_energies))
        except KeyboardInterrupt:
            self._log(f"\n [WARN] Interrupted (signal={interrupted['sig']}) at eval {self.vqe_eval_count}. Saving recovery checkpoint...")
            if self.best_params is not None:
                self.best_state_energies = state_energies(self.estimator, transpiled, self.qubit_op, self.best_params, shift=e_shift)
                self.state_energies = self.best_state_energies.copy()
            self.save_checkpoint()
            raise
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)

        self._log(f"\n {'Root':>4}   {'E_Init (Ha)':>18}   {'E_SS-VQE (Ha)':>18}"
                  + (f"   {'E_Exact (Ha)':>18}   {'Error (mHa)':>12}" if self.exact_energies is not None else ""))
        self._log(f" {'-'*4}   {'-'*18}   {'-'*18}" + (f"   {'-'*18}   {'-'*12}" if self.exact_energies is not None else ""))
        for i, e_fin in enumerate(self.state_energies):
            e_ini = cast(np.ndarray, self.initial_energies)[i]
            row = f" {i:>4}   {e_ini:18.12f}   {e_fin:18.12f}"
            if self.exact_energies is not None:
                err_mha = (e_fin - self.exact_energies[i]) * 1e3
                row += f"   {self.exact_energies[i]:18.12f}   {err_mha:12.6f}"
            self._log(row)

        self._log(f" final_vqe_runner completed in {perf_counter() - stage_start:.2f} s ({self.vqe_eval_count} evals).")
        self.save_checkpoint("final_vqe_runner")
        return {"state_energies": self.state_energies, "optimal_params": self.best_params,
                "best_cost": self.best_cost, "vqe_eval_count": self.vqe_eval_count,
                "t2_screening": self.t2_screening_data, "s_screening": self.s_screening_data,
                "result": self.final_result}
    # ------------------------------------------------------------------
    # End-to-End Pipeline Runner (Supports Fresh & Resumed Runs)
    # ------------------------------------------------------------------
    def run(self, compute_exact: bool = False, rdm_state_index: int | None = None) -> dict[str, Any]:
        self._run_start = perf_counter()
        completed = self._completed_stage
        idx = _STAGE_ORDER.index(completed) if completed in _STAGE_ORDER else -1

        if idx < _STAGE_ORDER.index("HF_SOLVER"):
            self.HF_SOLVER()
        if idx < _STAGE_ORDER.index("initialize_states"):
            self.initialize_states()
        if idx < _STAGE_ORDER.index("compute_1rdm"):
            self.compute_1rdm(state_index=rdm_state_index)
        if compute_exact and self.exact_energies is None:
            self.compute_exact_spectrum()
        if idx < _STAGE_ORDER.index("pool_creator"):
            self.pool_creator()
        if idx < _STAGE_ORDER.index("screener"):
            self.screener()
        if idx < _STAGE_ORDER.index("ansatz_assembler"):
            self.ansatz_assembler()
        res = self.final_vqe_runner()

        self._log(f"\n{'*'*74}\n Normal termination of SS-COMPASS-VQE in {perf_counter() - self._run_start:.2f} s.\n{'*'*74}")
        if self._log_fh is not None and not self._log_fh.closed:
            self._log_fh.close()
        return res