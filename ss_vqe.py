"""Subspace-Search Variational Quantum Eigensolver (SS-VQE)."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Sequence, cast

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms import VQEResult
from qiskit_algorithms.custom_types import Transpiler
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.gradients import (
    BaseEstimatorGradient,
    ParamShiftEstimatorGradient,
    ReverseEstimatorGradient,
)
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.minimum_eigensolvers.minimum_eigensolver import MinimumEigensolver
from qiskit_algorithms.observables_evaluator import estimate_observables
from qiskit_algorithms.optimizers import Minimizer, Optimizer, OptimizerResult
from qiskit_algorithms.utils import validate_bounds, validate_initial_point
from qiskit_algorithms.utils.set_batching import _set_default_batchsize
from qiskit_algorithms.variational_algorithm import VariationalAlgorithm

logger = logging.getLogger(__name__)

class SS_VQEResult(VQEResult):
    """Result object for SS-VQE containing per-state optimal circuits and observables."""

    def __init__(self) -> None:
        super().__init__()
        self.optimal_circuits: list[QuantumCircuit] = []
        self.aux_operators_evaluated_by_state: (
            list[ListOrDict[tuple[complex, dict[str, Any]]] | None] | None
        ) = None

class SS_VQE(VariationalAlgorithm, MinimumEigensolver):
    """Subspace-Search VQE minimizing a weighted sum of expectation values across states."""

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        ansatze: list[QuantumCircuit],
        weights: np.ndarray,
        optimizer: Optimizer | Minimizer,
        *,
        gradient: BaseEstimatorGradient | None = None,
        initial_point: np.ndarray | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self._ansatze = ansatze
        self._original_ansatze = ansatze
        self.weights = np.asarray(weights, dtype=float)

        if not self._ansatze:
            raise ValueError("SS-VQE requires at least one ansatz.")
        if self.weights.ndim != 1:
            raise ValueError("Weights must be a one-dimensional array.")
        if len(self._ansatze) != len(self.weights):
            raise ValueError("Number of ansatze must match number of weights.")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Weights must be finite.")

        n_params = self._ansatze[0].num_parameters
        for i, ansatz in enumerate(self._ansatze):
            if ansatz.num_parameters != n_params:
                raise ValueError(f"Ansatz {i} has {ansatz.num_parameters} parameters; expected {n_params}.")

        self.optimizer = optimizer
        self.gradient = gradient
        self.initial_point = initial_point
        self.callback = callback
        self._transpiler = transpiler
        self._transpiler_options = transpiler_options or {}
        if self._transpiler is not None:
            self.ansatze = ansatze

    @property
    def initial_point(self) -> np.ndarray | None:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray | None) -> None:
        self._initial_point = initial_point

    @property
    def ansatze(self) -> list[QuantumCircuit]:
        return cast(list[QuantumCircuit], self._ansatze)

    @ansatze.setter
    def ansatze(self, value: list[QuantumCircuit]) -> None:
        if self._transpiler is None:
            self._ansatze = value
            return
        transpiled: list[QuantumCircuit] = []
        for ansatz in value:
            res = self._transpiler.run(ansatz, **self._transpiler_options)
            transpiled.extend(res if isinstance(res, list) else [res])
        if len(transpiled) != len(value):
            raise ValueError("Transpilation changed the number of ansatz circuits.")
        self._ansatze = transpiled

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> SS_VQEResult:
        if not self.ansatze:
            raise ValueError("SS-VQE requires at least one ansatz.")
        if self.ansatze[0].layout is not None:
            operator = cast(Any, operator).apply_layout(self.ansatze[0].layout)

        self._check_operator_ansatze(operator)
        x0 = validate_initial_point(self.initial_point, self.ansatze[0])
        bounds = validate_bounds(self.ansatze[0])

        eval_energy = self.ss_get_evaluate_energy(self.ansatze, operator, self.weights)
        eval_grad = self.gradient or self.ss_get_evaluate_gradient(self.ansatze, operator, self.weights)

        start_time = time()
        if callable(self.optimizer):
            opt_res = cast(
                OptimizerResult,
                self.optimizer(fun=eval_energy, x0=x0, jac=cast(Any, eval_grad), bounds=cast(Any, bounds)),  # type: ignore[arg-type]
            )
        else:
            was_updated = _set_default_batchsize(self.optimizer)
            opt_res = self.optimizer.minimize(
                fun=eval_energy, x0=x0, jac=eval_grad, bounds=cast(Any, bounds)  # type: ignore[arg-type]
            )
            if was_updated:
                self.optimizer.set_max_evals_grouped(None)
        opt_time = time() - start_time
        logger.info("Optimization complete in %s seconds.\nFound optimal point %s", opt_time, opt_res.x)

        aux_eval = None
        if aux_operators is not None:
            if opt_res.x is None:
                raise ValueError("Optimizer did not return an optimal parameter vector.")
            aux_eval = [
                estimate_observables(
                    self.estimator,
                    ansatz,
                    self._apply_layout_to_aux(aux_operators, ansatz.layout),
                    cast(Sequence[float], opt_res.x),
                )
                for ansatz in self.ansatze
            ]

        return self._build_ss_vqe_result(self.ansatze, opt_res, aux_eval, opt_time)  # type: ignore[arg-type]

    def ss_get_evaluate_energy(
        self, ansatze: list[QuantumCircuit], operator: BaseOperator, weights: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray | float]:
        n_params = ansatze[0].num_parameters
        eval_count = 0

        def _eval_one(ansatz: QuantumCircuit, params: np.ndarray):
            return self.estimator.run([(ansatz, operator, params.tolist())]).result()[0]

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count
            param_sets = np.reshape(parameters, (-1, n_params))
            try:
                with ThreadPoolExecutor(max_workers=len(ansatze)) as pool:
                    futures = [pool.submit(_eval_one, a, p) for p in param_sets for a in ansatze]
                    results = [f.result() for f in futures]
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the SS-VQE energy failed!") from exc

            n_sets = len(param_sets)
            energies = []
            for idx in range(len(ansatze)):
                vals = [np.atleast_1d(np.asarray(cast(Any, r.data).evs)) for r in results[idx::len(ansatze)]]
                energies.append(np.asarray(vals).reshape(n_sets, -1).T)

            energies_arr = np.asarray(energies).squeeze()
            if energies_arr.ndim == 1:
                energies_arr = energies_arr[:, None]
            costs = np.sum(np.asarray(weights)[:, None] * energies_arr, axis=0)

            if self.callback is not None:
                for p, val in zip(param_sets, costs):
                    eval_count += 1
                    self.callback(eval_count, p, val, results[0].metadata)

            return costs[0] if len(costs) == 1 else costs

        return evaluate_energy

    def ss_get_evaluate_gradient(
        self, ansatze: list[QuantumCircuit], operator: BaseOperator, weights: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        grad_est = self.gradient or (
            ReverseEstimatorGradient()
            if isinstance(self.estimator, StatevectorEstimator)
            else ParamShiftEstimatorGradient(self.estimator)
        )

        def _eval_one(ansatz: QuantumCircuit, params: np.ndarray):
            return grad_est.run([ansatz], [operator], [params.tolist()]).result().gradients[0]

        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            try:
                with ThreadPoolExecutor(max_workers=len(ansatze)) as pool:
                    futures = [pool.submit(_eval_one, a, parameters) for a in ansatze]
                    grads = np.asarray([f.result() for f in futures])
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the SS-VQE gradient failed!") from exc
            return np.sum(np.asarray(weights)[:, None] * grads, axis=0)

        return evaluate_gradient

    @staticmethod
    def _apply_layout_to_aux(aux_operators: ListOrDict[BaseOperator], layout: Any) -> ListOrDict[BaseOperator]:
        if layout is None:
            return aux_operators
        zero_op = SparsePauliOp.from_list([("I" * len(layout.final_index_layout()), 0)])
        map_op = lambda op: cast(BaseOperator, (zero_op if op == 0 else cast(Any, op)).apply_layout(layout))
        if isinstance(aux_operators, list):
            return [map_op(op) if op is not None else cast(BaseOperator, zero_op) for op in aux_operators]
        return {k: map_op(op) for k, op in aux_operators.items() if op is not None}

    def _check_operator_ansatze(self, operator: BaseOperator) -> None:
        if operator.num_qubits != self.ansatze[0].num_qubits:
            try:
                cast(Any, self.ansatze[0]).num_qubits = operator.num_qubits
                warnings.warn("Resizing BlueprintCircuit ansatz is deprecated.", category=DeprecationWarning)
            except AttributeError as err:
                raise AlgorithmError("Ansatz qubit count does not match operator and cannot be resized.") from err
        if self.ansatze[0].num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

    def _build_ss_vqe_result(
        self,
        ansatze: list[QuantumCircuit],
        opt_res: OptimizerResult,
        aux_eval: list[ListOrDict[tuple[complex, dict[str, Any]]] | None] | None,
        opt_time: float,
    ) -> SS_VQEResult:
        if opt_res.fun is None or opt_res.x is None or opt_res.nfev is None:
            raise ValueError("Optimizer result is missing fun, x, or nfev.")
        optimal_point = np.asarray(opt_res.x, dtype=float).reshape(-1)
        if len(ansatze[0].parameters) != len(optimal_point):
            raise ValueError("Optimized parameter count does not match ansatz parameters.")

        result = SS_VQEResult()
        result.optimal_circuits = [a.assign_parameters(dict(zip(a.parameters, optimal_point))) for a in ansatze]
        result.eigenvalue = complex(opt_res.fun)
        result.cost_function_evals = int(opt_res.nfev)
        result.optimal_point = optimal_point
        result.optimal_parameters = dict(zip(ansatze[0].parameters, optimal_point))
        result.optimal_value = float(opt_res.fun)
        result.optimizer_time = opt_time
        result.aux_operators_evaluated_by_state = aux_eval
        result.optimizer_result = opt_res
        return result