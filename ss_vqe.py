# IMPORTANT NOTE 
# This code assumes that all the ansatz in a list of ansatz ansatze(as it is called in this code) are similar in terms
# of the number of parameters, number of qubits, Bounds of the parameter, Layout, and Share the Same initial Point 

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

import logging
import warnings
from time import time
from collections.abc import Callable
from typing import Any, cast, Sequence

import numpy as np
from qiskit_algorithms import VQEResult
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_algorithms.gradients import (BaseEstimatorGradient,ReverseEstimatorGradient,
    ParamShiftEstimatorGradient)
from qiskit_algorithms.custom_types import Transpiler
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.optimizers import Optimizer, Minimizer, OptimizerResult
from qiskit_algorithms.variational_algorithm import VariationalAlgorithm
from qiskit_algorithms.minimum_eigensolvers.minimum_eigensolver import MinimumEigensolver
from qiskit_algorithms.observables_evaluator import estimate_observables
from qiskit_algorithms.utils import validate_initial_point, validate_bounds

# private function as we expect this to be updated in the next release
from qiskit_algorithms.utils.set_batching import _set_default_batchsize

logger = logging.getLogger(__name__)

class SS_VQE(VariationalAlgorithm, MinimumEigensolver):

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        ansatze: list[QuantumCircuit],
        weights : np.ndarray,
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

        # TODO: remove the following line once AdaptVQE doesn't use the EvolvedOperatorAnsatz class
        #  anymore
        self._original_ansatze = ansatze

        self.weights = np.asarray(weights, dtype=float)

        if len(self._ansatze) != len(self.weights):
            raise ValueError("Number of ansatze must match number of weights.")

        if not self._ansatze:
            raise ValueError("SS-VQE requires at least one ansatz.")

        if self.weights.ndim != 1:
            raise ValueError("Weights must be a one-dimensional array.")

        if len(self._ansatze) != len(self.weights):
            raise ValueError("Number of ansatze must match number of weights.")

        num_parameters = self._ansatze[0].num_parameters

        for i, ansatz in enumerate(self._ansatze):
            if ansatz.num_parameters != num_parameters:
                raise ValueError(
                    f"Ansatz {i} has {ansatz.num_parameters} parameters; "
                    f"expected {num_parameters}.")

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Weights must be finite.")

        self.optimizer = optimizer
        self.gradient = gradient
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback

        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

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
        if self._transpiler is not None:
            transpiled = []
            for ansatz in value:
                result = self._transpiler.run(ansatz,**self._transpiler_options)
                if len(self._ansatze) != len(value):
                    raise ValueError("Transpilation changed the number of ansatz circuits.")
                if isinstance(result, list):
                    transpiled.extend(result)
                else:
                    transpiled.append(result)
            self._ansatze = transpiled
        else:
            self._ansatze = value

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
        initial_point = validate_initial_point(self.initial_point, self.ansatze[0])
        bounds = validate_bounds(self.ansatze[0])
        start_time = time()
        evaluate_energy = self.ss_get_evaluate_energy(self.ansatze, operator, self.weights)

        if self.gradient is not None:
            evaluate_gradient = self.gradient
        else:
            evaluate_gradient = self.ss_get_evaluate_gradient(self.ansatze, operator, self.weights)

        # perform optimization
        if callable(self.optimizer):
            optimizer_result = cast(OptimizerResult, self.optimizer(
                fun=evaluate_energy,  # type: ignore[arg-type]
                x0=initial_point,
                jac=cast(Any,evaluate_gradient),
                bounds=cast(Any, bounds)))
        else:
            # we always want to submit as many estimations per job as possible for minimal
            # overhead on the hardware
            was_updated = _set_default_batchsize(self.optimizer)

            optimizer_result = self.optimizer.minimize(
                fun=evaluate_energy,  # type: ignore[arg-type]
                x0=initial_point,
                jac=evaluate_gradient,  # type: ignore[arg-type]
                bounds=cast(Any, bounds))

            # reset to original value
            if was_updated:
                self.optimizer.set_max_evals_grouped(None)
        optimizer_time = time() - start_time
        logger.info(
            "Optimization complete in %s seconds.\nFound optimal point %s",
            optimizer_time,
            optimizer_result.x)
        
        if aux_operators is not None:
            aux_operators_evaluated = []

            for ansatz in self.ansatze:
                current_aux_operators: ListOrDict[BaseOperator] = aux_operators
                if ansatz.layout is not None:
                    zero_op = SparsePauliOp.from_list(
                        [("I" * len(ansatz.layout.final_index_layout()),0)])
                    if isinstance(aux_operators, list):
                        converted: ListOrDict[BaseOperator] = [
                            cast(BaseOperator, zero_op)
                            for _ in aux_operators]

                        for i, op in enumerate(aux_operators):
                            if op is not None:
                                converted[i] = cast(
                                    BaseOperator,
                                    zero_op.apply_layout(ansatz.layout)
                                    if op == 0
                                    else cast(Any, op).apply_layout(ansatz.layout))

                    else:
                        converted = {}
                        for key, op in aux_operators.items():
                            if op is not None:
                                converted[key] = cast(
                                    BaseOperator,
                                    zero_op.apply_layout(ansatz.layout)
                                    if op == 0
                                    else cast(Any, op).apply_layout(ansatz.layout))

                    current_aux_operators = converted

                if optimizer_result.x is None:
                    raise ValueError("Optimizer did not return an optimal parameter vector.")

                evaluated = estimate_observables(
                    self.estimator,
                    ansatz,
                    current_aux_operators,
                    cast(Sequence[float], optimizer_result.x))

                aux_operators_evaluated.append(evaluated)

        else:
            aux_operators_evaluated = None

        return self._build_ss_vqe_result(
            self.ansatze,
            optimizer_result,
            aux_operators_evaluated,  # type: ignore[arg-type]
            optimizer_time)

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def ss_get_evaluate_energy(self,ansatze: list[QuantumCircuit],
        operator: BaseOperator,weights: np.ndarray) -> Callable[[np.ndarray], np.ndarray | float]:
        num_parameters = ansatze[0].num_parameters
        eval_count = 0

        def _evaluate_one(ansatz, operator, parameters):
            job = self.estimator.run([(ansatz, operator, parameters.tolist())])
            return job.result()[0]

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count
            parameters = np.reshape(parameters, (-1, num_parameters))

            try:
                with ThreadPoolExecutor(max_workers=len(ansatze)) as executor:
                    futures = [
                        executor.submit(_evaluate_one, ansatz, operator, params)
                        for params in parameters
                        for ansatz in ansatze
                    ]
                    results = [future.result() for future in futures]

            except Exception as exc:
                raise AlgorithmError(
                    "The primitive job to evaluate the SS-VQE energy failed!"
                ) from exc

            n_parameter_sets = len(parameters)
            energies = []

            for state_idx in range(len(ansatze)):
                state_results = results[
                    state_idx::len(ansatze)
                ]

                values = []
                for result in state_results:
                    value = np.asarray(cast(Any, result.data).evs)
                    if not value.shape:
                        value = value.reshape(1)
                    values.append(value)

                energies.append(np.asarray(values).reshape(n_parameter_sets, -1).T)

            # Shape: (number of reference states, number of parameter sets)
            energies = np.asarray(energies).squeeze()
            if energies.ndim == 1:
                energies = energies[:, None]

            cost_values = np.sum(np.asarray(weights)[:, None] * energies, axis=0)

            if self.callback is not None:
                for params, value in zip(parameters, cost_values):
                    eval_count += 1
                    self.callback(eval_count, params, value, results[0].metadata)

            return cost_values[0] if len(cost_values) == 1 else cost_values

        return evaluate_energy    

    def ss_get_evaluate_gradient(
        self,
        ansatze: list[QuantumCircuit],
        operator: BaseOperator,
        weights: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:

        if self.gradient is not None:
            gradient = self.gradient
        elif isinstance(self.estimator, StatevectorEstimator):
            gradient = ReverseEstimatorGradient()
        else:
            gradient = ParamShiftEstimatorGradient(self.estimator)

        def _evaluate_one(grad_eval, ansatz, operator, parameters):
            job = grad_eval.run([ansatz], [operator], [parameters.tolist()])
            return job.result().gradients[0]

        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            try:
                with ThreadPoolExecutor(max_workers=len(ansatze)) as executor:
                    futures = [
                        executor.submit(_evaluate_one, gradient, ansatz, operator, parameters)
                        for ansatz in ansatze
                    ]
                    gradients = [future.result() for future in futures]

            except Exception as exc:
                raise AlgorithmError(
                    "The primitive job to evaluate the SS-VQE gradient failed!"
                ) from exc

            gradients = np.asarray(gradients)
            cost_gradient = np.sum(np.asarray(weights)[:, None] * gradients, axis=0)

            return cost_gradient

        return evaluate_gradient

    def _check_operator_ansatze(self, operator: BaseOperator):
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        if operator.num_qubits != self.ansatze[0].num_qubits:
            try:
                logger.info(
                    "Trying to resize ansatz to match operator on %s qubits.", operator.num_qubits
                )
                cast(Any, self.ansatze[0]).num_qubits = operator.num_qubits
                warnings.warn(
                    "Previously, it was possible to pass to VQE a BlueprintCircuit as an "
                    "ansatz without its number of qubits being set, the algorithm taking care "
                    "of setting it. Since BlueprintCircuits are now  deprecated, and those "
                    "being the only ones that can have their number of qubits set after their "
                    "initialization, this behavior is now also deprecated, and won't be "
                    "supported once the oldest supported Qiskit version is 3.0. As such, users "
                    "that made use of this feature would now need to ensure that the ansatz "
                    "they pass to these algorithms have their number of qubits set and matching "
                    "with that of the operator they wish to run the algorithm on.",
                    category=DeprecationWarning,
                )
            except AttributeError as error:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from error

        if self.ansatze[0].num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

    def _build_ss_vqe_result(
        self,
        ansatze: list[QuantumCircuit],
        optimizer_result: OptimizerResult,
        aux_operators_evaluated: (
            list[ListOrDict[tuple[complex, dict[str, Any]]] | None]
            | None
        ),
        optimizer_time: float,
    ) -> SS_VQEResult:

        if optimizer_result.fun is None:
            raise ValueError("Optimizer did not return an optimal objective value.")
        if optimizer_result.x is None:
            raise ValueError("Optimizer did not return optimal parameters.")
        if optimizer_result.nfev is None:
            raise ValueError("Optimizer did not return the number of function evaluations.")
        if not ansatze:
            raise ValueError("SS-VQE requires at least one ansatz circuit.")
        optimal_point = np.asarray(
            optimizer_result.x, dtype=float
        ).reshape(-1)
        if len(ansatze[0].parameters) != len(optimal_point):
            raise ValueError(
                "The number of optimized parameters does not match "
                "the parameters in the ansatz."
            )
        result = SS_VQEResult()
        result.optimal_circuits = [ansatz.assign_parameters(dict(zip(ansatz.parameters, optimal_point)))
            for ansatz in ansatze]

        result.eigenvalue = complex(optimizer_result.fun)
        result.cost_function_evals = int(optimizer_result.nfev)
        result.optimal_point = optimal_point

        result.optimal_parameters = dict(zip(ansatze[0].parameters, optimal_point))

        result.optimal_value = float(optimizer_result.fun)
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated_by_state = aux_operators_evaluated
        result.optimizer_result = optimizer_result

        return result

class SS_VQEResult(VQEResult):
    def __init__(self) -> None:
        super().__init__()

        self.optimal_circuits: list[QuantumCircuit] = []

        self.aux_operators_evaluated_by_state: (
            list[ListOrDict[tuple[complex, dict[str, Any]]] | None]
            | None
        ) = None
