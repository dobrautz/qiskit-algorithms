# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evaluator of observables for algorithms."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .exceptions import AlgorithmError
from .custom_types import ListOrDict
from .utils.hermitian_decomposition import hermitian_decomposition


# TODO: make estimate_observables accept EstimatorPubLike inputs
def estimate_observables(
    estimator: BaseEstimatorV2,
    quantum_state: QuantumCircuit,
    observables: ListOrDict[BaseOperator],
    parameter_values: Sequence[float] | None = None,
    threshold: float = 1e-12,
) -> ListOrDict[tuple[float, dict[str, Any]]]:
    """
    Accepts a sequence of operators and calculates their expectation values - means
    and metadata. They are calculated with respect to a quantum state provided. A user
    can optionally provide a threshold value which filters mean values falling below the threshold.

    Non-Hermitian operators, i.e. operators whose simplified Pauli representation has complex
    coefficients, are supported by evaluating their Hermitian and anti-Hermitian parts separately
    (see :func:`~qiskit_algorithms.utils.hermitian_decomposition`). Their expectation values are
    complex.

    Args:
        estimator: An estimator primitive used for calculations.
        quantum_state: A (parameterized) quantum circuit preparing a quantum state that expectation
            values are computed against. It is expected to be an ISA circuit.
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.
        parameter_values: Optional list of parameters values to evaluate the quantum circuit on.
        threshold: A threshold value that defines which mean values should be neglected (helpful for
            ignoring numerical instabilities close to 0).

    Returns:
        A list or a dictionary of tuples (mean, metadata).

    Raises:
        AlgorithmError: If a primitive job is not successful.
    """

    if isinstance(observables, dict):
        observables_list = list(observables.values())
    else:
        observables_list = observables

    if len(observables_list) > 0:
        # split non-Hermitian observables into their Hermitian and anti-Hermitian parts; the
        # latter are appended after all Hermitian parts
        hermitian_parts: list[BaseOperator] = []
        anti_hermitian_parts: list[BaseOperator] = []
        anti_hermitian_positions: list[int | None] = []
        for observable in observables_list:
            hermitian_part, anti_hermitian_part = hermitian_decomposition(observable)
            hermitian_parts.append(hermitian_part)
            if anti_hermitian_part is None:
                anti_hermitian_positions.append(None)
            else:
                anti_hermitian_positions.append(len(observables_list) + len(anti_hermitian_parts))
                anti_hermitian_parts.append(anti_hermitian_part)
        all_observables = hermitian_parts + anti_hermitian_parts

        purged_observables, mask = _handle_zero_ops(all_observables)
        parameter_values_: Sequence[float] | None = parameter_values
        if purged_observables:
            try:
                estimator_job = estimator.run(
                    [(quantum_state, purged_observables, parameter_values_)]
                )
                estimator_result = estimator_job.result()[0]
                expectation_values = estimator_result.data.evs
            except Exception as exc:
                raise AlgorithmError("The primitive job failed!") from exc
            metadata = estimator_result.metadata
        else:
            expectation_values = np.array([])
            metadata = {}

        # Discard values below threshold
        observables_means = expectation_values * (np.abs(expectation_values) > threshold)
        # Adding back the values from the nil observables
        split_means = []
        index = 0
        for mask_value in mask:
            # Non-zero observable
            if mask_value:
                split_means.append(observables_means[index])
                index += 1
            else:
                split_means.append(0.0)
        # recombine the Hermitian and anti-Hermitian parts; only the expectation values of
        # non-Hermitian observables are complex
        all_means = [
            (
                np.float64(np.real(split_means[i]))
                if position is None
                else np.complex128(split_means[i] + 1j * split_means[position])
            )
            for i, position in enumerate(anti_hermitian_positions)
        ]
        # zip means and metadata into tuples
        observables_results = list(zip(all_means, [metadata] * len(all_means)))
    else:
        observables_results = []

    return _prepare_result(observables_results, observables)


def _handle_zero_ops(
    observables_list: list[BaseOperator],
) -> tuple[list[BaseOperator], list[bool]]:
    """Returns the initial list of operators with the nil ones having been removed. Furthermore, returns
    a mask indicating which operators were kept."""
    mask: list[bool] = []
    purged_list: list[BaseOperator] = []

    if observables_list:
        for observable in observables_list:
            if observable == 0 or (
                isinstance(observable, SparsePauliOp) and not observable.simplify().coeffs.any()
            ):
                mask.append(False)
            else:
                mask.append(True)
                purged_list.append(observable)
    return purged_list, mask


def _prepare_result(
    observables_results: list[tuple[float, dict]],
    observables: ListOrDict[BaseOperator],
) -> ListOrDict[tuple[float, dict[str, Any]]]:
    """
    Prepares a list of tuples of eigenvalues and metadata tuples from
    ``observables_results`` and ``observables``.

    Args:
        observables_results: A list of tuples (mean, metadata).
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.

    Returns:
        A list or a dictionary of tuples (mean, metadata).
    """

    observables_eigenvalues: ListOrDict[tuple[float, dict]]

    if isinstance(observables, list):
        observables_eigenvalues = []
        for value in observables_results:
            observables_eigenvalues.append(value)

    else:
        observables_eigenvalues = {}
        for key, value in zip(observables.keys(), observables_results):
            observables_eigenvalues[key] = value

    return observables_eigenvalues
