# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for an Imaginary McLachlan's Variational Principle."""

from __future__ import annotations

import warnings

from collections.abc import Sequence

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .imaginary_variational_principle import ImaginaryVariationalPrinciple

from ....exceptions import AlgorithmError
from ....gradients import (
    BaseEstimatorGradient,
    BaseQGT,
    DerivativeType,
    LinCombQGT,
    LinCombEstimatorGradient,
)
from ....run_estimator_job import run_estimator_job
from ....utils.hermitian_decomposition import hermitian_decomposition


class ImaginaryMcLachlanPrinciple(ImaginaryVariationalPrinciple):
    r"""Class for an Imaginary McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Wick-rotated Schrödinger equation with a quantum state given as a
    parametrized trial state. The principle leads to a system of linear equations handled by a
    linear solver. The imaginary variant means that we consider imaginary time dynamics.

    The principle also supports non-Hermitian Hamiltonians, such as the transcorrelated
    Hamiltonians of electronic structure theory. For these, the normalized imaginary time
    evolution :math:`e^{-H\tau}|\psi\rangle` converges to the right eigenvector of :math:`H` with
    the lowest real part of the eigenvalue [1, 2]. Writing :math:`H = A + iB` with Hermitian
    :math:`A` and :math:`B`, the right-hand side of the linear system becomes

    .. math::

        -\mathrm{Re}\left[\langle\partial_i\psi|H|\psi\rangle
        - \langle\partial_i\psi|\psi\rangle\langle\psi|H|\psi\rangle\right]
        = -\frac{1}{2}\left[2\mathrm{Re}\langle\psi|A|\partial_i\psi\rangle
        + 2\mathrm{Im}\langle\psi|B - \langle B\rangle|\partial_i\psi\rangle\right],

    so only expectation values and gradients of Hermitian observables are required. A Hamiltonian
    is treated as non-Hermitian if its simplified Pauli representation has complex coefficients
    (see :func:`~qiskit_algorithms.utils.hermitian_decomposition`). This requires a gradient
    that supports :attr:`.DerivativeType.IMAG`, such as :class:`.LinCombEstimatorGradient` (the
    default) or :class:`.ReverseEstimatorGradient`. The term containing
    :math:`\langle B\rangle` accounts for the global phase and is only included if the QGT is
    phase-fixed (the default).

    Note that :math:`\langle\psi|H|\psi\rangle` is not variational for non-Hermitian
    Hamiltonians and is in general complex.

    **References:**

    [1] S. McArdle and D. P. Tew, Improving the accuracy of quantum computational chemistry
    using the transcorrelated method, `arXiv:2006.11181 <https://arxiv.org/abs/2006.11181>`_.

    [2] I. O. Sokolov, W. Dobrautz, H. Luo, A. Alavi and I. Tavernelli, Orders of magnitude
    increased accuracy for quantum many-body problems on quantum computers via an exact
    transcorrelated method, `Phys. Rev. Research 5, 023174 (2023)
    <https://doi.org/10.1103/PhysRevResearch.5.023174>`_.
    """

    def __init__(
        self,
        qgt: BaseQGT | None = None,
        gradient: BaseEstimatorGradient | None = None,
    ) -> None:
        """
        Args:
            qgt: Instance of a the GQT class used to compute the QFI.
                If ``None`` provided, ``LinCombQGT`` is used.
            gradient: Instance of a class used to compute the state gradient.
                If ``None`` provided, ``LinCombEstimatorGradient`` is used.

        Raises:
            AlgorithmError: If the gradient instance does not contain an estimator.
        """

        self._validate_grad_settings(gradient)

        if gradient is not None:
            try:
                estimator = gradient._estimator
            except Exception as exc:
                raise AlgorithmError(
                    "The provided gradient instance does not contain an estimator primitive."
                ) from exc
        else:
            estimator = StatevectorEstimator()
            gradient = LinCombEstimatorGradient(estimator)

        if qgt is None:
            qgt = LinCombQGT(estimator)

        super().__init__(qgt, gradient)
        self._transpiled_ansatz: dict[int, tuple[QuantumCircuit, QuantumCircuit]] = {}

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence[Parameter] | None = None,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of parameters to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
                If ``None`` given, gradients w.r.t. all parameters will be computed.

        Returns:
            An evolution gradient.

        Raises:
            AlgorithmError: If a gradient job fails or if a non-Hermitian Hamiltonian is used
                with a gradient that does not support the imaginary part of the gradient.
        """
        hermitian_part, anti_hermitian_part = hermitian_decomposition(hamiltonian)

        if anti_hermitian_part is None:
            evolution_grad_lse_rhs = self._run_gradient(
                hamiltonian, ansatz, param_values, gradient_params, DerivativeType.REAL
            )
            return -0.5 * evolution_grad_lse_rhs

        # non-Hermitian Hamiltonian H = A + iB with Hermitian A and B
        if getattr(self.qgt, "_phase_fix", True):
            circuit, observable = ansatz, anti_hermitian_part
            transpiler = getattr(self.gradient, "_transpiler", None)
            if transpiler is not None:
                # the estimator may require circuits compiled for its backend, like the gradient
                circuit = self._transpile(ansatz, transpiler)
                if circuit.layout is not None:
                    observable = observable.apply_layout(circuit.layout)
            try:
                expectation_b = run_estimator_job(
                    self.gradient._estimator, [(circuit, observable, param_values)]
                )[0].data.evs
            except Exception as exc:
                raise AlgorithmError("The estimator primitive job failed!") from exc
            anti_hermitian_part = (
                anti_hermitian_part
                - SparsePauliOp("I" * anti_hermitian_part.num_qubits, float(expectation_b))
            ).simplify()

        evolution_grad_lse_rhs = self._run_gradient(
            hermitian_part, ansatz, param_values, gradient_params, DerivativeType.REAL
        ) + self._run_gradient(
            anti_hermitian_part, ansatz, param_values, gradient_params, DerivativeType.IMAG
        )
        return -0.5 * evolution_grad_lse_rhs

    def _transpile(self, ansatz: QuantumCircuit, transpiler) -> QuantumCircuit:
        """Transpiles the ansatz with the transpiler of the gradient, caching the result."""
        key = id(ansatz)
        cached = self._transpiled_ansatz.get(key)
        if cached is None or cached[0] is not ansatz:
            options = getattr(self.gradient, "_transpiler_options", None) or {}
            cached = (ansatz, transpiler.run(ansatz, **options))
            self._transpiled_ansatz[key] = cached
        return cached[1]

    def _run_gradient(
        self,
        observable: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence[Parameter] | None,
        derivative_type: DerivativeType,
    ) -> np.ndarray:
        """Runs the gradient for the given derivative type, which is restored afterwards."""
        num_params = len(ansatz.parameters) if gradient_params is None else len(gradient_params)
        if isinstance(observable, SparsePauliOp) and not observable.coeffs.any():
            return np.zeros(num_params)

        original_type = getattr(self.gradient, "derivative_type", DerivativeType.REAL)
        if derivative_type != original_type:
            try:
                # the gradients supporting the imaginary part define a setter
                self.gradient.derivative_type = derivative_type  # type: ignore[misc]
            except AttributeError as exc:
                raise AlgorithmError(
                    "Non-Hermitian Hamiltonians require a gradient that supports "
                    "DerivativeType.IMAG, e.g., LinCombEstimatorGradient or "
                    f"ReverseEstimatorGradient. The provided gradient of type "
                    f"{type(self.gradient).__name__} does not."
                ) from exc
        try:
            return (
                self.gradient.run([ansatz], [observable], [param_values], [gradient_params])
                .result()
                .gradients[0]
            )
        except Exception as exc:
            raise AlgorithmError("The gradient primitive job failed!") from exc
        finally:
            if derivative_type != original_type:
                self.gradient.derivative_type = original_type  # type: ignore[misc]

    @staticmethod
    def _validate_grad_settings(gradient):
        if (
            gradient is not None
            and hasattr(gradient, "_derivative_type")
            and gradient._derivative_type != DerivativeType.REAL
        ):
            warnings.warn(
                "A gradient instance with a setting for calculating imaginary part of "
                "the gradient was provided. This variational principle requires the"
                "real part. The setting to real was changed automatically."
            )
            gradient._derivative_type = DerivativeType.REAL
