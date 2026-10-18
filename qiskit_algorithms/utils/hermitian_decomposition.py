# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Decomposition of (possibly non-Hermitian) operators into Hermitian parts."""

from __future__ import annotations

import numpy as np

from qiskit.quantum_info import Operator, SparseObservable, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


def hermitian_decomposition(
    operator: BaseOperator,
) -> tuple[BaseOperator, SparsePauliOp | None]:
    r"""Splits an operator into its Hermitian and anti-Hermitian parts.

    Every operator can be written as :math:`H = A + iB` with the two Hermitian operators
    :math:`A = (H + H^\dagger)/2` and :math:`B = (H - H^\dagger)/(2i)`. Since Pauli strings are
    Hermitian, a simplified :class:`~qiskit.quantum_info.SparsePauliOp`
    :math:`H = \sum_k c_k P_k` is Hermitian if and only if all of its coefficients :math:`c_k` are
    real, and :math:`A` and :math:`B` are given by the real and imaginary parts of the
    coefficients, respectively.

    The imaginary parts are considered to vanish if they would be accepted by the estimator
    primitives (see :class:`~qiskit.primitives.containers.ObservablesArray`), i.e. if
    ``numpy.real_if_close`` would discard them.

    Args:
        operator: The operator to decompose.

    Returns:
        A tuple ``(hermitian_part, anti_hermitian_part)``. If ``operator`` is Hermitian (or of a
        type that cannot be inspected), it is returned unchanged as ``hermitian_part`` and
        ``anti_hermitian_part`` is ``None``. Otherwise, both parts are returned as
        :class:`~qiskit.quantum_info.SparsePauliOp` such that ``operator = hermitian_part +
        1j * anti_hermitian_part``.
    """
    pauli_op = _to_sparse_pauli_op(operator)
    if pauli_op is None:
        return operator, None

    pauli_op = pauli_op.simplify()
    coeffs = np.real_if_close(pauli_op.coeffs)
    if not np.iscomplexobj(coeffs):
        return operator, None

    paulis = pauli_op.paulis
    hermitian_part = SparsePauliOp(paulis, coeffs.real).simplify()
    anti_hermitian_part = SparsePauliOp(paulis, coeffs.imag).simplify()
    return hermitian_part, anti_hermitian_part


def is_hermitian_operator(operator: BaseOperator) -> bool:
    """Checks whether an operator is Hermitian.

    See :func:`hermitian_decomposition` for the criterion that is used.

    Args:
        operator: The operator to check.

    Returns:
        Whether the operator is Hermitian. Operators of a type that cannot be inspected are
        assumed to be Hermitian.
    """
    return hermitian_decomposition(operator)[1] is None


def _to_sparse_pauli_op(operator: BaseOperator) -> SparsePauliOp | None:
    """Converts an operator to a numerical SparsePauliOp, if possible."""
    if isinstance(operator, SparsePauliOp):
        if operator.coeffs.dtype == object:
            # e.g. a time-dependent Hamiltonian; only bound parameters can be inspected
            try:
                return SparsePauliOp(operator.paulis, np.asarray(operator.coeffs, dtype=complex))
            except TypeError:
                return None
        return operator
    if isinstance(operator, SparseObservable):
        return SparsePauliOp.from_sparse_observable(operator)
    if isinstance(operator, Operator):
        return SparsePauliOp.from_operator(operator)
    return None
