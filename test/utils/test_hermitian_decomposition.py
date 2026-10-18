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

"""Test the decomposition of operators into Hermitian parts."""

import unittest
from test import QiskitAlgorithmsTestCase

import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparseObservable, SparsePauliOp

from qiskit_algorithms.utils import hermitian_decomposition, is_hermitian_operator


class TestHermitianDecomposition(QiskitAlgorithmsTestCase):
    """Test the decomposition of operators into Hermitian parts."""

    def test_hermitian(self):
        """Test that Hermitian operators are returned unchanged."""
        operator = SparsePauliOp(["XY", "ZZ", "YI"], [0.5, -1.0, 0.25])
        hermitian_part, anti_hermitian_part = hermitian_decomposition(operator)
        self.assertIs(hermitian_part, operator)
        self.assertIsNone(anti_hermitian_part)
        self.assertTrue(is_hermitian_operator(operator))

    def test_numerical_noise(self):
        """Test that imaginary parts accepted by the estimator primitives are ignored."""
        operator = SparsePauliOp(["XY", "ZZ"], [0.5 + 1e-17j, -1.0])
        self.assertTrue(is_hermitian_operator(operator))

    def test_non_hermitian(self):
        """Test the decomposition of a non-Hermitian operator."""
        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        for operator in (
            SparsePauliOp.from_operator(Operator(matrix)),
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_operator(Operator(matrix))),
            Operator(matrix),
        ):
            with self.subTest(type(operator).__name__):
                self.assertFalse(is_hermitian_operator(operator))
                hermitian_part, anti_hermitian_part = hermitian_decomposition(operator)
                np.testing.assert_allclose(
                    hermitian_part.to_matrix(), (matrix + matrix.conj().T) / 2, atol=1e-12
                )
                np.testing.assert_allclose(
                    anti_hermitian_part.to_matrix(), (matrix - matrix.conj().T) / 2j, atol=1e-12
                )

    def test_parametrized(self):
        """Test operators with parameters."""
        time = Parameter("t")
        operator = SparsePauliOp(["XY", "ZZ"], [time, 1j])
        self.assertTrue(is_hermitian_operator(operator))
        self.assertFalse(is_hermitian_operator(operator.assign_parameters([0.5])))


if __name__ == "__main__":
    unittest.main()
