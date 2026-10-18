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

"""Test the evaluator of observables."""

import unittest
from test import QiskitAlgorithmsTestCase

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_algorithms.observables_evaluator import estimate_observables


class TestObservablesEvaluator(QiskitAlgorithmsTestCase):
    """Test the evaluator of observables."""

    def test_non_hermitian(self):
        """Test that non-Hermitian observables yield complex expectation values."""
        ansatz = efficient_su2(2, reps=1)
        parameter_values = np.random.default_rng(2).uniform(-1, 1, ansatz.num_parameters)
        state = Statevector(ansatz.assign_parameters(parameter_values))
        observables = {
            "hermitian": SparsePauliOp(["XY", "ZI"], [0.5, -1.0]),
            "non_hermitian": SparsePauliOp(["XY", "ZI", "YY"], [0.5, 1j, 0.3 - 0.2j]),
            "anti_hermitian": SparsePauliOp(["XZ"], [2j]),
            "zero": SparsePauliOp(["II"], [0.0]),
        }
        result = estimate_observables(StatevectorEstimator(), ansatz, observables, parameter_values)
        for name, observable in observables.items():
            with self.subTest(name):
                value = result[name][0]
                expected = state.expectation_value(observable)
                np.testing.assert_allclose(value, expected, atol=1e-12)
        self.assertIsInstance(result["hermitian"][0], float)

    def test_only_zero_observables(self):
        """Test that zero observables are evaluated without running the estimator."""

        class FailingEstimator(StatevectorEstimator):
            """An estimator which must not be run."""

            def run(self, pubs, *, precision=None):
                raise AssertionError("the estimator was run")

        ansatz = efficient_su2(2, reps=1)
        values = np.zeros(ansatz.num_parameters)
        zero = SparsePauliOp(["XX", "ZZ"], [0.0, 0.0])
        results = estimate_observables(FailingEstimator(), ansatz, [zero, zero], values)
        self.assertEqual([value for value, _ in results], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
