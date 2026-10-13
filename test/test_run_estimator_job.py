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

"""Test run_estimator_job."""

import unittest
from test import QiskitAlgorithmsTestCase

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_algorithms.run_estimator_job import run_estimator_job


class TestRunEstimatorJob(QiskitAlgorithmsTestCase):
    """Test run_estimator_job."""

    def test_estimators(self):
        """Test the reference estimator and the Qiskit Aer estimator, which holds its default
        precision in its options instead of an attribute."""
        estimators = [StatevectorEstimator()]
        try:
            from qiskit_aer.primitives import EstimatorV2

            estimators.append(EstimatorV2(options={"default_precision": 1e-3}))
        except ImportError:
            pass

        circuit = efficient_su2(2, reps=1)
        values = np.random.default_rng(0).uniform(-1, 1, circuit.num_parameters)
        observables = [SparsePauliOp(["XZ", "YY"], [0.5, -1.0]), SparsePauliOp(["II"], [0.0])]
        expected = [
            Statevector(circuit.assign_parameters(values)).expectation_value(obs).real
            for obs in observables
        ]
        for estimator in estimators:
            with self.subTest(type(estimator).__module__):
                result = run_estimator_job(estimator, [(circuit, observables, values)])[0]
                np.testing.assert_allclose(result.data.evs, expected, atol=1e-2)
                self.assertIn("target_precision", result.metadata)


if __name__ == "__main__":
    unittest.main()
