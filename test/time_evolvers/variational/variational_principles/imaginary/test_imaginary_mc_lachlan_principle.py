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

"""Test imaginary McLachlan's variational principle."""

import unittest

import numpy as np
from ddt import ddt, data
from qiskit import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, efficient_su2
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms import AlgorithmError
from qiskit_algorithms.gradients import (
    DerivativeType,
    LinCombEstimatorGradient,
    LinCombQGT,
    ParamShiftEstimatorGradient,
    ReverseEstimatorGradient,
    ReverseQGT,
)
from qiskit_algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
# fmt: off
# pylint: disable=wrong-import-order
from test.time_evolvers.variational.variational_principles.expected_results.\
    test_imaginary_mc_lachlan_variational_principle_expected1 import expected_bound_metric_tensor_1
# pylint: enable=wrong-import-order
# fmt: on
from test import QiskitAlgorithmsTestCase  # pylint: disable=wrong-import-order


def _non_hermitian_operator(seed: int = 7) -> np.ndarray:
    """Returns a real, non-Hermitian 2-qubit matrix with a real spectrum."""
    rng = np.random.default_rng(seed)
    similarity = np.eye(4) + 0.4 * rng.normal(size=(4, 4))
    return similarity @ np.diag([-1.3, -0.4, 0.5, 1.1]) @ np.linalg.inv(similarity)


def _exact_evolution_gradient(matrix, ansatz, param_values, phase_fix=True, eps=1e-6):
    """The McLachlan right-hand side -Re[<d_i psi|H|psi> - <d_i psi|psi><H>] by finite
    differences."""

    def state(values):
        return Statevector(ansatz.assign_parameters(values)).data

    psi = state(param_values)
    energy = psi.conj() @ matrix @ psi
    rhs = []
    for i in range(len(param_values)):
        shift = np.zeros(len(param_values))
        shift[i] = eps
        d_psi = (state(param_values + shift) - state(param_values - shift)) / (2 * eps)
        value = d_psi.conj() @ matrix @ psi
        if phase_fix:
            value -= (d_psi.conj() @ psi) * energy
        rhs.append(-np.real(value))
    return np.array(rhs)


@ddt
class TestImaginaryMcLachlanPrinciple(QiskitAlgorithmsTestCase):
    """Test imaginary McLachlan's variational principle."""

    def test_calc_metric_tensor(self):
        """Test calculating a metric tensor."""
        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        d = 2  # pylint: disable=invalid-name
        ansatz = efficient_su2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = list(ansatz.parameters)
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = ImaginaryMcLachlanPrinciple()

        bound_metric_tensor = var_principle.metric_tensor(ansatz, list(param_dict.values()))

        np.testing.assert_almost_equal(bound_metric_tensor, expected_bound_metric_tensor_1)

    def test_calc_calc_evolution_gradient(self):
        """Test calculating evolution gradient."""
        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        d = 2  # pylint: disable=invalid-name
        ansatz = efficient_su2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = list(ansatz.parameters)
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = ImaginaryMcLachlanPrinciple()

        bound_evolution_gradient = var_principle.evolution_gradient(
            observable, ansatz, list(param_dict.values()), parameters
        )

        expected_evolution_gradient = [
            (0.19308934095957098 - 1.4e-17j),
            (0.007027674650099142 - 0j),
            (0.03192524520091862 - 0j),
            (-0.06810314606309673 - 1e-18j),
            (0.07590371669521798 - 7e-18j),
            (0.11891968269385343 + 1.5e-18j),
            (-0.0012030273438232639 + 0j),
            (-0.049885258804562266 + 1.8500000000000002e-17j),
            (-0.20178860797540302 - 5e-19j),
            (-0.0052269232310933195 + 1e-18j),
            (0.022892905637005266 - 3e-18j),
            (-0.022892905637005294 + 3.5e-18j),
        ]

        np.testing.assert_almost_equal(bound_evolution_gradient, expected_evolution_gradient)

    def test_gradient_setting(self):
        """Test reactions to wrong gradient settings.."""
        estimator = StatevectorEstimator(seed=123)
        gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)

        with self.assertWarns(Warning):
            var_principle = ImaginaryMcLachlanPrinciple(gradient=gradient)

        np.testing.assert_equal(var_principle.gradient._derivative_type, DerivativeType.REAL)

    @data("lin_comb", "reverse")
    def test_non_hermitian_evolution_gradient(self, method):
        """Test the evolution gradient of a non-Hermitian Hamiltonian against the exact one."""
        matrix = _non_hermitian_operator()
        hamiltonian = SparsePauliOp.from_operator(Operator(matrix))
        self.assertTrue(np.iscomplexobj(np.real_if_close(hamiltonian.simplify().coeffs)))

        ansatz = efficient_su2(2, reps=1)
        param_values = np.random.default_rng(3).uniform(-1, 1, ansatz.num_parameters)

        for phase_fix in (True, False):
            with self.subTest(phase_fix=phase_fix):
                if method == "lin_comb":
                    estimator = StatevectorEstimator()
                    var_principle = ImaginaryMcLachlanPrinciple(
                        LinCombQGT(estimator, phase_fix=phase_fix),
                        LinCombEstimatorGradient(estimator),
                    )
                else:
                    var_principle = ImaginaryMcLachlanPrinciple(
                        ReverseQGT(phase_fix=phase_fix), ReverseEstimatorGradient()
                    )
                evolution_gradient = var_principle.evolution_gradient(
                    hamiltonian, ansatz, param_values
                )
                expected = _exact_evolution_gradient(matrix, ansatz, param_values, phase_fix)
                np.testing.assert_allclose(evolution_gradient, expected, atol=1e-7)
                # the derivative type of the gradient is restored
                self.assertEqual(var_principle.gradient.derivative_type, DerivativeType.REAL)

    def test_non_hermitian_with_transpiler(self):
        """Test the evolution gradient of a non-Hermitian Hamiltonian when the circuits are
        transpiled for a backend with a non-trivial layout. Note that the transpiler may not fix
        the layout since the gradient circuits contain an additional ancilla qubit."""
        try:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator
        except ImportError:
            self.skipTest("qiskit-aer is not available")

        matrix = _non_hermitian_operator()
        hamiltonian = SparsePauliOp.from_operator(Operator(matrix))
        # a PauliEvolutionGate cannot be executed by the Aer estimator without transpilation
        theta = Parameter("theta")
        ansatz = efficient_su2(2, reps=1)
        ansatz.append(PauliEvolutionGate(SparsePauliOp(["XY", "YX"], [0.5, -0.5]), theta), [0, 1])
        param_values = np.random.default_rng(4).uniform(-1, 1, ansatz.num_parameters)

        backend = GenericBackendV2(num_qubits=4, coupling_map=[[0, 3], [3, 2], [2, 1]], seed=3)
        pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=1, seed_transpiler=5
        )
        # the circuits are not placed on the physical qubits [0, 1]
        self.assertNotEqual(pass_manager.run(ansatz).layout.final_index_layout(), [0, 1])

        estimator = AerEstimator()
        var_principle = ImaginaryMcLachlanPrinciple(
            LinCombQGT(estimator, transpiler=pass_manager),
            LinCombEstimatorGradient(estimator, transpiler=pass_manager),
        )
        evolution_gradient = var_principle.evolution_gradient(hamiltonian, ansatz, param_values)
        expected = _exact_evolution_gradient(matrix, ansatz, param_values)
        np.testing.assert_allclose(evolution_gradient, expected, atol=1e-7)

    def test_non_hermitian_requires_imaginary_gradient(self):
        """Test that a gradient without support for the imaginary part raises for non-Hermitian
        Hamiltonians."""
        hamiltonian = SparsePauliOp.from_operator(Operator(_non_hermitian_operator()))
        ansatz = efficient_su2(2, reps=1)
        var_principle = ImaginaryMcLachlanPrinciple(
            gradient=ParamShiftEstimatorGradient(StatevectorEstimator())
        )
        with self.assertRaises(AlgorithmError):
            var_principle.evolution_gradient(hamiltonian, ansatz, np.zeros(ansatz.num_parameters))


if __name__ == "__main__":
    unittest.main()
