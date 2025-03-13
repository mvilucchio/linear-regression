from unittest import main
from numpy.random import randn, rand
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
import linear_regression.aux_functions.moreau_proximals as mpl
import linear_regression.aux_functions.loss_functions as lf
from tests_linear_regression.function_comparison import (
    TestFunctionComparison,
    stencil_derivative_1d,
)


# ---------------------------------------------------------------------------- #
#                            Loss functions proximal                           #
# ---------------------------------------------------------------------------- #
class TestProximalHingeLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9

    def test_values(self) -> None:
        def true_proximal_hinge_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.hinge_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                tol=self.tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Hinge_loss, true_proximal_hinge_loss, arg_signatures=("+/-", "n", "+")
        )


class TestDProximalHingeLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-12
        self.eps_derivative = 5e-4

    def test_values(self) -> None:
        def true_proximal_hinge_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.hinge_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                tol=self.tol,
            )["x"]

        def true_Dω_proximal_Hinge_loss(y, omega, V):
            return stencil_derivative_1d(
                lambda z: true_proximal_hinge_loss(y, z, V), self.eps_derivative, omega
            )

        self.compare_two_functions(
            mpl.Dω_proximal_Hinge_loss,
            true_Dω_proximal_Hinge_loss,
            arg_signatures=("+/-", "n", "+"),
        )


# ------------------------------- logistic loss ------------------------------ #
class TestProximalLogisticLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12

    def test_values(self) -> None:
        def true_proximal_logistic_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.logistic_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                tol=self.small_tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Logistic_loss,
            true_proximal_logistic_loss,
            num_points=200,
            tolerance=5e1 * self.tol,
            arg_signatures=("+/-", "n", "+"),
        )


class TestDProximalLogisticLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-12
        self.eps_derivative = 1e-3

    def test_values(self) -> None:
        def true_proximal_logistic_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.logistic_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                tol=self.tol,
            )["x"]

        def true_Dω_proximal_Logistic_loss(y, omega, V):
            return stencil_derivative_1d(
                lambda z: true_proximal_logistic_loss(y, z, V), self.eps_derivative, omega
            )

        self.compare_two_functions(
            mpl.Dω_proximal_Logistic_loss,
            true_Dω_proximal_Logistic_loss,
            num_points=200,
            arg_signatures=("+/-", "n", "+"),
        )


# ------------------------- adversarial logistic loss ------------------------ #
class TestProximalLogisticAdversarial(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12

    def test_values(self) -> None:
        def true_proximal_logistic_adversarial(y, omega, V, P, eps):
            return minimize_scalar(
                lambda z: lf.logistic_loss(y, z - y * eps * P) + 0.5 * (z - omega) ** 2 / V,
                tol=self.small_tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Logistic_adversarial,
            true_proximal_logistic_adversarial,
            num_points=200,
            tolerance=1e-5,
            arg_signatures=("+/-", "n", "+", "+", "0-1"),
        )


class TestDProximalLogisticAdversarial(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-12
        self.eps_derivative = 1e-3
        self.tol_comparison = 1e-3

    def test_values(self) -> None:
        def true_proximal_logistic_adversarial(y, omega, V, P, eps):
            return minimize_scalar(
                lambda z: lf.logistic_loss(y, z - y * eps * P) + 0.5 * (z - omega) ** 2 / V,
                tol=self.tol,
            )["x"]

        def true_Dω_proximal_Logistic_adversarial(y, omega, V, P, eps):
            return stencil_derivative_1d(
                lambda z: true_proximal_logistic_adversarial(y, z, V, P, eps),
                self.eps_derivative,
                omega,
            )

        self.compare_two_functions(
            mpl.Dω_proximal_Logistic_adversarial,
            true_Dω_proximal_Logistic_adversarial,
            num_points=200,
            tolerance=self.tol_comparison,
            arg_signatures=("+/-", "n", "+", "+", "0-1"),
        )


# ----------------------------- exponential loss ----------------------------- #
class TestProximalExponentialLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12

    def test_values(self) -> None:
        def true_proximal_exponential_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.exponential_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                tol=self.small_tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Exponential_loss,
            true_proximal_exponential_loss,
            num_points=200,
            tolerance=5e1 * self.tol,
            arg_signatures=("+/-", "n", "+"),
        )


# ---------------------------------------------------------------------------- #
#                           Regularisation proximals                           #
# ---------------------------------------------------------------------------- #
class TestProximalSumAbsoluteValues(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12

    def test_values(self) -> None:
        def true_proximal_sum_absolute(gamma, Λ, lambda_p, p, lambda_q, q):
            return minimize_scalar(
                lambda z: lambda_p * abs(z) ** p
                + lambda_q * abs(z) ** q
                + 0.5 * Λ * z**2
                - gamma * z,  # + 0.5 * Λ * (z - gamma / Λ) ** 2,
                tol=self.small_tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_sum_absolute,
            true_proximal_sum_absolute,
            num_points=100,
            tolerance=1e2 * self.tol,
            arg_signatures=(
                "n",
                "+",
                "+",
                (1.0, 2.0, 3.0, 4.0, 5.0),
                "+",
                (1.0, 2.0, 3.0, 4.0, 5.0),
            ),
        )


class TestDProximalSumAbsoluteValues(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-5
        self.small_tol = 1e-12
        self.eps_derivative = 1e-3

    def test_values(self) -> None:
        def true_proximal_sum_absolute(gamma, Λ, lambda_p, p, lambda_q, q):
            return minimize_scalar(
                lambda z: lambda_p * abs(z) ** p
                + lambda_q * abs(z) ** q
                + 0.5 * Λ * z**2
                - gamma * z,  # + 0.5 * Λ * (z - gamma / Λ) ** 2,
                tol=self.small_tol,
            )["x"]

        def true_DƔ_proximal_sum_absolute(gamma, Λ, lambda_p, p, lambda_q, q):
            return stencil_derivative_1d(
                lambda gamma: true_proximal_sum_absolute(gamma, Λ, lambda_p, p, lambda_q, q),
                self.eps_derivative,
                gamma,
            )

        self.compare_two_functions(
            mpl.DƔ_proximal_sum_absolute,
            true_DƔ_proximal_sum_absolute,
            num_points=100,
            tolerance=1e2 * self.tol,
            arg_signatures=(
                "n",
                "+",
                "+",
                (1.0, 2.0, 3.0, 4.0, 5.0),
                "+",
                (1.0, 2.0, 3.0, 4.0, 5.0),
            ),
        )


class TestProximalL1(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12

    def test_values(self) -> None:
        def true_proximal_l1(gamma, Λ, reg_param):
            return minimize_scalar(
                lambda z: reg_param * abs(z) + 0.5 * Λ * z**2 - gamma * z,
                tol=self.small_tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_L1,
            true_proximal_l1,
            num_points=100,
            tolerance=1e2 * self.tol,
            arg_signatures=(
                "n",
                "+",
                "+",
            ),
        )


class TestDProximalL1(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12
        self.eps_derivative = 1e-3

    def test_values(self) -> None:
        def true_proximal_l1(gamma, Λ, reg_param):
            return minimize_scalar(
                lambda z: reg_param * abs(z) + 0.5 * Λ * z**2 - gamma * z,
                tol=self.small_tol,
            )["x"]

        def true_DƔ_proximal_L1(gamma, Λ, reg_param):
            return stencil_derivative_1d(
                lambda gamma: true_proximal_l1(gamma, Λ, reg_param), self.eps_derivative, gamma
            )

        self.compare_two_functions(
            mpl.DƔ_proximal_L1,
            true_DƔ_proximal_L1,
            num_points=100,
            tolerance=1e2 * self.tol,
            arg_signatures=(
                "n",
                "+",
                "+",
            ),
        )


class TestProximalL2(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12

    def test_values(self) -> None:
        def true_proximal_l2(gamma, Λ, reg_param):
            return minimize_scalar(
                lambda z: reg_param * z**2 + 0.5 * Λ * z**2 - gamma * z,
                tol=self.small_tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_L2,
            true_proximal_l2,
            num_points=100,
            tolerance=1e2 * self.tol,
            arg_signatures=(
                "n",
                "+",
                "+",
            ),
        )


class TestDProximalL2(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9
        self.small_tol = 1e-12
        self.eps_derivative = 1e-3

    def test_values(self) -> None:
        def true_proximal_l2(gamma, Λ, reg_param):
            return minimize_scalar(
                lambda z: reg_param * z**2 + 0.5 * Λ * z**2 - gamma * z,
                tol=self.small_tol,
            )["x"]

        def true_DƔ_proximal_L2(gamma, Λ, reg_param):
            return stencil_derivative_1d(
                lambda gamma: true_proximal_l2(gamma, Λ, reg_param), self.eps_derivative, gamma
            )

        self.compare_two_functions(
            mpl.DƔ_proximal_L2,
            true_DƔ_proximal_L2,
            num_points=100,
            tolerance=1e2 * self.tol,
            arg_signatures=(
                "n",
                "+",
                "+",
            ),
        )


if __name__ == "__main__":
    main()
