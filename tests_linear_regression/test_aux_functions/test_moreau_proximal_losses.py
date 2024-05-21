from unittest import main
from numpy.random import randn, rand
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
import linear_regression.aux_functions.moreau_proximal_losses as mpl
import linear_regression.aux_functions.loss_functions as lf
from tests_linear_regression.function_comparison import TestFunctionComparison


class TestProximalHingeLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9

    def test_values(self) -> None:
        def true_proximal_hinge_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.hinge_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                # bounds=(-1e2, 1e2),
                tol=self.tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Hinge_loss, true_proximal_hinge_loss, arg_signatures=("b", "n", "0-1")
        )


class TestProximalLogisticLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9

    def test_values(self) -> None:
        def true_proximal_logistic_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.logistic_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                # bounds=(-1e5, 1e5),
                tol=self.tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Logistic_loss,
            true_proximal_logistic_loss,
            num_points=200,
            tolerance=5e1 * self.tol,
            arg_signatures=("b", "n", "0-1"),
        )


class TestProximalExponentialLoss(TestFunctionComparison):
    def setUp(self) -> None:
        self.tol = 1e-9

    def test_values(self) -> None:
        def true_proximal_exponential_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.exponential_loss(y, z) + 0.5 * (z - omega) ** 2 / V,
                # bounds=(-1e5, 1e5),
                tol=self.tol,
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Exponential_loss,
            true_proximal_exponential_loss,
            num_points=200,
            tolerance=5e1 * self.tol,
            arg_signatures=("b", "n", "0-1"),
        )


if __name__ == "__main__":
    main()
