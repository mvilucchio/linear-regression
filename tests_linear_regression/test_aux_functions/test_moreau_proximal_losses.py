from unittest import main
from numpy.random import randn, rand
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
import linear_regression.aux_functions.moreau_proximal_losses as mpl
import linear_regression.aux_functions.loss_functions as lf
from tests_linear_regression.function_comparison import TestFunctionComparison


class TestProximalHingeLoss(TestFunctionComparison):
    def test_values(self):
        def true_proximal_hinge_loss(y, omega, V):
            return minimize_scalar(
                lambda z: lf.hinge_loss(y, z ) + 0.5 * (z - omega) ** 2 / V, bounds=(-1e2, 1e2)
            )["x"]

        self.compare_two_functions(
            mpl.proximal_Hinge_loss, true_proximal_hinge_loss, arg_signatures=("b", "n", "0-1")
        )


if __name__ == "__main__":
    main()
