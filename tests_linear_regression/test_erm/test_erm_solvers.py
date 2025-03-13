from unittest import main
from numpy.random import randn, rand
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from jax import jit
import jax.numpy as jnp
import linear_regression.erm.erm_solvers as es
from tests_linear_regression.function_comparison import TestFunctionComparison


class TestLossLogistcAdv(TestFunctionComparison):
    def setUp(self) -> None:
        self.ys = randn(100)
        self.xs = randn(100, 10)
        self.eps_t = 0.1
        self.eps_g = 0.1
        self.reg_param = 1.0
        self.pstar = 1.0

    def test_shape(self):
        pass


# ---------------------------------------------------------------------------- #
#                          Projected Gradient Descent                          #
# ---------------------------------------------------------------------------- #


@jit
def non_lin(x):
    return jnp.tanh(x)


class TestPGDNonLinearRandomFeatures(TestFunctionComparison):
    def setUp(self) -> None:
        self.ys = randn(100)
        self.xs = randn(100, 10)
        self.eps_t = 0.1
        self.eps_g = 0.1
        self.reg_param = 1.0
        self.pstar = 1.0

    def test_shape_single(self):
        es.linear


if __name__ == "__main__":
    main()
