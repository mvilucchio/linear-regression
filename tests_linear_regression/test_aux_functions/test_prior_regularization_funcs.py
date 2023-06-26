from unittest import main
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import numpy as np
import linear_regression.aux_functions.prior_regularization_funcs as prf
from tests_linear_regression.function_comparison import TestFunctionComparison


# class TestZwBayesGaussianPrior(TestFunctionComparison):
#     def test_values(self):
#         self.compare_two_functions(
#             prf.Z_w_Bayes_gaussian_prior,
#             lambda gamma, Lambda, mu, sigma: quad(
#                 lambda z: np.exp(
#                     -0.5 * (z - mu) ** 2 / sigma**2 - 0.5 * Lambda * z**2 + gamma * z
#                 )
#                 / np.sqrt(2 * np.pi * sigma),
#                 -np.inf, np.inf
#             )[0],
#             arg_signatures=("u", "+", "u", "+"),
#         )


class TestFwL2Regularization(TestFunctionComparison):
    def test_values(self):
        reg_params_test = [0.001, 0.01, 0.1, 1, 10, 100]
        for reg_param in reg_params_test:
            self.compare_two_functions(
                lambda gamma, Lambda: prf.f_w_L2_regularization(gamma, Lambda, reg_param),
                lambda gamma, Lambda: (
                    minimize_scalar(
                        lambda z: 0.5 * reg_param * z**2
                        + 0.5 * Lambda * (z - gamma / Lambda) ** 2,
                        bracket=[-1e20, 1e20],
                        tol=1e-10,
                    ).x
                ),
                arg_signatures=("u", "+"),
            )


class TestFwL1Regularization(TestFunctionComparison):
    def test_values(self):
        reg_params_test = [0.001, 0.01, 0.1, 1, 10, 100]
        for reg_param in reg_params_test:
            self.compare_two_functions(
                lambda gamma, Lambda: prf.f_w_L1_regularization(gamma, Lambda, reg_param),
                lambda gamma, Lambda: (
                    minimize_scalar(
                        lambda z: reg_param * abs(z) + 0.5 * Lambda * (z - gamma / Lambda) ** 2,
                        bracket=[-1e20, 1e20],
                        tol=1e-10,
                    ).x
                ),
                arg_signatures=("u", "+"),
            )


if __name__ == "__main__":
    main()
