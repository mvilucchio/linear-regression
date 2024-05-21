
from unittest import main
from scipy.optimize import minimize_scalar
from tests_linear_regression.function_comparison import TestFunctionComparison
import linear_regression.aux_functions.likelihood_channel_functions as lcf


class TestFoutL2(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lcf.f_out_L2,
            lambda y, omega, V: (
                minimize_scalar(
                    lambda z: 0.5 * (y - z) ** 2 + 0.5 / V * (z - omega) ** 2,
                    bracket=[-1e20, 1e20],
                    tol=1e-10,
                ).x
                - omega
            )
            / V,
            arg_signatures=("u", "u", "+")
        )


class TestFoutL1(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lcf.f_out_L1,
            lambda y, omega, V: (
                minimize_scalar(
                    lambda z: abs(y - z) + 0.5 / V * (z - omega) ** 2,
                    bracket=[-1e20, 1e20],
                    tol=1e-10,
                ).x
                - omega
            )
            / V,
            arg_signatures=("u", "u", "+")
        )


class TestFoutHuber(TestFunctionComparison):
    def test_values(self):
        def true_huber(x, y, a):
            if abs(x - y) <= a:
                return 0.5 * (x - y) ** 2
            else:
                return a * abs(x - y) - 0.5 * a**2

        as_test = [0.001, 0.01, 0.1, 1, 10, 100]
        for a in as_test:
            self.compare_two_functions(
                lambda y, omega, V : lcf.f_out_Huber(y, omega, V, a),
                lambda y, omega, V: (
                    minimize_scalar(
                        lambda z : true_huber(y, z, a) + 0.5 / V * (z - omega) ** 2,
                        bracket=[-1e20, 1e20],
                        tol=1e-10,
                    ).x
                    - omega
                )
                / V,
                arg_signatures=("u", "u", "+")
            )


if __name__ == "__main__":
    main()
