from unittest import TestCase, main
from numpy.random import randn, normal
import linear_regression.aux_functions.loss_functions as lf
from tests_linear_regression.function_comparison import TestFunctionComparison


class TestFoutL2(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lf.l2_loss, lambda x, y: 0.5 * (x - y) ** 2, arg_signatures=("u", "u")
        )


class TestFoutL1(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(lf.l1_loss, lambda x, y: abs(x - y), arg_signatures=("u", "u"))


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
                lambda x, y: lf.huber_loss(x, y, a),
                lambda x, y: true_huber(x, y, a),
                arg_signatures=("u", "u"),
            )


if __name__ == "__main__":
    main()
