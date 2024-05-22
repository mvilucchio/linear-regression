from unittest import TestCase, main
from numpy.random import randn, normal, randint, rand
import linear_regression.aux_functions.loss_functions as lf
from tests_linear_regression.function_comparison import TestFunctionComparison
from autograd import elementwise_grad as egrad
import autograd.numpy as anp


def log_loss_agrad(x, y):
    return anp.log(1 + anp.exp(-y * x))


Dlog_loss_agrad = egrad(log_loss_agrad)
DDlog_loss_agrad = egrad(Dlog_loss_agrad)


def exp_loss_agrad(x, y):
    return anp.exp(-y * x)


Dexp_loss_agrad = egrad(exp_loss_agrad)
DDexp_loss_agrad = egrad(Dexp_loss_agrad)


# ---------------------------------------------------------------------------- #
#                             test function classes                            #
# ---------------------------------------------------------------------------- #
class TestL2Loss(TestFunctionComparison):
    def test_output_shape(self):
        n = randint(1, 101)
        xs, ys = randn(2, n)
        self.assertEqual(lf.l2_loss(xs, ys).shape, (n,))

    def test_values(self):
        self.compare_two_functions(
            lf.l2_loss, lambda x, y: 0.5 * (x - y) ** 2, arg_signatures=("u", "u")
        )


class TestL1Loss(TestFunctionComparison):
    def test_output_shape(self):
        n = randint(1, 101)
        xs, ys = randn(2, n)
        self.assertEqual(lf.l1_loss(xs, ys).shape, (n,))

    def test_values(self):
        self.compare_two_functions(lf.l1_loss, lambda x, y: abs(x - y), arg_signatures=("u", "u"))


class TestHuberLoss(TestFunctionComparison):
    def test_output_shape_fixed_a(self):
        n = randint(1, 101)
        a = rand()
        xs, ys = randn(2, n)
        self.assertEqual(lf.huber_loss(xs, ys, a).shape, (n,))

    def test_output_shape_varaible_a(self):
        n = randint(1, 101)
        xs, ys, aa = randn(3, n)
        self.assertEqual(lf.huber_loss(xs, ys, aa).shape, (n,))

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


class TestHingeLoss(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lf.hinge_loss, lambda x, y: max(0.0, 1.0 - x * y), arg_signatures=("u", "u")
        )


class TestLogisticLoss(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lambda x, y: lf.logistic_loss(y, x), log_loss_agrad, arg_signatures=("u", "b")
        )


class TestDzLogisticLoss(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lambda x, y: lf.Dz_logistic_loss(y, x),
            Dlog_loss_agrad,
            arg_signatures=("u", "b"),
        )


class TestD2zLogisticLoss(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lambda x, y: lf.DDz_logistic_loss(y, x),
            DDlog_loss_agrad,
            arg_signatures=("u", "b"),
        )


class TestExponentialLoss(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lambda x, y: lf.exponential_loss(y, x), exp_loss_agrad, arg_signatures=("u", "b")
        )


class TestDzExponentialLoss(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lambda x, y: lf.Dz_exponential_loss(y, x),
            Dexp_loss_agrad,
            arg_signatures=("u", "b"),
        )


class TestD2zExponentialLoss(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            lambda x, y: lf.DDz_exponential_loss(y, x),
            DDexp_loss_agrad,
            arg_signatures=("u", "b"),
        )


if __name__ == "__main__":
    main()
