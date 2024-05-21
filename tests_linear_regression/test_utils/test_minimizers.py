from unittest import TestCase, main
import numpy as np
from numba import njit
from math import sin, cos, exp
from numpy.testing import assert_allclose
import linear_regression.utils.minimizers as mm


@njit
def test_fun_1(x):
    return x**2 - 1


@njit
def test_fun_2(x):
    return sin(x) + x / 2 - 1


@njit
def test_fun_3(x):
    return exp(x) - sin(x)


@njit
def test_fun_4(x):
    return sin(10 * x) + cos(3 * x) - x**2 + x - 1.5


class TestBrentMinimizer(TestCase):
    def setUp(self):
        self.tol = 1e-8
        self.max_iter = 300
        self.multiple_tols = [1e-8, 1e-10, 1e-12]

    def test_output(self):
        x0, x1 = -1.1, 1.1
        out = mm.brent_minimize_scalar(test_fun_1, x0, x1, self.tol, self.tol, self.max_iter, ())
        self.assertIsInstance(out, tuple)
        xstar, fxstar = out
        self.assertIsInstance(xstar, float)
        self.assertIsInstance(fxstar, float)

    def test_minimisation(self):
        x0, x1 = -1.1, 1.1
        xstar, fxstar = mm.brent_minimize_scalar(
            test_fun_1, x0, x1, self.tol, self.tol, self.max_iter, ()
        )

        self.assertAlmostEqual(xstar, 0.0, delta=self.tol)
        self.assertAlmostEqual(fxstar, -1.0, delta=self.tol)


class TestBraket(TestCase):
    def test_output(self):
        pass
        # raise NotImplementedError


if __name__ == "__main__":
    main()
