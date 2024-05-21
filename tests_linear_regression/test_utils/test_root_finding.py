from unittest import TestCase, main
import numpy as np
from numba import njit
from math import sin, cos, exp
from numpy.testing import assert_allclose
import linear_regression.utils.root_finding as rf


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


class TestBrentRootFinder(TestCase):
    def setUp(self):
        self.tol = 1e-10
        self.max_iter = 100
        self.multiple_tols = [1e-8, 1e-10, 1e-12]

    def test_output(self):
        x0, x1 = 0.0, 1.1
        root = rf.brent_root_finder(
            test_fun_1, x0, x1, self.tol, self.tol, self.max_iter, ())
        self.assertIsInstance(root, float)

    def test_interval_with_root(self):
        x0, x1 = 0.1, 1.1
        root = rf.brent_root_finder(
            test_fun_1, x0, x1, self.tol, self.tol, self.max_iter, ())
        assert_allclose(test_fun_1(root), 0.0, atol=self.tol)

        x0, x1 = 0.1, 1.1
        root = rf.brent_root_finder(
            test_fun_2, x0, x1, self.tol, self.tol, self.max_iter, ())
        assert_allclose(test_fun_2(root), 0.0, atol=self.tol)

    def test_interval_without_root(self):
        x0, x1 = 1.1, 1.5
        with self.assertRaises(ValueError):
            rf.brent_root_finder(test_fun_2, x0, x1, self.tol,
                                 self.tol, self.max_iter, ())

    def test_max_iter_reached(self):
        x0, x1 = -100, 100
        with self.assertRaises(RuntimeError):
            rf.brent_root_finder(test_fun_3, x0, x1, self.tol, self.tol, 5, ())

    def test_precision(self):
        x0, x1 = 0.75, 1.25
        for tol in self.multiple_tols:
            root = rf.brent_root_finder(
                test_fun_1, x0, x1, tol, tol, self.max_iter, ())
            assert_allclose(test_fun_1(root), 0.0, atol=tol)


if __name__ == "__main__":
    main()
