from unittest import TestCase, main
import numpy as np
from numpy.testing import assert_allclose
import linear_regression.utils.root_finding as rf


class TestBrentRootFinder(TestCase):
    def setUp(self):
        self.tol = 1e-6
        self.max_iter = 100

    def test_function_with_root(self):
        def func(x):
            return np.sin(x) + x / 2

        x0, x1 = 0.1, 0.9
        root = rf.brent_root_finder(func, x0, x1, self.tol, self.tol, self.max_iter, ())
        assert_allclose(func(root), 0.0, atol=self.tol)

    def test_function_without_root(self):
        def func(x):
            return np.sin(x) + x / 2 - 1

        x0, x1 = 0.1, 0.9
        with self.assertRaises(ValueError):
            rf.brent_root_finder(func, x0, x1, self.tol, self.tol, self.max_iter, ())

    def test_max_iter_reached(self):
        def func(x):
            return np.exp(x) - np.sin(x)

        x0, x1 = -10, 10
        with self.assertRaises(RuntimeError):
            rf.brent_root_finder(func, x0, x1, self.tol, self.tol, 10, ())  # max_iter too low

    def test_precision(self):
        # Define a difficult function with a root at x=1.5
        def fun(x):
            return np.sin(10 * x) + np.cos(3 * x) - x**2 + x - 1.5

        # Find the root using rf.brent_root_finder
        x_root = rf.brent_root_finder(fun, 0, 2, xtol=1e-10, rtol=1e-10, max_iter=1000, args=())

        # Assert that the root is close to the actual value
        self.assertAlmostEqual(x_root, 1.5, delta=1e-10)


if __name__ == "__main__":
    main()
