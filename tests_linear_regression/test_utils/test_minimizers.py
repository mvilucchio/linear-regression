from unittest import TestCase, main
import numpy as np
from numba import njit
from math import sin, cos, exp
from numpy.testing import assert_allclose
import linear_regression.utils.minimizers as mm
import linear_regression.aux_functions.moreau_proximals as mpl


@njit
def test_fun_1(x):
    return x**2 - 1


@njit
def test_fun_2(x):
    return 10 * sin(x) + x**2 + x / 2 - 1


@njit
def test_fun_3(x):
    return exp(x) - sin(x)


@njit
def test_fun_4(x):
    return sin(10 * x) + cos(3 * x) - x**2 + x - 1.5


class TestBraket(TestCase):
    def setUp(self):
        self.grow_limit = 110.0
        self.max_iter = 100

    def test_output(self):
        out = mm.bracket(test_fun_1, -1.0, 1.0, (), self.grow_limit, self.max_iter)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 7)
        xa, xb, xc, fa, fb, fc, funcalls = out
        self.assertIsInstance(xa, float)
        self.assertIsInstance(xb, float)
        self.assertIsInstance(xc, float)
        self.assertIsInstance(fa, float)
        self.assertIsInstance(fb, float)
        self.assertIsInstance(fc, float)
        self.assertIsInstance(funcalls, int)

    def test_output_conditions(self):
        x0, x1 = 100, 120
        xa, xb, xc, fa, fb, fc, funcalls = mm.bracket(
            test_fun_1, x0, x1, (), self.grow_limit, self.max_iter
        )
        condition_1 = (xa <= xb) and (xb <= xc)
        condition_2 = (xc <= xb) and (xb <= xa)
        self.assertTrue(condition_1 or condition_2)

        self.assertLessEqual(fb, fa)
        self.assertLessEqual(fb, fc)

    def test_initial_condition_with_minima_inside(self):
        x0, x1 = -3.0, 0.0
        xa, xb, xc, fa, fb, fc, funcalls = mm.bracket(
            test_fun_2, x0, x1, (), self.grow_limit, self.max_iter
        )
        self.assertLessEqual(fb, fa)
        self.assertLessEqual(fb, fc)
        condition_1 = (xa <= xb) and (xb <= xc)
        condition_2 = (xc <= xb) and (xb <= xa)
        self.assertTrue(condition_1 or condition_2)

        # to be fixed
        x0, x1 = 0.3, 0.68
        xa, xb, xc, fa, fb, fc, funcalls = mm.bracket(test_fun_4, x0, x1, (), 1.0, self.max_iter)
        self.assertLess(xb, x1)
        self.assertGreater(xb, x0)

    def test_initial_condition_with_multiple_minima_inside(self):
        x0, x1 = -5.0, 5.0
        xa, xb, xc, fa, fb, fc, funcalls = mm.bracket(
            test_fun_2, x0, x1, (), self.grow_limit, self.max_iter
        )
        self.assertLessEqual(fb, fa)
        self.assertLessEqual(fb, fc)
        condition_1 = (xa <= xb) and (xb <= xc)
        condition_2 = (xc <= xb) and (xb <= xa)
        self.assertTrue(condition_1 or condition_2)

    def test_max_iter_reached(self):
        x0, x1 = 290.0, 300.0
        with self.assertRaises(RuntimeError):
            mm.bracket(test_fun_3, x0, x1, (), 1.0, 3)


class TestGetBraketInfo(TestCase):
    def test_output(self):
        pass


class TestBrentMinimizer(TestCase):
    def setUp(self):
        self.tol = 1e-8
        self.max_iter = 300
        self.multiple_tols = [1e-8, 1e-10, 1e-12]

    def test_output(self):
        x0, x1 = -1.1, 1.1
        out = mm.brent_minimize_scalar(test_fun_1, x0, x1, self.tol, self.max_iter, ())
        self.assertIsInstance(out, tuple)
        xstar, fxstar = out
        self.assertIsInstance(xstar, float)
        self.assertIsInstance(fxstar, float)

    def test_interval_with_minima(self):
        x0, x1 = -1.1, 1.1
        xstar, fxstar = mm.brent_minimize_scalar(test_fun_1, x0, x1, self.tol, self.max_iter, ())
        self.assertAlmostEqual(xstar, 0.0, delta=self.tol)
        self.assertAlmostEqual(fxstar, -1.0, delta=self.tol)

    def test_interval_without_minima(self):
        pass

    def test_max_iter_reached(self):
        pass
        # x0, x1 = -10000, 1000000
        # with self.assertRaises(RuntimeError):
        #     mm.brent_minimize_scalar(test_fun_1, x0, x1, self.tol, 2, ())

    def test_precision(self):
        pass


if __name__ == "__main__":
    main()
