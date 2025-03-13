from unittest import TestCase, main
from numpy import sign
import linear_regression.fixed_point_equations.fpeqs as fpe
from linear_regression.utils.errors import ConvergenceError


class TestFixedPointFinder(TestCase):
    def setUp(self):
        self.inital_condition = (0.0, 0.0, 0.0)
        self.expected_solution = (1.0, 3.0, 4.0)
        self.f_kwargs = {}
        self.f_hat_kwargs = {}
        return super().setUp()

    def _f_func(x1_hat, x2_hat, x3_hat, **kwargs):
        x1 = 0.5 * x1_hat
        x2 = x2_hat
        x3 = 2.0 * x3_hat
        return x1, x2, x3

    def _f_hat_func(x1, x2, x3, **kwargs):
        x1_hat = sign(3.0 * x1) + 1.0
        x2_hat = sign(2.0 * x2) + 2.0
        x3_hat = sign(5.0 * x3) + 1.0
        return x1_hat, x2_hat, x3_hat

    def test_output_format(self):
        self.assertAlmostEqual(0.0, 0.0)

    def test_fixed_point(self):
        m, q, V = fpe.fixed_point_finder(
            self._f_func,
            self._f_hat_func,
            self.inital_condition,
            self.f_kwargs,
            self.f_hat_kwargs,
            abs_tol=1e-8,
            min_iter=100,
            max_iter=10000,
        )

        self.assertAlmostEqual(
            m, self.expected_solution[0], delta=1e-5, msg="Fixed point for m not found"
        )
        self.assertAlmostEqual(
            q, self.expected_solution[1], delta=1e-5, msg="Fixed point for q not found"
        )
        self.assertAlmostEqual(
            V, self.expected_solution[2], delta=1e-5, msg="Fixed point for V not found"
        )

    def test_max_iter(self):
        abs_tol = 1e-8

        with self.assertRaises(ConvergenceError):
            fpe.fixed_point_finder(
                self._f_func,
                self._f_hat_func,
                self.inital_condition,
                self.f_kwargs,
                self.f_hat_kwargs,
                abs_tol=abs_tol,
                min_iter=100,
                max_iter=1,
            )

    def test_precision(self):
        for abs_tol, max_iter in zip([1e-2, 1e-5, 1e-8, 1e-11], [1_000, 1_000, 10_000, 100_000]):
            m, q, V = fpe.fixed_point_finder(
                self._f_func,
                self._f_hat_func,
                self.inital_condition,
                self.f_kwargs,
                self.f_hat_kwargs,
                abs_tol=abs_tol,
                min_iter=100,
                max_iter=max_iter,
            )

            self.assertAlmostEqual(
                m,
                self.expected_solution[0],
                delta=abs_tol,
                msg="Precision of {abs_tol} for m not reached}",
            )
            self.assertAlmostEqual(
                q,
                self.expected_solution[1],
                delta=abs_tol,
                msg="Precision of {abs_tol} for q not reached}",
            )
            self.assertAlmostEqual(
                V,
                self.expected_solution[2],
                delta=abs_tol,
                msg="Precision of {abs_tol} for V not reached}",
            )


class TestFixedPointFinderAnderson(TestCase):
    def setUp(self):
        self.inital_condition = (0.0, 0.0, 0.0)
        self.expected_solution = (1.0, 3.0, 4.0)
        self.f_kwargs = {}
        self.f_hat_kwargs = {}
        return super().setUp()

    def _f_func(x1_hat, x2_hat, x3_hat, **kwargs):
        x1 = 0.5 * x1_hat
        x2 = x2_hat
        x3 = 2.0 * x3_hat
        return x1, x2, x3

    def _f_hat_func(x1, x2, x3, **kwargs):
        x1_hat = sign(3.0 * x1) + 1.0
        x2_hat = sign(2.0 * x2) + 2.0
        x3_hat = sign(5.0 * x3) + 1.0
        return x1_hat, x2_hat, x3_hat

    def test_output_format(self):
        self.assertAlmostEqual(0.0, 0.0)

    def test_fixed_point(self):
        m, q, V = fpe.anderson_fixed_point_finder(
            self._f_func,
            self._f_hat_func,
            self.inital_condition,
            self.f_kwargs,
            self.f_hat_kwargs,
            abs_tol=1e-8,
            min_iter=100,
            max_iter=10000,
        )

        self.assertAlmostEqual(
            m, self.expected_solution[0], delta=1e-5, msg="Fixed point for m not found"
        )
        self.assertAlmostEqual(
            q, self.expected_solution[1], delta=1e-5, msg="Fixed point for q not found"
        )
        self.assertAlmostEqual(
            V, self.expected_solution[2], delta=1e-5, msg="Fixed point for V not found"
        )

    def test_max_iter(self):
        abs_tol = 1e-8

        with self.assertRaises(ConvergenceError):
            fpe.anderson_fixed_point_finder(
                self._f_func,
                self._f_hat_func,
                self.inital_condition,
                self.f_kwargs,
                self.f_hat_kwargs,
                abs_tol=abs_tol,
                min_iter=100,
                max_iter=1,
            )


if __name__ == "__main__":
    main()
