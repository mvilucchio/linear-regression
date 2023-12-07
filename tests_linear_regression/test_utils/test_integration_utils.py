from unittest import TestCase, main
from numba import njit
import linear_regression.utils.integration_utils as iu
import numpy as np


@njit
def constant_fun(x: float) -> float:
    return 1.0


@njit
def linear_fun(x: float) -> float:
    return x


@njit
def quadratic_fun(x: float) -> float:
    return x**2


class TestGaussHermiteQuadrature(TestCase):
    def test_constant_function(self):
        mean = 0.0
        std = 1.0
        result = iu.gauss_hermite_quadrature(constant_fun, mean, std)
        self.assertAlmostEqual(result, 5.0 * (mean + std) - 5.0 * (mean - std), places=5)

    def test_linear_function(self):
        mean = 0.0
        std = 1.0
        result = iu.gauss_hermite_quadrature(linear_fun, mean, std)
        self.assertAlmostEqual(result, 0.5 * (mean + std) ** 2 - 0.5 * (mean - std) ** 2, places=5)

    def test_quadratic_function(self):
        mean = 0.0
        std = 1.0
        result = iu.gauss_hermite_quadrature(quadratic_fun, mean, std)
        expected = (1 / 3) * (mean + std) ** 3 - (1 / 3) * (mean - std) ** 3
        self.assertAlmostEqual(result, expected, places=5)


class TestFindIntegrationBordersSquare(TestCase):
    def test1(self):
        self.assertAlmostEqual(0.0, 0.0)


class TestDivideIntegrationBordersMultipleGrid(TestCase):
    def test_N_equals_0(self):
        square_borders = [(-1, 1), (-1, 1)]
        N = 0
        with self.assertRaises(ValueError):
            iu.divide_integration_borders_multiple_grid(square_borders, N)

    def test_N_equals_1(self):
        square_borders = [(-1, 1), (-1, 1)]
        N = 1
        domain_x, domain_y = iu.divide_integration_borders_multiple_grid(square_borders, N)
        expected_domain_x = [[-1.0, 1.0]]
        expected_domain_y = [[-1.0, 1.0]]
        self.assertEqual(domain_x, expected_domain_x)
        self.assertEqual(domain_y, expected_domain_y)

    def test_N_greater_than_1(self):
        square_borders = [(-1, 1), (-1, 1)]
        N = 4
        domain_x, domain_y = iu.divide_integration_borders_multiple_grid(square_borders, N)
        expected_domain_x = [
            [-1.0, -0.5],
            [-1.0, -0.5],
            [-1.0, -0.5],
            [-1.0, -0.5],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
        ]
        expected_domain_y = [
            [-1.0, -0.5],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [-1.0, -0.5],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [-1.0, -0.5],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [-1.0, -0.5],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
        ]
        self.assertEqual(len(domain_x), len(expected_domain_x))
        self.assertEqual(len(domain_y), len(expected_domain_y))

        self.assertEqual(domain_x, expected_domain_x)
        self.assertEqual(domain_y, expected_domain_y)

    def test_N_max_range_zero(self):
        square_borders = [(0, 0), (0, 0)]
        N = 3
        domain_x, domain_y = iu.divide_integration_borders_multiple_grid(square_borders, N)
        expected_domain_x = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
        expected_domain_y = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
        self.assertEqual(domain_x, expected_domain_x)
        self.assertEqual(domain_y, expected_domain_y)

    def test_N_max_range_negative(self):
        square_borders = [(-1, 1), (-1, 1)]
        N = 10
        domain_x, domain_y = iu.divide_integration_borders_multiple_grid(square_borders, N)
        for sublist in domain_x + domain_y:
            self.assertGreaterEqual(sublist[1], sublist[0])

    def test_output_dimensions(self):
        square_borders = [(-1, 1), (-1, 1)]
        N = 10
        domain_x, domain_y = iu.divide_integration_borders_multiple_grid(square_borders, N)
        self.assertEqual(len(domain_x), N * N)
        self.assertEqual(len(domain_y), N * N)


if __name__ == "__main__":
    main()
