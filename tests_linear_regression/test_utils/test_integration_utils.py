import unittest
import linear_regression.utils.integration_utils as iu
import numpy as np


class TestIntegrationUtils(unittest.TestCase):
    def test1(self):
        self.assertAlmostEqual(0.0, 0.0)

    # def test_output(self):
    #     # Test that the output of the function is correct
    #     def f(x):
    #         return x**2

    #     mean = 1.0
    #     std = 2.0
    #     expected_output = np.exp(mean**2) * np.sum(iu.w_ge * f(np.sqrt(2) * std * iu.x_ge + mean))

    #     # Note: we cannot use @njit-ed function in assertAlmostEqual,
    #     # thus, we test np.isclose for the expected output and the function output
    #     self.assertTrue(np.isclose(iu.gauss_hermite_quadrature(f, mean, std), expected_output))

    # def test_error_raised(self):
    #     # Test that an error is raised if the input function does not return a scalar
    #     def f(x):
    #         return np.array([x**2, x**3])

    #     mean = 1.0
    #     std = 2.0
    #     with self.assertRaises(ValueError):
    #         iu.gauss_hermite_quadrature(f, mean, std)

    # def test_valid_input(self):
    #     # Test that the function executes without errors for valid input values
    #     def f(x):
    #         return np.exp(-x**2)

    #     mean = 0.0
    #     std = 1.0
    #     iu.gauss_hermite_quadrature(f, mean, std)  # Should not raise an error


class TestDivideIntegrationBordersMultipleGrid(unittest.TestCase):
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
    unittest.main()
