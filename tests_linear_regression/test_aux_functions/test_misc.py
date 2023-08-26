from unittest import main
from numpy.random import randn, rand
from numpy.linalg import norm
import linear_regression.aux_functions.misc as misc
from tests_linear_regression.function_comparison import TestFunctionComparison


class TestSampleVectorRandom(TestFunctionComparison):
    def test_shape(self):
        n_features_list = [10, 100, 1_000, 10_000]
        squared_radius = 2.0
        for n_features in n_features_list:
            random_vector = misc.sample_vector_random(n_features, squared_radius)
            self.assertEqual(random_vector.shape, (n_features,))

    def test_norm(self):
        squared_radius_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        n_features = 100
        for squared_radius in squared_radius_list:
            random_vector = misc.sample_vector_random(n_features, squared_radius)
            self.assertAlmostEqual(norm(random_vector)**2, squared_radius)


class TestDampedUpdate(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            misc.damped_update, lambda x, y, d: d * x + (1 - d) * y, arg_signatures=("u", "u", "0-1")
        )

    def test_broadcasting(self):
        new = rand(5, 5)
        old = rand(5, 5)
        damping = 0.5
        damping_matrix = rand(5, 5)

        # Test with correct input shapes
        try:
            misc.damped_update(new, old, damping)
        except ValueError:
            self.fail("damped_update raised ValueError unexpectedly!")

        new = rand(5, 5)
        old = rand(5, 5)
        damping = rand(5, 5)

        try:
            misc.damped_update(new, old, damping)
        except ValueError:
            self.fail("damped_update raised ValueError unexpectedly!")

        # Test with incorrect input shapes for the first two arguments
        new = rand(5, 4)
        old = rand(5, 5)
        with self.assertRaises(ValueError):
            misc.damped_update(new, old, damping)

        # Test with incorrect input shape for the last argument
        new = rand(5, 5)
        old = rand(5, 5)
        damping = rand(5, 4)
        with self.assertRaises(ValueError):
            misc.damped_update(new, old, damping)


if __name__ == "__main__":
    main()
