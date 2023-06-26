from unittest import main
from numpy.random import randn, rand
import linear_regression.aux_functions.misc as misc
from tests_linear_regression.function_comparison import TestFunctionComparison


class TestDampedUpdate(TestFunctionComparison):
    def test_values(self):
        self.compare_two_functions(
            misc.damped_update, lambda x, y: 0.5 * (x - y) ** 2, arg_signatures=("u", "u")
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
