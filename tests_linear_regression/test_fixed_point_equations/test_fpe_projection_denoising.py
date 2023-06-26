from unittest import TestCase, main
from numpy.random import rand
import linear_regression.fixed_point_equations.fpe_projection_denoising as fpe_pd


class TestVarFuncProjectionDenoising(TestCase):
    def test_output_values(self):
        random_vals = 10.0 * rand(10) + 0.01
        for rv in random_vals:
            self.assertEqual(fpe_pd.var_func_projection_denoising(1.0, 1.0, 1.0, rv)[1], rv)


if __name__ == "__main__":
    main()
