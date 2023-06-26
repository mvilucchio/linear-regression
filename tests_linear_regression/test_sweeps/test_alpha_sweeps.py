from unittest import TestCase, main
from numpy import logspace, empty
import linear_regression.sweeps.alpha_sweeps as alsw


class TestSweepAlphaFixedPoint(TestCase):
    def setUp(self):
        super().setUp()
        self.var_func_placeholder = lambda x: x
        self.var_hat_func_placeholder = lambda x: x
        self.var_func_kwargs_placeholder = {}
        self.var_hat_func_kwargs_placeholder = {}
        self.decreasing_placeholder = False
        self.alpha_min_placeholder = 1.0
        self.alpha_max_placeholder = 10.0
        self.n_alpha_pts_placeholder = 100
        self.intital_guess_placeholder = (1.0, 2.0, 3.0)
        self.funs_placeholder = [lambda x: x]
        self.funs_args_placeholder = [[]]
    
    def test_funs_funs_args_mismatch(self):
        funs = [lambda x: x]
        funs_args = [[], []]
        with self.assertRaises(ValueError):
            alsw.sweep_alpha_fixed_point(
                self.var_func_placeholder,
                self.var_hat_func_placeholder,
                self.alpha_min_placeholder,
                self.alpha_max_placeholder,
                self.n_alpha_pts_placeholder,
                self.var_func_kwargs_placeholder,
                self.var_hat_func_kwargs_placeholder,
                self.intital_guess_placeholder,
                funs,
                funs_args,
                self.decreasing_placeholder,
            )

    def test_alpha_min_greater_than_alpha_max(self):
        alpha_min = 1.0
        alpha_max = 0.1
        with self.assertRaises(ValueError):
            alsw.sweep_alpha_fixed_point(
                self.var_func_placeholder,
                self.var_hat_func_placeholder,
                alpha_min,
                alpha_max,
                self.n_alpha_pts_placeholder,
                self.var_func_kwargs_placeholder,
                self.var_hat_func_kwargs_placeholder,
                self.intital_guess_placeholder,
                self.funs_placeholder,
                self.funs_args_placeholder,
                self.decreasing_placeholder,
            )

    def test_alpha_min_negative(self):
        var_func = lambda x: x
        var_hat_func = lambda x: x
        alpha_min = -0.1
        alpha_max = 1.0
        n_alpha_pts = 10
        var_func_kwargs = {}
        var_hat_func_kwargs = {}
        initial_cond_fpe = (0.6, 0.01, 0.9)
        funs = [lambda x: x]
        funs_args = [[]]
        decreasing = False
        with self.assertRaises(ValueError):
            alsw.sweep_alpha_fixed_point(
                var_func,
                var_hat_func,
                alpha_min,
                alpha_max,
                n_alpha_pts,
                var_func_kwargs,
                var_hat_func_kwargs,
                initial_cond_fpe,
                funs,
                funs_args,
                decreasing,
            )


if __name__ == "__main__":
    main()
