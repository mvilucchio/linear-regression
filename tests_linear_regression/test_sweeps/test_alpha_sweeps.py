from unittest import TestCase, main
from numpy import logspace, empty
import linear_regression.sweeps.alpha_sweeps as alsw


class TestSweepAlphaFixedPoint(TestCase):
    def setUp(self):
        super().setUp()
        self.f_placeholder = lambda x: x
        self.f_hat_placeholder = lambda x: x
        self.f_kwargs_placeholder = {}
        self.f_hat_kwargs_placeholder = {}
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
                self.f_placeholder,
                self.f_hat_placeholder,
                self.alpha_min_placeholder,
                self.alpha_max_placeholder,
                self.n_alpha_pts_placeholder,
                self.f_kwargs_placeholder,
                self.f_hat_kwargs_placeholder,
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
                self.f_placeholder,
                self.f_hat_placeholder,
                alpha_min,
                alpha_max,
                self.n_alpha_pts_placeholder,
                self.f_kwargs_placeholder,
                self.f_hat_kwargs_placeholder,
                self.intital_guess_placeholder,
                self.funs_placeholder,
                self.funs_args_placeholder,
                self.decreasing_placeholder,
            )

    def test_alpha_min_negative(self):
        f_func = lambda x: x
        f_hat_func = lambda x: x
        alpha_min = -0.1
        alpha_max = 1.0
        n_alpha_pts = 10
        f_kwargs = {}
        f_hat_kwargs = {}
        initial_cond_fpe = (0.6, 0.01, 0.9)
        funs = [lambda x: x]
        funs_args = [[]]
        decreasing = False
        with self.assertRaises(ValueError):
            alsw.sweep_alpha_fixed_point(
                f_func,
                f_hat_func,
                alpha_min,
                alpha_max,
                n_alpha_pts,
                f_kwargs,
                f_hat_kwargs,
                initial_cond_fpe,
                funs,
                funs_args,
                decreasing,
            )


if __name__ == "__main__":
    main()
