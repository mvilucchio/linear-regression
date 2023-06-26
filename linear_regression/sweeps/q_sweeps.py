from typing import Tuple
from ..regression_numerics import TOL_GAMP, BLEND_GAMP, MAX_ITER_GAMP
from numpy import logspace, empty, mean, std, around
from ..regression_numerics.amp_funcs import GAMP_algorithm_unsimplified
from ..regression_numerics.data_generation import data_generation
from ..utils.errors import ConvergenceError
from math import log10
from ..aux_functions.misc import estimation_error


# add function to sweep over q in the state evolution


def sweep_fw_first_arg_GAMP(
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    measure_fun: callable,
    alpha : float,
    fw_arg_min: float,
    fw_arg_max: float,
    n_fw_arg_pts: int,
    repetitions: int,
    n_features: int,
    f_out_args: tuple,
    measure_fun_args: Tuple,
    funs=[estimation_error],
    funs_args=[list()],
    decreasing=False,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
    tolerate_not_convergence=True,
):
    n_observables = len(funs)
    fw_args = (
        logspace(log10(fw_arg_min), log10(fw_arg_max), n_fw_arg_pts)
        if not decreasing
        else logspace(log10(fw_arg_max), log10(fw_arg_min), n_fw_arg_pts)
    )
    out_list_mean = empty((n_observables, n_fw_arg_pts))
    out_list_std = empty((n_observables, n_fw_arg_pts))

    for idx, fw_arg in enumerate(fw_args):
        all_values = [list() for _ in range(n_observables)]

        for _ in range(repetitions):
            try:
                xs, ys, _, _, ground_truth_theta = data_generation(
                    measure_fun,
                    n_features=n_features,
                    n_samples=max(int(around(n_features * alpha)), 1),
                    n_generalization=1,
                    measure_fun_args=measure_fun_args,
                )

                # we want to initialize them at the fixed point so:
                estimated_theta = GAMP_algorithm_unsimplified(
                    f_w,
                    Df_w,
                    f_out,
                    Df_out,
                    ys,
                    xs,
                    (fw_arg,),
                    f_out_args,
                    ground_truth_theta ,
                    abs_tol=abs_tol,
                    max_iter=max_iter,
                    blend=blend,
                )

                for kdx, (f, f_args) in enumerate(zip(funs, funs_args)):
                    all_values[kdx].append(f(ys, xs, estimated_theta, ground_truth_theta, *f_args))

                del xs
                del ys
                del ground_truth_theta

            except ConvergenceError as e:
                if not tolerate_not_convergence:
                    raise e

                print(e)

                del xs
                del ys
                del ground_truth_theta

        for kdx in range(n_observables):
            out_list_mean[kdx][idx] = mean(all_values[kdx])
            out_list_std[kdx][idx] = std(all_values[kdx])

    return fw_args, out_list_mean, out_list_std
