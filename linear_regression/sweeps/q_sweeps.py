from ..fixed_point_equations.fpeqs import fixed_point_finder, fixed_point_finder_loser

from ..erm import TOL_GAMP, BLEND_GAMP, MAX_ITER_GAMP
from numpy import logspace, empty, mean, std, around
from ..amp.amp_funcs import GAMP_algorithm_unsimplified
from ..data.generation import data_generation
from ..utils.errors import ConvergenceError
from math import log10
from ..aux_functions.misc import estimation_error
from ..fixed_point_equations.regularisation.fpe_projection_denoising import f_projection_denoising


def sweep_q_fixed_point_proj_denoiser(
    f_hat_func,
    q_min: float,
    q_max: float,
    n_q_pts: int,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[{}],
    update_funs_args=None,
    decreasing: bool = False,
):
    if update_funs_args is None:
        update_funs_args = [False] * len(funs)

    n_funs = len(funs)
    n_funs_args = len(funs_args)
    n_update_funs_args = len(update_funs_args)

    if not (n_funs == n_funs_args == n_update_funs_args):
        raise ValueError(
            "funs, funs_args and update_funs_args must have the same length, in this case: {}, {}, {}".format(
                n_funs, n_funs_args, n_update_funs_args
            )
        )

    if q_min <= 0.0 or q_max <= 0.0:
        raise ValueError("q_min and q_max must be positive")

    if q_min >= q_max:
        raise ValueError("q_min must be smaller than q_max")

    if n_q_pts <= 0:
        raise ValueError("n_q_pts must be positive")

    n_observables = len(funs)
    qs = (
        logspace(log10(q_min), log10(q_max), n_q_pts)
        if not decreasing
        else logspace(log10(q_max), log10(q_min), n_q_pts)
    )

    out_list = [empty(n_q_pts) for _ in range(n_observables)]
    ms_qs_Vs = empty((n_q_pts, 3))

    old_initial_cond_fpe = initial_cond_fpe
    for idx, q in enumerate(qs):
        # print(f"\tq = {q}")
        f_kwargs.update({"q_fixed": q})
        ms_qs_Vs[idx] = fixed_point_finder_loser(
            f_projection_denoising,
            f_hat_func,
            old_initial_cond_fpe,
            f_kwargs,
            f_hat_kwargs,
            control_variate=(True, True, True),
        )
        old_initial_cond_fpe = ms_qs_Vs[idx]
        m, q, V = ms_qs_Vs[idx]
        # print(f"\tm = {m}, q = {q}, V = {V}")

        for jdx, (f, f_args, update_f_args) in enumerate(zip(funs, funs_args, update_funs_args)):
            if update_f_args:
                f_kwargs.update(f_args)
                out_list[jdx][idx] = f(m, q, V, **f_kwargs)
            else:
                out_list[jdx][idx] = f(m, q, V, **f_args)

    if decreasing:
        qs = qs[::-1]
        ms_qs_Vs = ms_qs_Vs[::-1]
        for idx in range(n_observables):
            out_list[idx] = out_list[idx][::-1]

    return qs, out_list


def sweep_fw_first_arg_GAMP(
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    measure_fun: callable,
    alpha: float,
    fw_arg_min: float,
    fw_arg_max: float,
    n_fw_arg_pts: int,
    repetitions: int,
    n_features: int,
    f_out_args: tuple,
    measure_fun_args: tuple,
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
                    ground_truth_theta,
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
