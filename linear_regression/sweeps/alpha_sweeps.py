from numpy import logspace, linspace, empty, nan, around, mean, std
from numpy.random import normal
from math import log10
from typing import Tuple
from ..utils.errors import ConvergenceError
from ..fixed_point_equations.fpeqs import fixed_point_finder
from ..data.generation import data_generation
from ..amp.amp_funcs import GAMP_algorithm_unsimplified
from ..aux_functions.misc import estimation_error
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM
from ..erm import TOL_GAMP, MAX_ITER_GAMP, BLEND_GAMP
from ..fixed_point_equations.optimality_finding import (
    find_optimal_reg_param_function,
    find_optimal_reg_and_huber_parameter_function,
)
import numpy as np


def sweep_alpha_fixed_point(
    f_func,
    f_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[{}],
    update_funs_args=None,
    decreasing=False,
):
    if update_funs_args is None:
        update_funs_args = [False] * len(funs)

    n_funs = len(funs)
    n_funs_args = len(funs_args)
    n_update_funs_args = len(update_funs_args)

    if not (n_funs == n_funs_args == n_update_funs_args):
        raise ValueError(
            "The length of funs, funs_args and update_funs_args should be the same, in this case is {:d}, {:d} and {:d}".format(
                n_funs, n_funs_args, n_update_funs_args
            )
        )

    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    if alpha_min <= 0:
        raise ValueError("alpha_min should be positive, in this case is {:f}".format(alpha_min))

    n_observables = len(funs)
    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    out_list = [empty(n_alpha_pts) for _ in range(n_observables)]
    # this is not needed
    ms_qs_sigmas = empty((n_alpha_pts, 3))

    old_initial_cond = initial_cond_fpe
    for idx, alpha in enumerate(alphas):
        print(f"\t{alpha = }")
        f_hat_kwargs.update({"alpha": alpha})
        ms_qs_sigmas[idx] = fixed_point_finder(
            f_func, f_hat_func, old_initial_cond, f_kwargs, f_hat_kwargs
        )
        old_initial_cond = tuple(ms_qs_sigmas[idx])
        m, q, sigma = ms_qs_sigmas[idx]

        print(f"\t\t{m = :.3e}, {q = :.3e}, {sigma = :.3e}")
        for jdx, (f, f_args, update_flag) in enumerate(zip(funs, funs_args, update_funs_args)):
            if update_flag:
                f_args.update({"m": m, "q": q, "sigma": sigma})
                out_list[jdx][idx] = f(m, q, sigma, **f_args)
            else:
                out_list[jdx][idx] = f(m, q, sigma, **f_args)

    if decreasing:
        alphas = alphas[::-1]
        for idx, obs_vals in enumerate(out_list):
            out_list[idx] = obs_vals[::-1]

    return alphas, out_list


def sweep_alpha_optimal_lambda_fixed_point(
    f_func,
    f_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    inital_guess_lambda: float,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[{}],
    update_funs_args=None,
    f_min=estimation_error,
    f_min_args={},
    update_f_min_args=False,
    min_reg_param=SMALLEST_REG_PARAM,
    decreasing=False,
):
    if update_funs_args is None:
        update_funs_args = [False] * len(funs)

    n_funs = len(funs)
    n_funs_args = len(funs_args)
    n_update_funs_args = len(update_funs_args)

    if not (n_funs == n_funs_args == n_update_funs_args):
        raise ValueError(
            "The length of funs, funs_args and update_funs_args should be the same, in this case is {:d}, {:d} and {:d}".format(
                n_funs, n_funs_args, n_update_funs_args
            )
        )

    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    if alpha_min <= 0:
        raise ValueError("alpha_min should be positive, in this case is {:f}".format(alpha_min))

    n_observables = len(funs)
    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    f_min_vals = empty(n_alpha_pts)
    reg_params_opt = empty(n_alpha_pts)
    funs_values = [empty(n_alpha_pts) for _ in range(n_observables)]

    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_lambda
    for idx, alpha in enumerate(alphas):
        print(f"\talpha = {alpha}")
        copy_f_hat_kwargs.update({"alpha": float(alpha)})
        copy_f_kwargs.update({"reg_param": np.random.rand(1) + 0.0001})

        if update_f_min_args:
            f_min_args.update({"alpha": float(alpha)})

        for jdx, update_flag in enumerate(update_funs_args):
            if update_flag:
                copy_funs_args[jdx].update({"alpha": float(alpha)})

        (
            f_min_vals[idx],
            reg_params_opt[idx],
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_param_function(
            f_func,
            f_hat_func,
            copy_f_kwargs,
            copy_f_hat_kwargs,
            old_reg_param_opt,
            old_initial_cond_fpe,
            funs=funs,
            funs_args=copy_funs_args,
            f_min=f_min,
            f_min_args=f_min_args,
            min_reg_param=min_reg_param,
        )
        old_reg_param_opt = reg_params_opt[idx]
        old_initial_cond_fpe = (m, q, sigma)

        for jdx in range(n_observables):
            funs_values[jdx][idx] = out_values[jdx]

    if decreasing:
        alphas = alphas[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return alphas, f_min_vals, reg_params_opt, funs_values


def sweep_alpha_optimal_lambda_hub_param_fixed_point(
    f_func,
    f_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    inital_guess_params: Tuple[float, float],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[list()],
    update_funs_args=None,
    f_min=estimation_error,
    f_min_args={},
    update_f_min_args=False,
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
    decreasing=False,
):
    if update_funs_args is None:
        update_funs_args = [False] * len(funs)

    n_funs = len(funs)
    n_funs_args = len(funs_args)
    n_update_funs_args = len(update_funs_args)

    if not (n_funs == n_funs_args == n_update_funs_args):
        raise ValueError(
            "The length of funs, funs_args and update_funs_args should be the same, in this case is {:d}, {:d} and {:d}".format(
                n_funs, n_funs_args, n_update_funs_args
            )
        )

    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    if alpha_min <= 0:
        raise ValueError("alpha_min should be positive, in this case is {:f}".format(alpha_min))

    n_observables = len(funs)
    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    f_min_vals = empty(n_alpha_pts)
    reg_params_opt = empty(n_alpha_pts)
    hub_params_opt = empty(n_alpha_pts)
    funs_values = [empty(n_alpha_pts) for _ in range(n_observables)]

    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_params[0]
    old_hub_param_opt = inital_guess_params[1]
    for idx, alpha in enumerate(alphas):
        # print(f"\talpha = {alpha}")
        copy_f_hat_kwargs.update({"alpha": alpha, "a": old_hub_param_opt})
        copy_f_kwargs.update({"reg_param": old_reg_param_opt})

        if update_f_min_args:
            f_min_args.update({"alpha": alpha})

        for jdx, update_flag in enumerate(update_funs_args):
            if update_flag:
                copy_funs_args[jdx].update({"alpha": alpha})

        (
            f_min_vals[idx],
            (reg_params_opt[idx], hub_params_opt[idx]),
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_and_huber_parameter_function(
            f_func,
            f_hat_func,
            copy_f_kwargs,
            copy_f_hat_kwargs,
            (
                old_reg_param_opt,
                old_hub_param_opt,
            ),  # (float(np.random.rand(1) + 1), old_hub_param_opt),
            old_initial_cond_fpe,
            funs=funs,
            funs_args=copy_funs_args,
            f_min=f_min,
            f_min_args=f_min_args,
            min_reg_param=min_reg_param,
            min_huber_param=min_huber_param,
        )

        old_reg_param_opt = reg_params_opt[idx]
        old_hub_param_opt = hub_params_opt[idx]
        old_initial_cond_fpe = (m, q, sigma)

        for jdx in range(n_observables):
            funs_values[jdx][idx] = out_values[jdx]

    if decreasing:
        alphas = alphas[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return alphas, f_min_vals, (reg_params_opt, hub_params_opt), funs_values


# ------------------- #


def sweep_alpha_GAMP(
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    measure_fun: callable,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    repetitions: int,
    n_features: int,
    f_w_args: tuple,
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
    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    out_list_mean = empty((n_observables, n_alpha_pts))
    out_list_std = empty((n_observables, n_alpha_pts))

    for idx, alpha in enumerate(alphas):
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

                estimated_theta = GAMP_algorithm_unsimplified(
                    f_w,
                    Df_w,
                    f_out,
                    Df_out,
                    ys,
                    xs,
                    f_w_args,
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

    return alphas, out_list_mean, out_list_std


# ------------------- #


def sweep_alpha_descend_lambda(
    f_func,
    f_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    lambda_min: float,
    lambda_max: float,
    n_lambda_pts: int,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    funs=[estimation_error],
    funs_args=[list()],
    initial_cond_fpe=(0.6, 0.01, 0.9),
):
    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    if alpha_min <= 0:
        raise ValueError("alpha_min should be positive, in this case is {:f}".format(alpha_min))

    if lambda_min > lambda_max:
        raise ValueError(
            "lambda_min should be smaller than lambda_max, in this case are {:f} and {:f}".format(
                lambda_min, lambda_max
            )
        )

    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    alphas = logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
    reg_params = linspace(lambda_min, lambda_max, n_lambda_pts)
    funs_vals = [empty((n_lambda_pts, n_alpha_pts)) for _ in range(len(funs))]

    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()
    old_initial_cond = initial_cond_fpe
    first_inital_cond_column = initial_cond_fpe
    for idx, alpha in enumerate(alphas):
        copy_f_hat_kwargs.update({"alpha": alpha})
        old_initial_cond = first_inital_cond_column

        already_brokern = False
        for jdx, reg_param in enumerate(reg_params[::-1]):
            copy_f_kwargs.update({"reg_param": reg_param})

            # if reg_param <= min(0,1-alpha):
            #     for kdx, (f, f_args) in enumerate(zip(funs, funs_args)):
            #         funs_vals[kdx][n_lambda_pts - 1 - jdx, idx] = nan
            #     continue

            if already_brokern:
                for kdx, (f, f_args) in enumerate(zip(funs, funs_args)):
                    funs_vals[kdx][n_lambda_pts - 1 - jdx, idx] = nan
                continue

            try:
                m, q, sigma = fixed_point_finder(
                    f_func,
                    f_hat_func,
                    old_initial_cond,
                    copy_f_kwargs,
                    copy_f_hat_kwargs,
                )
                old_initial_cond = tuple([m, q, sigma])

                if jdx == 0:
                    first_inital_cond_column = old_initial_cond

                for kdx, (f, f_args) in enumerate(zip(funs, funs_args)):
                    funs_vals[kdx][n_lambda_pts - 1 - jdx, idx] = f(m, q, sigma, *f_args)

            except ConvergenceError as e:
                for kdx, (f, f_args) in enumerate(zip(funs, funs_args)):
                    funs_vals[kdx][n_lambda_pts - 1 - jdx, idx] = nan

                already_brokern = True
                continue

            except ValueError:
                for kdx, (f, f_args) in enumerate(zip(funs, funs_args)):
                    funs_vals[kdx][n_lambda_pts - 1 - jdx, idx] = nan

                already_brokern = True
                continue
    return (alphas, reg_params), funs_vals


def sweep_alpha_minimal_stable_reg_param(
    f_func,
    f_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    condition_func,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    decreasing=False,
    bounds_reg_param_search=(-10.0, 0.01),
    points_per_run=1000,
):
    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    if alpha_min <= 0:
        raise ValueError("alpha_min should be positive, in this case is {:f}".format(alpha_min))

    if bounds_reg_param_search[0] > bounds_reg_param_search[1]:
        raise ValueError(
            "bounds_reg_param_search[0] should be smaller than bounds_reg_param_search[1], in this case are {:f} and {:f}".format(
                bounds_reg_param_search[0], bounds_reg_param_search[1]
            )
        )

    if bounds_reg_param_search[1] <= 0.0:
        raise ValueError(
            "bounds_reg_param_search[1] should be larger than 0.0, in this case is {:f}".format(
                bounds_reg_param_search[1]
            )
        )

    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    last_reg_param_stable = empty(n_alpha_pts)

    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()
    old_initial_cond = initial_cond_fpe
    for idx, alpha in enumerate(alphas):
        copy_f_hat_kwargs.update({"alpha": alpha})

        not_converged_idx = 0
        reg_params_test = linspace(
            bounds_reg_param_search[0], bounds_reg_param_search[1], points_per_run
        )

        for jdx, reg_param in enumerate(reg_params_test[::-1]):
            copy_f_kwargs.update({"reg_param": reg_param})

            try:
                m, q, sigma = fixed_point_finder(
                    f_func,
                    f_hat_func,
                    old_initial_cond,
                    copy_f_kwargs,
                    copy_f_hat_kwargs,
                )
                old_initial_cond = tuple([m, q, sigma])

                if condition_func(m, q, sigma, **copy_f_kwargs, **copy_f_hat_kwargs) <= 0.0:
                    not_converged_idx = points_per_run - 1 - jdx
                    break

            except ConvergenceError as e:
                not_converged_idx = points_per_run - 1 - jdx
                break
            except ValueError as e:
                not_converged_idx = points_per_run - 1 - jdx
                break

        last_reg_param_stable[idx] = reg_params_test[not_converged_idx]

    if decreasing:
        alphas = alphas[::-1]
        last_reg_param_stable = last_reg_param_stable[::-1]

    return alphas, last_reg_param_stable
