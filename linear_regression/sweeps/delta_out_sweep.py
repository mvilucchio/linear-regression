from numpy import logspace, empty
from numpy.random import rand
from math import log10
from typing import Tuple
from ..fixed_point_equations.fpeqs import fixed_point_finder
from ..aux_functions.misc import estimation_error
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM
from ..fixed_point_equations.optimality_finding import (
    find_optimal_reg_param_function,
    find_optimal_reg_and_huber_parameter_function,
)


def sweep_delta_out_fixed_point(
    var_func,
    var_hat_func,
    delta_out_min: float,
    delta_out_max: float,
    n_deltas_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond=(0.6, 0.01, 0.9),
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

    if delta_out_min > delta_out_max:
        raise ValueError(
            "delta_out_min should be smaller than delta_out_max, in this case are {:f} and {:f}".format(
                delta_out_min, delta_out_max
            )
        )

    n_observables = len(funs)
    deltas = (
        logspace(log10(delta_out_min), log10(delta_out_max), n_deltas_pts)
        if not decreasing
        else logspace(log10(delta_out_max), log10(delta_out_min), n_deltas_pts)
    )
    out_list = [empty(n_deltas_pts) for _ in range(n_observables)]

    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()

    old_initial_cond = initial_cond
    for idx, delta in enumerate(deltas):
        copy_var_hat_func_kwargs.update({"delta_out": delta})

        m, q, sigma = fixed_point_finder(
            var_func, var_hat_func, old_initial_cond, var_func_kwargs, copy_var_hat_func_kwargs
        )

        old_initial_cond = tuple([m, q, sigma])

        for jdx, (f, f_args, update_flag) in enumerate(zip(funs, funs_args, update_funs_args)):
            if update_flag:
                f_args.update({"delta_out": float(delta)})
                out_list[jdx][idx] = f(m, q, sigma, **f_args)
            else:
                out_list[jdx][idx] = f(m, q, sigma, **f_args)

    if decreasing:
        deltas = deltas[::-1]
        for idx, obs_vals in enumerate(out_list):
            out_list[idx] = obs_vals[::-1]

    return deltas, out_list


def sweep_delta_out_optimal_lambda_fixed_point(
    var_func,
    var_hat_func,
    delta_out_min: float,
    delta_out_max: float,
    n_delta_out_pts: int,
    inital_guess_lambda: float,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[{}],
    update_funs_args=None,
    f_min=estimation_error,
    f_min_args={},
    update_f_min_args=False,
    min_reg_param=SMALLEST_REG_PARAM,
    decreasing=False,
    following_sol=True,
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

    if delta_out_min > delta_out_max:
        raise ValueError(
            "delta_out_min should be smaller than delta_out_max, in this case are {:f} and {:f}".format(
                delta_out_min, delta_out_max
            )
        )

    n_observables = len(funs)
    delta_outs = (
        logspace(log10(delta_out_min), log10(delta_out_max), n_delta_out_pts)
        if not decreasing
        else logspace(log10(delta_out_max), log10(delta_out_min), n_delta_out_pts)
    )
    f_min_vals = empty(n_delta_out_pts)
    reg_params_opt = empty(n_delta_out_pts)
    funs_values = [empty(n_delta_out_pts) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_lambda
    for idx, delta_out in enumerate(delta_outs):
        copy_var_hat_func_kwargs.update({"delta_out": delta_out})
        copy_var_func_kwargs.update({"reg_param": old_reg_param_opt})

        if update_f_min_args:
            f_min_args.update({"delta_out": delta_out})

        for jdx, update_flag in enumerate(update_funs_args):
            if update_flag:
                copy_funs_args[jdx].update({"delta_out": delta_out})

        (
            f_min_vals[idx],
            reg_params_opt[idx],
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_param_function(
            var_func,
            var_hat_func,
            copy_var_func_kwargs,
            copy_var_hat_func_kwargs,
            old_reg_param_opt if following_sol else float(rand(1)),
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

    # change this
    if decreasing:
        delta_outs = delta_outs[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return delta_outs, f_min_vals, reg_params_opt, funs_values


def sweep_delta_out_optimal_lambda_hub_param_fixed_point(
    var_func,
    var_hat_func,
    delta_out_min: float,
    delta_out_max: float,
    n_delta_out_pts: int,
    inital_guess_params: Tuple[float, float],
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[{}],
    update_funs_args=None,
    f_min=estimation_error,
    f_min_args={},
    update_f_min_args=False,
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
    decreasing=False,
    following_sol=True,
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

    if delta_out_min > delta_out_max:
        raise ValueError(
            "delta_out_min should be smaller than delta_out_max, in this case are {:f} and {:f}".format(
                delta_out_min, delta_out_max
            )
        )

    n_observables = len(funs)
    delta_outs = (
        logspace(log10(delta_out_min), log10(delta_out_max), n_delta_out_pts)
        if not decreasing
        else logspace(log10(delta_out_max), log10(delta_out_min), n_delta_out_pts)
    )
    f_min_vals = empty(n_delta_out_pts)
    reg_params_opt = empty(n_delta_out_pts)
    hub_params_opt = empty(n_delta_out_pts)
    funs_values = [empty(n_delta_out_pts) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_params[0]
    old_hub_param_opt = inital_guess_params[1]
    for idx, delta_out in enumerate(delta_outs):
        copy_var_hat_func_kwargs.update({"delta_out": delta_out, "a": old_hub_param_opt})
        copy_var_func_kwargs.update({"reg_param": old_reg_param_opt})

        if update_f_min_args:
            f_min_args.update({"delta_out": delta_out})

        for jdx, update_flag in enumerate(update_funs_args):
            if update_flag:
                copy_funs_args[jdx].update({"delta_out": delta_out})
        (
            f_min_vals[idx],
            (reg_params_opt[idx], hub_params_opt[idx]),
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_and_huber_parameter_function(
            var_func,
            var_hat_func,
            copy_var_func_kwargs,
            copy_var_hat_func_kwargs,
            (
                old_reg_param_opt if following_sol else float(rand(1)),
                old_hub_param_opt if following_sol else float(rand(1)),
            ),
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

    #  to be checked
    if decreasing:
        delta_outs = delta_outs[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        hub_params_opt = hub_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return delta_outs, f_min_vals, (reg_params_opt, hub_params_opt), funs_values
