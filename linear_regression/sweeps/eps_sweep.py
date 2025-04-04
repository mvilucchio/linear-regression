from numpy import logspace, empty, linspace
from math import log10

from ..fixed_point_equations.fpeqs import fixed_point_finder
from ..aux_functions.misc import estimation_error
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM
from ..fixed_point_equations.optimality_finding import (
    find_optimal_reg_param_function,
    find_optimal_reg_and_huber_parameter_function,
)


def sweep_eps_fixed_point(
    f_func,
    f_hat_func,
    eps_min: float,
    eps_max: float,
    n_eps_pts: int,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_cond=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[list()],
    update_funs_args=None,
    decreasing=False,
    linear=False,
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

    if eps_min > eps_max:
        raise ValueError(
            "eps_min should be smaller than eps_max, in this case are {:f} and {:f}".format(
                eps_min, eps_max
            )
        )

    if eps_min < 0:
        raise ValueError(
            "eps_min should be positive or equal to zero, in this case is {:f}".format(eps_min)
        )

    n_observables = len(funs)
    if linear:
        epsilons = (
            linspace(eps_min, eps_max, n_eps_pts)
            if not decreasing
            else linspace(eps_max, eps_min, n_eps_pts)
        )
    else:
        epsilons = (
            logspace(log10(eps_min), log10(eps_max), n_eps_pts)
            if not decreasing
            else logspace(log10(eps_max), log10(eps_min), n_eps_pts)
        )
    out_list = [empty(n_eps_pts) for _ in range(n_observables)]

    copy_f_hat_kwargs = f_hat_kwargs.copy()

    old_initial_cond = initial_cond
    for idx, eps in enumerate(epsilons):
        copy_f_hat_kwargs.update({"percentage": eps})

        m, q, V = fixed_point_finder(
            f_func, f_hat_func, old_initial_cond, f_kwargs, copy_f_hat_kwargs
        )

        old_initial_cond = tuple([m, q, V])

        for jdx, (f, f_args, update_flag) in enumerate(zip(funs, funs_args, update_funs_args)):
            if update_flag:
                f_args.update({"percentage": float(eps)})
                out_list[jdx][idx] = f(m, q, V, **f_args)
            else:
                out_list[jdx][idx] = f(m, q, V, **f_args)

    if decreasing:
        epsilons = epsilons[::-1]
        for idx, obs_vals in enumerate(out_list):
            out_list[idx] = obs_vals[::-1]

    return epsilons, out_list


def sweep_eps_optimal_lambda_fixed_point(
    f_func,
    f_hat_func,
    eps_min: float,
    eps_max: float,
    n_eps_pts: int,
    inital_guess_lambda: float,
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
    decreasing=False,
    linear=False,
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

    if eps_min > eps_max:
        raise ValueError(
            "eps_min should be smaller than eps_max, in this case are {:f} and {:f}".format(
                eps_min, eps_max
            )
        )

    n_observables = len(funs)
    if linear:
        epsilons = (
            linspace(eps_min, eps_max, n_eps_pts)
            if not decreasing
            else linspace(eps_max, eps_min, n_eps_pts)
        )
    else:
        epsilons = (
            logspace(log10(eps_min), log10(eps_max), n_eps_pts)
            if not decreasing
            else logspace(log10(eps_max), log10(eps_min), n_eps_pts)
        )
    f_min_vals = empty(n_eps_pts)
    reg_params_opt = empty(n_eps_pts)
    funs_values = [empty(n_eps_pts) for _ in range(n_observables)]

    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_lambda
    for idx, eps in enumerate(epsilons):
        copy_f_hat_kwargs.update({"percentage": eps})
        copy_f_kwargs.update({"reg_param": old_reg_param_opt})

        if update_f_min_args:
            f_min_args.update({"percentage": float(eps)})

        for jdx, update_flag in enumerate(update_funs_args):
            if update_flag:
                copy_funs_args[jdx].update({"percentage": float(eps)})

        (
            f_min_vals[idx],
            reg_params_opt[idx],
            (m, q, V),
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
        old_initial_cond_fpe = (m, q, V)

        for jdx in range(n_observables):
            funs_values[jdx][idx] = out_values[jdx]

    if decreasing:
        epsilons = epsilons[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return epsilons, f_min_vals, reg_params_opt, funs_values


def sweep_eps_optimal_lambda_hub_param_fixed_point(
    f_func,
    f_hat_func,
    eps_min: float,
    eps_max: float,
    n_eps_pts: int,
    inital_guess_params: tuple[float, float],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[list()],
    update_funs_args=None,
    f_min=estimation_error,
    f_min_args={},
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
    decreasing=False,
    update_f_min_args=False,
    linear=False,
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

    if eps_min > eps_max:
        raise ValueError(
            "eps_min should be smaller than eps_max, in this case are {:f} and {:f}".format(
                eps_min, eps_max
            )
        )

    n_observables = len(funs)
    if linear:
        epsilons = (
            linspace(eps_min, eps_max, n_eps_pts)
            if not decreasing
            else linspace(eps_max, eps_min, n_eps_pts)
        )
    else:
        epsilons = (
            logspace(log10(eps_min), log10(eps_max), n_eps_pts)
            if not decreasing
            else logspace(log10(eps_max), log10(eps_min), n_eps_pts)
        )
    f_min_vals = empty(n_eps_pts)
    reg_params_opt = empty(n_eps_pts)
    hub_params_opt = empty(n_eps_pts)
    funs_values = [empty(n_eps_pts) for _ in range(n_observables)]

    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_params[0]
    old_hub_param_opt = inital_guess_params[1]
    for idx, eps in enumerate(epsilons):
        copy_f_hat_kwargs.update({"percentage": eps, "a": old_hub_param_opt})
        copy_f_kwargs.update({"reg_param": old_reg_param_opt})

        if update_f_min_args:
            f_min_args.update({"percentage": float(eps)})

        for jdx, update_flag in enumerate(update_funs_args):
            if update_flag:
                copy_funs_args[jdx].update({"percentage": float(eps)})

        (
            f_min_vals[idx],
            (reg_params_opt[idx], hub_params_opt[idx]),
            (m, q, V),
            out_values,
        ) = find_optimal_reg_and_huber_parameter_function(
            f_func,
            f_hat_func,
            copy_f_kwargs,
            copy_f_hat_kwargs,
            (old_reg_param_opt, old_hub_param_opt),
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
        old_initial_cond_fpe = (m, q, V)

        for jdx in range(n_observables):
            funs_values[jdx][idx] = out_values[jdx]

    if decreasing:
        epsilons = epsilons[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        hub_params_opt = hub_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return epsilons, f_min_vals, (reg_params_opt, hub_params_opt), funs_values
