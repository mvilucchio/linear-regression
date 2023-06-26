from ..aux_functions.misc import estimation_error
from numpy import logspace, empty, empty_like, meshgrid, nditer
from math import log10
from typing import Tuple
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM
from ..fixed_point_equations.optimality_finding import (
    find_optimal_reg_param_function,
    find_optimal_reg_and_huber_parameter_function,
)


def sweep_eps_delta_out_optimal_lambda_fixed_point(
    var_func,
    var_hat_func,
    eps_min: float,
    eps_max: float,
    n_eps_pts: int,
    delta_out_min: float,
    delta_out_max: float,
    n_delta_out_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_guess_reg_param: float,
    initial_cond_fpe: Tuple[float, float, float],
    funs=[estimation_error],
    funs_args=[{}],
    update_funs_args=None,
    f_min=estimation_error,
    f_min_args={},
    update_f_min_args=True,
    min_reg_param=SMALLEST_REG_PARAM,
    decreasing=[False, True],
):
    # for a meshgrid application we have that the first index is connected to increase in y while the second index is connected to the increase in x
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

    if len(decreasing) != 2:
        raise ValueError("The length of decreasing should be 2, in this case is {:d}".format(len(decreasing)))

    if eps_min <= 0.0 or eps_max <= 0.0:
        raise ValueError(
            "eps_min and eps_max should be positive, in this case are {:f} and {:f}".format(eps_min, eps_max)
        )

    if delta_out_min <= 0.0 or delta_out_max <= 0.0:
        raise ValueError(
            "delta_out_min and delta_out_max should be positive, in this case are {:f} and {:f}".format(
                delta_out_min, delta_out_max
            )
        )

    if n_eps_pts <= 0 or n_delta_out_pts <= 0:
        raise ValueError(
            "n_eps_pts and n_delta_out_pts should be positive, in this case are {:d} and {:d}".format(
                n_eps_pts, n_delta_out_pts
            )
        )

    if not isinstance(n_eps_pts, int) or not isinstance(n_delta_out_pts, int):
        raise TypeError("n_eps_pts and n_delta_out_pts should be integers")

    if not isinstance(update_f_min_args, bool):
        raise TypeError("update_f_min_args should be a boolean")

    epsilons = (
        logspace(log10(eps_min), log10(eps_max), n_eps_pts)
        if not decreasing[0]
        else logspace(log10(eps_max), log10(eps_min), n_eps_pts)
    )
    delta_outs = (
        logspace(log10(delta_out_min), log10(delta_out_max), n_delta_out_pts)
        if not decreasing[1]
        else logspace(log10(delta_out_max), log10(delta_out_min), n_delta_out_pts)
    )

    epseps, deltadelta = meshgrid(epsilons, delta_outs)

    n_observables = len(funs)
    f_min_vals = empty_like(epseps) # empty((n_delta_out_pts, n_eps_pts))
    reg_params_opt = empty_like(epseps) 
    funs_values = [empty_like(epseps)  for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_reg_param_opt_begin_delta_sweep = initial_guess_reg_param
    old_initial_cond_fpe_begin_delta_sweep = initial_cond_fpe

    it = nditer(epseps, flags=["multi_index"], order='F')
    while not it.finished:
    # for idx, epseps[it.multi_index] in enumerate(epsilons):
        if it.multi_index[0] == 0:
            old_reg_param_opt = old_reg_param_opt_begin_delta_sweep
            old_initial_cond_fpe = old_initial_cond_fpe_begin_delta_sweep

        # print(f"index: {it.multi_index} epsilon: {epseps[it.multi_index]} delta_out: {deltadelta[it.multi_index]}")

        # for jdx, delta_out in enumerate(delta_outs):
        copy_var_func_kwargs.update({"reg_param": old_reg_param_opt})
        copy_var_hat_func_kwargs.update({"percentage": epseps[it.multi_index], "delta_out": deltadelta[it.multi_index]})
        # print(copy_var_hat_func_kwargs)

        if update_f_min_args:
            f_min_args.update({"percentage": epseps[it.multi_index], "delta_out": deltadelta[it.multi_index]})

        for kdx in range(n_funs):
            if update_funs_args[kdx]:
                copy_funs_args[kdx].update({"percentage": epseps[it.multi_index], "delta_out": deltadelta[it.multi_index]})

        (
            f_min_vals[it.multi_index],
            reg_params_opt[it.multi_index],
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_param_function(
            var_func,
            var_hat_func,
            copy_var_func_kwargs,
            copy_var_hat_func_kwargs,
            initial_guess_reg_param,
            old_initial_cond_fpe,
            funs=funs,
            funs_args=copy_funs_args,
            f_min=f_min,
            f_min_args=f_min_args,
            min_reg_param=min_reg_param,
        )

        old_reg_param_opt = reg_params_opt[it.multi_index]
        old_initial_cond_fpe = (m, q, sigma)

        if it.multi_index[0] == 0:
            old_reg_param_opt_begin_delta_sweep = reg_params_opt[it.multi_index]
            old_initial_cond_fpe_begin_delta_sweep = (m, q, sigma)

        for kdx in range(n_observables):
            funs_values[kdx][it.multi_index] = out_values[kdx]
        
        it.iternext()
    # if decreasing[0]:
    #     print("decreasing epsilons")
    #     epsilons = epsilons[::-1]
    #     f_min_vals = f_min_vals[::-1, :]
    #     reg_params_opt = reg_params_opt[::-1, :]
    #     for kdx in range(n_observables):
    #         funs_values[kdx] = funs_values[kdx][::-1, :]

    # if decreasing[1]:
    #     print("decreasing delta_outs")
    #     delta_outs = delta_outs[::-1]
    #     f_min_vals = f_min_vals[:, ::-1]
    #     reg_params_opt = reg_params_opt[:, ::-1]
    #     for kdx in range(n_observables):
    #         funs_values[kdx] = funs_values[kdx][:, ::-1]

    return epseps, deltadelta, f_min_vals, reg_params_opt, funs_values


def sweep_eps_delta_out_optimal_lambda_hub_param_fixed_point(
    var_func,
    var_hat_func,
    eps_min: float,
    eps_max: float,
    n_eps_pts: int,
    delta_out_min: float,
    delta_out_max: float,
    n_delta_out_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_guess_reg_param: float,
    initial_guess_huber_param: float,
    initial_cond_fpe: Tuple[float, float, float],
    funs=[estimation_error],
    funs_args=[{}],
    update_funs_args=None,
    f_min=estimation_error,
    f_min_args={},
    update_f_min_args=True,
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
    decreasing=[False, True],
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

    if len(decreasing) != 2:
        raise ValueError("The length of decreasing should be 2, in this case is {:d}".format(len(decreasing)))

    if eps_min <= 0.0 or eps_max <= 0.0:
        raise ValueError(
            "eps_min and eps_max should be positive, in this case are {:f} and {:f}".format(eps_min, eps_max)
        )

    if delta_out_min <= 0.0 or delta_out_max <= 0.0:
        raise ValueError(
            "delta_out_min and delta_out_max should be positive, in this case are {:f} and {:f}".format(
                delta_out_min, delta_out_max
            )
        )

    if n_eps_pts <= 0 or n_delta_out_pts <= 0:
        raise ValueError(
            "n_eps_pts and n_delta_out_pts should be positive, in this case are {:d} and {:d}".format(
                n_eps_pts, n_delta_out_pts
            )
        )

    if not isinstance(n_eps_pts, int) or not isinstance(n_delta_out_pts, int):
        raise TypeError("n_eps_pts and n_delta_out_pts should be integers")

    if not isinstance(update_f_min_args, bool):
        raise TypeError("update_f_min_args should be a boolean")

    epsilons = (
        logspace(log10(eps_min), log10(eps_max), n_eps_pts)
        if not decreasing[0]
        else logspace(log10(eps_max), log10(eps_min), n_eps_pts)
    )
    delta_outs = (
        logspace(log10(delta_out_min), log10(delta_out_max), n_delta_out_pts)
        if not decreasing[1]
        else logspace(log10(delta_out_max), log10(delta_out_min), n_delta_out_pts)
    )

    epseps, deltadelta = meshgrid(epsilons, delta_outs)

    n_observables = len(funs)
    f_min_vals = empty((n_delta_out_pts, n_eps_pts))
    reg_params_opt = empty((n_delta_out_pts, n_eps_pts))
    huber_params_opt = empty((n_delta_out_pts, n_eps_pts))
    funs_values = [empty((n_delta_out_pts, n_eps_pts)) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()
    copy_funs_args = funs_args.copy()

    old_reg_param_opt_begin_delta_sweep = initial_guess_reg_param
    old_huber_param_opt_begin_delta_sweep = initial_guess_huber_param
    old_initial_cond_fpe_begin_delta_sweep = initial_cond_fpe

    it = nditer(epseps, flags=["multi_index"], order='F')
    while not it.finished:
    # for idx, epseps[it.multi_index] in enumerate(epsilons):
        if it.multi_index[0] == 0:
            old_reg_param_opt = old_reg_param_opt_begin_delta_sweep
            old_huber_param_opt = old_huber_param_opt_begin_delta_sweep
            old_initial_cond_fpe = old_initial_cond_fpe_begin_delta_sweep
        # print(f"index: {it.multi_index} epsilon: {epseps[it.multi_index]:.2e} delta_out: {deltadelta[it.multi_index]:.2e}")

        # for jdx, delta_out in enumerate(delta_outs):
        copy_var_func_kwargs.update({"reg_param": old_reg_param_opt})
        copy_var_hat_func_kwargs.update({"percentage": epseps[it.multi_index], "delta_out": deltadelta[it.multi_index], "a": old_huber_param_opt})

        if update_f_min_args:
            f_min_args.update({"percentage": epseps[it.multi_index], "delta_out": deltadelta[it.multi_index]})

        # print("\t", f_min_args)

        for kdx in range(n_funs):
            if update_funs_args[kdx]:
                copy_funs_args[kdx].update({"percentage": epseps[it.multi_index], "delta_out": deltadelta[it.multi_index]})

        (
            f_min_vals[it.multi_index],
            (reg_params_opt[it.multi_index], huber_params_opt[it.multi_index]),
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_and_huber_parameter_function(
            var_func,
            var_hat_func,
            copy_var_func_kwargs,
            copy_var_hat_func_kwargs,
            (old_reg_param_opt, old_huber_param_opt),
            old_initial_cond_fpe,
            funs=funs,
            funs_args=copy_funs_args,
            f_min=f_min,
            f_min_args=f_min_args,
            min_reg_param=min_reg_param,
            min_huber_param=min_huber_param,
        )

        if it.multi_index[0] == 0:
            old_reg_param_opt_begin_delta_sweep = reg_params_opt[it.multi_index]
            old_huber_param_opt_begin_delta_sweep = huber_params_opt[it.multi_index]
            old_initial_cond_fpe_begin_delta_sweep = (m, q, sigma)

        old_reg_param_opt = reg_params_opt[it.multi_index]
        old_huber_param_opt = huber_params_opt[it.multi_index]
        old_initial_cond_fpe = (m, q, sigma)

        for kdx in range(n_observables):
            funs_values[kdx][it.multi_index] = out_values[kdx]

        it.iternext()
    # if decreasing[0]:
    #     print("decreasing epsilons")
    #     epsilons = epsilons[::-1]
    #     f_min_vals = f_min_vals[::-1, :]
    #     reg_params_opt = reg_params_opt[::-1, :]
    #     huber_params_opt = huber_params_opt[::-1, :]
    #     for kdx in range(n_observables):
    #         funs_values[kdx] = funs_values[kdx][::-1, :]

    # if decreasing[1]:
    #     print("decreasing delta_outs")
    #     delta_outs = delta_outs[::-1]
    #     f_min_vals = f_min_vals[:, ::-1]
    #     reg_params_opt = reg_params_opt[:, ::-1]
    #     huber_params_opt = huber_params_opt[:, ::-1]
    #     for kdx in range(n_observables):
    #         funs_values[kdx] = funs_values[kdx][:, ::-1]

    # if decreasing[0]:
    #     epsilons = epsilons[::-1]
    #     f_min_vals = f_min_vals[:, ::-1]
    #     reg_params_opt = reg_params_opt[:, ::-1]
    #     huber_params_opt = huber_params_opt[:, ::-1]
    #     for kdx in range(n_observables):
    #         funs_values[kdx] = funs_values[kdx][:, ::-1]

    # if decreasing[1]:
    #     delta_outs = delta_outs[::-1]
    #     f_min_vals = f_min_vals[::-1, :]
    #     reg_params_opt = reg_params_opt[::-1, :]
    #     reg_params_opt = reg_params_opt[::-1, :]
    #     for kdx in range(n_observables):
    #         funs_values[kdx] = funs_values[kdx][::-1, :]

    return epseps, deltadelta, f_min_vals, (reg_params_opt, huber_params_opt), funs_values
