from numpy import logspace, linspace, empty, nan
from math import log10
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import estimation_error
from ..fixed_point_equations.fpeqs import fixed_point_finder


def sweep_reg_param_fixed_point(
    var_func,
    var_hat_func,
    reg_param_min: float,
    reg_param_max: float,
    n_reg_param_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond=(0.6, 0.01, 0.9),
    funs=[estimation_error],
    funs_args=[list()],
    update_funs_args=None,
    linear=False,
    decreasing=True,
):
    if update_funs_args is None:
        update_funs_args = [False] * len(funs)

    n_funs = len(funs)
    n_funs_args = len(funs_args)
    n_update_funs_args = len(update_funs_args)

    if not (n_funs == n_funs_args == n_update_funs_args):
        raise ValueError(
            "funs, funs_args and update_funs_args should have the same length, in this case are {:d}, {:d} and {:d}".format(
                n_funs, n_funs_args, n_update_funs_args
            )
        )

    if reg_param_min > reg_param_max:
        raise ValueError(
            "reg_param_min should be smaller than reg_param_max, in this case are {:f} and {:f}".format(
                reg_param_min, reg_param_max
            )
        )

    if not linear:
        if reg_param_min <= 0.0 or reg_param_max <= 0.0:
            raise ValueError(
                "reg_param_min and reg_param_max should be positive in this case are {:f} and {:f}".format(
                    reg_param_min, reg_param_max
                )
            )

    n_observables = len(funs)
    if linear:
        reg_params = (
            linspace(reg_param_min, reg_param_max, n_reg_param_pts)
            if not decreasing
            else linspace(reg_param_max, reg_param_min, n_reg_param_pts)
        )
    else:
        reg_params = (
            logspace(log10(reg_param_min), log10(reg_param_max), n_reg_param_pts)
            if not decreasing
            else logspace(log10(reg_param_max), log10(reg_param_min), n_reg_param_pts)
        )
    out_list = [empty(n_reg_param_pts) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_funs_args = funs_args.copy()

    not_converged_flag = False
    old_initial_cond = initial_cond
    for idx, reg_param in enumerate(reg_params):
        copy_var_func_kwargs.update({"reg_param": reg_param})
        try:
            if not_converged_flag:
                for jdx, (f, f_args) in enumerate(zip(funs, funs_args)):
                    out_list[jdx][idx] = nan
            else:
                m, q, sigma = fixed_point_finder(
                    var_func,
                    var_hat_func,
                    old_initial_cond,
                    copy_var_func_kwargs,
                    var_hat_func_kwargs,
                )

                old_initial_cond = tuple([m, q, sigma])

                for jdx, (f, f_args, update_flag) in enumerate(zip(funs, funs_args, update_funs_args)):
                    if update_flag:
                        copy_funs_args[jdx].update({"reg_param" : reg_param})
                        out_list[jdx][idx] = f(m, q, sigma, **copy_funs_args[jdx])
                    else:
                        out_list[jdx][idx] = f(m, q, sigma, **f_args)

        except ConvergenceError:
            not_converged_flag = True
            for jdx, (f, f_args) in enumerate(zip(funs, funs_args)):
                out_list[jdx][idx] = nan

    if decreasing:
        reg_params = reg_params[::-1]
        for idx, obs_vals in enumerate(out_list):
            out_list[idx] = obs_vals[::-1]

    return reg_params, out_list
