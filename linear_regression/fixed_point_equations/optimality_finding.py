from numpy import empty
from numba import njit

from scipy.optimize import minimize
from ..utils.errors import MinimizationError
from .fpeqs import fixed_point_finder
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM, XATOL, FATOL
from ..aux_functions.misc import estimation_error


# --------------------------------


def find_optimal_reg_param_function(
    f_func,
    f_hat_func,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_guess_reg_param: float,
    initial_cond_fpe: tuple[float, float, float],
    funs=[estimation_error],
    funs_args=[list()],
    f_min=estimation_error,
    f_min_args={},
    min_reg_param=SMALLEST_REG_PARAM,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )
    n_observables = len(funs)
    copy_f_kwargs = f_kwargs.copy()

    def minimize_fun(reg_param):
        copy_f_kwargs.update({"reg_param": float(reg_param)})
        print("\t\tλ = {:.5f}".format(float(reg_param)))
        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=f_hat_kwargs,
        )
        return f_min(m, q, V, **f_min_args)

    bnds = [(min_reg_param, None)]
    obj = minimize(
        minimize_fun,
        x0=initial_guess_reg_param,
        method="Nelder-Mead",
        bounds=bnds,
        options={"xatol": XATOL, "fatol": FATOL},
    )

    if obj.success:
        fun_min_val = obj.fun
        reg_param_opt = obj.x

        copy_f_kwargs.update({"reg_param": float(reg_param_opt)})
        out_values = empty(n_observables)
        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=f_hat_kwargs,
        )

        for idx, (f, f_args) in enumerate(zip(funs, funs_args)):
            out_values[idx] = f(m, q, V, **f_args)  # Beware of the ** instead of * in the f_args

        return fun_min_val, reg_param_opt, (m, q, V), out_values
    else:
        raise MinimizationError("find_optimal_reg_param_function", initial_guess_reg_param)


def find_optimal_reg_and_huber_parameter_function(
    f_func,
    f_hat_func,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_guess_reg_and_huber_param: tuple[float, float],
    initial_cond_fpe: tuple[float, float, float],
    funs=[estimation_error],
    funs_args=[list()],
    f_min=estimation_error,
    f_min_args={},
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )
    n_observables = len(funs)
    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()

    def minimize_fun(x):
        copy_f_kwargs.update({"reg_param": x[0]})
        copy_f_hat_kwargs.update({"a": x[1]})

        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=copy_f_hat_kwargs,
        )
        return f_min(m, q, V, **f_min_args)

    bnds = [(min_reg_param, None), (min_huber_param, None)]
    obj = minimize(
        minimize_fun,
        x0=list(initial_guess_reg_and_huber_param),
        method="Nelder-Mead",
        bounds=bnds,
        options={
            "xatol": XATOL,
            "fatol": FATOL,
            "adaptive": True,
        },
    )
    if obj.success:
        fun_min_val = obj.fun
        reg_param_opt, a_opt = obj.x

        copy_f_kwargs.update({"reg_param": reg_param_opt})
        copy_f_hat_kwargs.update({"a": a_opt})
        out_values = empty(n_observables)
        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=copy_f_hat_kwargs,
        )

        for idx, (f, f_args) in enumerate(zip(funs, funs_args)):
            out_values[idx] = f(m, q, V, **f_args) # Beware of the ** instead of * in the f_args

        return fun_min_val, (reg_param_opt, a_opt), (m, q, V), out_values
    else:
        raise MinimizationError(
            "find_optimal_reg_and_huber_parameter_function", initial_guess_reg_and_huber_param
        )
