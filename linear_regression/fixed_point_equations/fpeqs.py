from typing import Tuple
from numba import njit
from ..fixed_point_equations import BLEND_FPE, TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import damped_update


def fixed_point_finder(
    var_func,
    var_hat_func,
    initial_condition: Tuple[float, float, float],
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
):
    m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, **var_hat_func_kwargs)
        new_m, new_q, new_sigma = var_func(m_hat, q_hat, sigma_hat, **var_func_kwargs)

        err = max([abs(new_m - m), abs(new_q - q), abs(new_sigma - sigma)])

        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        sigma = damped_update(new_sigma, sigma, BLEND_FPE)

        print("\t\tm = {:.5f} q = {:.5f} Σ = {:.5f}".format(m, q, sigma))

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)

    return m, q, sigma


def plateau_fixed_point_finder(
        x_next_func,
        inital_condition: float,
        x_next_func_kwargs: dict,
        abs_tol: float = TOL_FPE,
        min_iter: int = MIN_ITER_FPE,
        max_iter: int = MAX_ITER_FPE,
):
    x = inital_condition
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        x_next = x_next_func(x, **x_next_func_kwargs)
        err = abs(x_next - x)
        x = x_next
        if iter_nb > max_iter:
            raise ConvergenceError("plateau_fixed_point_finder", iter_nb)
    return x
