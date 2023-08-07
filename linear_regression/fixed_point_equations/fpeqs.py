from typing import Tuple
from numba import njit
from ..fixed_point_equations import BLEND_FPE, TOL_FPE, REL_TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import damped_update
import numpy as np

def fixed_point_finder(
    f_func,
    f_hat_func,
    initial_condition: Tuple[float, float, float],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    abs_tol: float = TOL_FPE,
    # rel_tol: float = REL_TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
):
    m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        m_hat, q_hat, Σ_hat = f_hat_func(m, q, sigma, **f_hat_kwargs)
        new_m, new_q, new_sigma = f_func(m_hat, q_hat, Σ_hat, **f_kwargs)

        err = max([abs(new_m - m), abs(new_q - q), abs(new_sigma - sigma)])
        # err = max([abs(new_m - m), abs(new_q - q)])

        # print(
        #     "\t\t\tm = {:.1e} Δm = {:.1e} q = {:.1e} Δq = {:.1e} Σ = {:.1e} ΔΣ = {:.1e} ".format(
        #         m, abs(new_m - m), q, abs(new_q - q), sigma, abs(new_sigma - sigma)
        #     )
        # )

        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        sigma = damped_update(new_sigma, sigma, BLEND_FPE)

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)

    return m, q, sigma


def fixed_point_finder_loser(
    f_func,
    f_hat_func,
    initial_condition: Tuple[float, float, float],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    abs_tol: float = TOL_FPE,
    rel_tol: float = REL_TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
    control_variate: Tuple[bool, bool, bool] = (True, True, True),
):
    m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        m_hat, q_hat, Σ_hat = f_hat_func(m, q, sigma, **f_hat_kwargs)
        new_m, new_q, new_sigma = f_func(m_hat, q_hat, Σ_hat, **f_kwargs)

        errs = list()
        if control_variate[0]:
            errs.append(abs(new_m - m))
        if control_variate[1]:
            errs.append(abs(new_q - q))
        if control_variate[2]:
            errs.append(abs(new_sigma - sigma))
            
        err = max(errs)
    
        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        sigma = damped_update(new_sigma, sigma, BLEND_FPE)

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)
        
    # print(
    #     "\t\t\tm = {:.1e} Δm = {:.1e} q = {:.1e} Δq = {:.1e} Σ = {:.1e} ΔΣ = {:.1e} ".format(
    #         m, abs(new_m - m), q, abs(new_q - q), sigma, abs(new_sigma - sigma)
    #     )
    # )
    
    return m, q, sigma


# implementation of the same as fixed_point_finder but using Anderson acceleration for the fixed point equation
# def anderson_fixed_point_finder(
#     f_func,
#     f_hat_func,
#     initial_condition: Tuple[float, float, float],
#     f_kwargs: dict,
#     f_hat_kwargs: dict,
#     abs_tol: float = TOL_FPE,
#     min_iter: int = MIN_ITER_FPE,
#     max_iter: int = MAX_ITER_FPE,
# ):
#     m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
#     err = 1.0
#     iter_nb = 0
#     for idx in range(max_iter):


#         if err < abs_tol:
#             break
#         if iter_nb > min_iter:
#             raise ConvergenceError("anderson_fixed_point_finder", iter_nb)

#     return m, q, sigma


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


# def gradient_at_fp(jacobian_at_point : callable, fp : Tuple[float, float, float]):
#     # the output gradient is an array of size 6 for each of the order parameters
#     g = np.empty(6)

#     J = jacobian_at_point()
