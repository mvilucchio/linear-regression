# import numpy as np
from numpy import pi, ndarray, mean, abs, amax, zeros, ones
from math import exp, sqrt
from numpy.random import random
from numba import njit
from ..regression_numerics import TOL_GAMP, BLEND_GAMP, MAX_ITER_GAMP
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import damped_update


# @njit
def GAMP_algorithm_unsimplified(
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    ys: ndarray,
    xs: ndarray,
    f_w_args: tuple,
    f_out_args: tuple,
    init_w_hat,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
):
    n, d = xs.shape

    # random init
    w_hat_t = init_w_hat
    c_w_t = zeros(d)
    f_out_t_1 = zeros(n)

    F = xs / sqrt(d)
    F2 = F**2

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = (F @ w_hat_t) - (V_t * f_out_t_1)

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = (f_out_t @ F) + (Lambda_t * w_hat_t)

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = mean(abs(new_w_hat_t - w_hat_t))

        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)
        f_out_t_1 = f_out_t

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("GAMP_algorithm", iter_nb)

    return w_hat_t


# @njit
def GAMP_unsimplified_iters(
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    ys: ndarray,
    xs: ndarray,
    f_w_args: tuple,
    f_out_args: tuple,
    init_w_hat,
    multiplier_c_w: float,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
):
    _, d = xs.shape

    F = xs / sqrt(d)
    F2 = F**2

    # start from the init 
    w_hat_t = init_w_hat
    c_w_t = multiplier_c_w * ones(d)

    V_t = F2 @ c_w_t
    omega_t = F @ w_hat_t
    f_out_t_1 = f_out(ys, omega_t, V_t, *f_out_args)

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = (F @ w_hat_t) - (V_t * f_out_t_1)

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = (f_out_t @ F) + (Lambda_t * w_hat_t)

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = mean(abs(new_w_hat_t - w_hat_t))

        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)
        # update the onsager term with damping?
        f_out_t_1 = f_out_t

        iter_nb += 1
        if iter_nb > max_iter:
            return w_hat_t, iter_nb

    return w_hat_t, iter_nb


def GAMP_algorithm_unsimplified_mod(
    multiplier_c_w: float,
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    ys: ndarray,
    xs: ndarray,
    f_w_args: tuple,
    f_out_args: tuple,
    init_w_hat,
    ground_truth,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
):
    n, d = xs.shape

    F = xs / sqrt(d)
    F2 = F**2

    # random init
    w_hat_t = init_w_hat  # 0.1 * random(d) + 0.95
    c_w_t = multiplier_c_w * ones(d)  # zeros(d) # 0.1 * random(d) + 0.01
    f_out_t_1 = zeros(n)  # 0.5 * random(n) + 0.001

    V_t = F2 @ c_w_t
    omega_t = F @ w_hat_t

    f_out_t_1 = f_out(ys, omega_t, V_t, *f_out_args)

    print(
        f"q_init = {mean(w_hat_t**2)}, m_init = {mean(w_hat_t * ground_truth)}, q_fixed = {f_w_args[0]}"
    )

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = (F @ w_hat_t) - (V_t * f_out_t_1)

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = (f_out_t @ F) + (Lambda_t * w_hat_t)

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = mean(abs(new_w_hat_t - w_hat_t))

        # there is somehting strange here since the error is lower than tolerance
        if iter_nb % 10 == 0:
            print(
                f"err = {err}, q = {mean(new_w_hat_t**2)}, m = {mean(new_w_hat_t * ground_truth)}"
            )

        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)
        # update of the Onsager term
        f_out_t_1 = f_out_t

        iter_nb += 1
        if iter_nb > max_iter:
            return w_hat_t  # raise ConvergenceError("GAMP_algorithm", iter_nb)

    return w_hat_t


def GAMP_algorithm_unsimplified_mod_2(
    multiplier_c_w: float,
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    ys: ndarray,
    xs: ndarray,
    f_w_args: tuple,
    f_out_args: tuple,
    init_w_hat,
    ground_truth,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
):
    n, d = xs.shape

    F = xs / sqrt(d)
    F2 = F**2

    # random init
    w_hat_t = init_w_hat  # 0.1 * random(d) + 0.95
    c_w_t = multiplier_c_w * ones(d)  # zeros(d) # 0.1 * random(d) + 0.01
    f_out_t_1 = zeros(n)  # 0.5 * random(n) + 0.001

    V_t = F2 @ c_w_t
    omega_t = F @ w_hat_t

    f_out_t_1 = f_out(ys, omega_t, V_t, *f_out_args)

    q_list = [mean(w_hat_t**2)]
    m_list = [mean(w_hat_t * ground_truth)]
    previous_dot_list = [1.0]
    eps_list = [1.0]
    # q_list.append()
    # m_list.append()
    # previous_dot_list.append(1.0)

    print(
        f"q_init = {mean(w_hat_t**2)}, m_init = {mean(w_hat_t * ground_truth)}, q_fixed = {f_w_args[0]}"
    )

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = (F @ w_hat_t) - (V_t * f_out_t_1)

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = (f_out_t @ F) + (Lambda_t * w_hat_t)

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = mean(abs(new_w_hat_t - w_hat_t) ** 2)

        if iter_nb % 1 == 0 and iter_nb > 0:
            # print(f"err = {err}, q = {mean(new_w_hat_t**2)}, m = {mean(new_w_hat_t * ground_truth)}")
            q_list.append(mean(new_w_hat_t**2))
            m_list.append(mean(new_w_hat_t * ground_truth))
            previous_dot_list.append(mean((new_w_hat_t - w_hat_t) ** 2))
            eps_list.append(mean((new_w_hat_t - init_w_hat) ** 2 / (w_hat_t - init_w_hat) ** 2))

        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)
        # update of the Onsager term
        f_out_t_1 = f_out_t

        iter_nb += 1
        if iter_nb > max_iter:
            return w_hat_t, q_list, m_list, previous_dot_list, eps_list

    return w_hat_t, q_list, m_list, previous_dot_list, eps_list


def GAMP_algorithm_unsimplified_mod_3(
    multiplier_c_w: float,
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    ys: ndarray,
    xs: ndarray,
    f_w_args: tuple,
    f_out_args: tuple,
    init_w_hat,
    ground_truth,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
):
    n, d = xs.shape

    F = xs / sqrt(d)
    F2 = F**2

    # random init
    w_hat_t = init_w_hat
    c_w_t = multiplier_c_w * ones(d)
    f_out_t_1 = zeros(n)

    V_t = F2 @ c_w_t
    omega_t = F @ w_hat_t

    f_out_t_1 = f_out(ys, omega_t, V_t, *f_out_args)

    print(
        f"q_init = {mean(w_hat_t**2)}, m_init = {mean(w_hat_t * ground_truth)}, q_fixed = {f_w_args[0]}"
    )

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = (F @ w_hat_t) - (V_t * f_out_t_1)

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = (f_out_t @ F) + (Lambda_t * w_hat_t)

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = mean(abs(new_w_hat_t - w_hat_t))

        if iter_nb % 500 == 0:
            print(
                f"err = {err}, q = {mean(new_w_hat_t**2)}, m = {mean(new_w_hat_t * ground_truth)}"
            )

        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)
        # update of the Onsager term
        f_out_t_1 = f_out_t

        iter_nb += 1
        if iter_nb > max_iter:
            return w_hat_t, iter_nb

    return w_hat_t, iter_nb


def GAMP_algorithm_unsimplified_mod_4(
    multiplier_c_w: float,
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    ys: ndarray,
    xs: ndarray,
    f_w_args: tuple,
    f_out_args: tuple,
    init_w_hat,
    ground_truth,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
    each_how_many=10
):
    n, d = xs.shape

    F = xs / sqrt(d)
    F2 = F**2

    # random init
    w_hat_t = init_w_hat
    c_w_t = multiplier_c_w * ones(d)
    f_out_t_1 = zeros(n)

    V_t = F2 @ c_w_t
    omega_t = F @ w_hat_t

    f_out_t_1 = f_out(ys, omega_t, V_t, *f_out_args)
    previous_ones = list()

    print(
        f"q_init = {mean(w_hat_t**2)}, m_init = {mean(w_hat_t * ground_truth)}, q_fixed = {f_w_args[0]}"
    )

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = (F @ w_hat_t) - (V_t * f_out_t_1)

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = (f_out_t @ F) + (Lambda_t * w_hat_t)

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = mean(abs(new_w_hat_t - w_hat_t))

        if iter_nb % each_how_many == 0 or iter_nb % each_how_many == 1:
            previous_ones.append(new_w_hat_t)
            print(
                f"err = {err}, q = {mean(new_w_hat_t**2)}, m = {mean(new_w_hat_t * ground_truth)}"
            )

        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)
        # update of the Onsager term
        f_out_t_1 = f_out_t

        iter_nb += 1
        if iter_nb > max_iter:
            return w_hat_t, iter_nb, previous_ones

    return w_hat_t, iter_nb, previous_ones
