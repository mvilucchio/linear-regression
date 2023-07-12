from numba import njit
import numpy as np
from scipy.integrate import dblquad
from ...utils.integration_utils import (
    divide_integration_borders_multiple_grid,
    find_integration_borders_square,
)
from ...aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
    f_out_Bayes_decorrelated_noise,
    Z_out_Bayes_single_noise,
    f_out_Bayes_single_noise,
)

N_GRID = 5


# @njit(error_model="numpy", fastmath=True)
def m_integral_Logistic_single_noise(y, xi, q, m, sigma, delta):
    raise NotImplementedError
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * f_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * f_out_Hinge(y, np.sqrt(q) * xi, sigma)
    )


# @njit(error_model="numpy", fastmath=True)
def q_integral_Logistic_single_noise(y, xi, q, m, sigma, delta):
    raise NotImplementedError
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * (f_out_Hinge(y, np.sqrt(q) * xi, sigma) ** 2)
    )


# @njit(error_model="numpy", fastmath=True)
def sigma_integral_Logistic_single_noise(y, xi, q, m, sigma, delta):
    raise NotImplementedError
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * Df_out_Hinge(y, np.sqrt(q) * xi, sigma)
    )


# -----------------------------------


def f_hat_Logistic_decorrelated_noise(m, q, sigma, alpha, delta_in, delta_out, percentage, beta):
    raise NotImplementedError
    borders = find_integration_borders_square(
        m_integral_Logistic_single_noise,
        3 * np.sqrt((1 + max(delta_in, delta_out))),
        3.0,
        args=(q, m, sigma, delta_in, delta_out, percentage, beta),
    )

    domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=N_GRID)

    print("m integral")
    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Logistic_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]
    m_hat = alpha * integral_value

    borders = find_integration_borders_square(
        q_integral_Logistic_single_noise,
        3 * np.sqrt((1 + max(delta_in, delta_out))),
        3.0,
        args=(q, m, sigma, delta_in, delta_out, percentage, beta),
    )

    domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=N_GRID)

    print("q integral")
    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Logistic_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]
    q_hat = alpha * integral_value

    borders = find_integration_borders_square(
        sigma_integral_Logistic_single_noise,
        3 * np.sqrt((1 + max(delta_in, delta_out))),
        3.0,
        args=(q, m, sigma, delta_in, delta_out, percentage, beta),
    )

    domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=N_GRID)

    print("sigma integral")
    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Logistic_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]
    Σ_hat = -alpha * integral_value

    return m_hat, q_hat, Σ_hat
