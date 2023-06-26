from numba import njit
import numpy as np
from math import sqrt
from scipy.integrate import dblquad
from scipy.optimize import root_scalar
from ..utils.integration_utils import (
    divide_integration_borders_multiple_grid,
    find_integration_borders_square,
    domains_sep_hyperboles_inside,
    domains_sep_hyperboles_above,
)
from ..aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
    f_out_Bayes_decorrelated_noise,
    f_out_hinge,
    Df_out_hinge,
)

N_GRID = 5


@njit(error_model="numpy", fastmath=True)
def m_integral_Hinge_decorrelated_noise(y, xi, q, m, sigma, delta_in, delta_out, percentage, beta):
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_decorrelated_noise(y, np.sqrt(eta) * xi, 1 - eta, delta_in, delta_out, percentage, beta)
        * f_out_Bayes_decorrelated_noise(y, np.sqrt(eta) * xi, 1 - eta, delta_in, delta_out, percentage, beta)
        * f_out_hinge(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Hinge_decorrelated_noise(y, xi, q, m, sigma, delta_in, delta_out, percentage, beta):
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_decorrelated_noise(y, np.sqrt(eta) * xi, 1 - eta, delta_in, delta_out, percentage, beta)
        * (f_out_hinge(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Hinge_decorrelated_noise(y, xi, q, m, sigma, delta_in, delta_out, percentage, beta):
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_decorrelated_noise(y, np.sqrt(eta) * xi, 1 - eta, delta_in, delta_out, percentage, beta)
        * Df_out_hinge(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy")
def hyperbole(x, const):
    return const / x


def var_hat_func_Hinge_num_decorrelated_noise(m, q, sigma, alpha, delta_in, delta_out, percentage, beta):
    # print("m = {:.5f}\nq = {:.5f}\nsigma = {:.5f}".format(m, q, sigma))

    borders = find_integration_borders_square(
        m_integral_Hinge_decorrelated_noise,
        1 * np.sqrt((1 + max(delta_in, delta_out))),
        1.0,
        args=(q, m, sigma, delta_in, delta_out, percentage, beta),
    )

    domain_xi_1, domain_y_1 = domains_sep_hyperboles_inside(
        borders, hyperbole, hyperbole, {"const": (1.0 - sigma) / sqrt(q)}, {"const": 1.0 / sqrt(q)}
    )
    domain_xi_2, domain_y_2 = domains_sep_hyperboles_above(borders, hyperbole, {"const": (1.0 - sigma) / sqrt(q)})

    domain_xi_m_hat, domain_y_m_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
    # print("m integral")
    integral_value_m_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_m_hat, domain_y_m_hat):
        # print("xi funs ", xi_funs, " y funs ", y_funs)
        integral_value_m_hat += dblquad(
            m_integral_Hinge_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]
    m_hat = alpha * integral_value_m_hat

    # --- q hat itegral ---
    # borders = find_integration_borders_square(
    #     q_integral_Hinge_decorrelated_noise,
    #     1.0 * np.sqrt((1 + max(delta_in, delta_out))),
    #     1.0,
    #     args=(q, m, sigma, delta_in, delta_out, percentage, beta),
    # )

    # domain_xi_1, domain_y_1 = domains_sep_hyperboles_inside(
    #     borders, hyperbole, hyperbole, {"const": (1.0 - sigma) / sqrt(q)}, {"const": 1.0 / sqrt(q)}
    # )
    # domain_xi_2, domain_y_2 = domains_sep_hyperboles_above(borders, hyperbole, {"const": (1.0 - sigma) / sqrt(q)})
    domain_xi_q_hat, domain_y_q_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
    # print("q integral")
    integral_value_q_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_q_hat, domain_y_q_hat):
        integral_value_q_hat += dblquad(
            q_integral_Hinge_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]
    q_hat = alpha * integral_value_q_hat

    # --- Sigma hat integral ---
    # borders = find_integration_borders_square(
    #     sigma_integral_Hinge_decorrelated_noise,
    #     1.0 * np.sqrt((1 + max(delta_in, delta_out))),
    #     1.0,
    #     args=(q, m, sigma, delta_in, delta_out, percentage, beta),
    # )

    # domain_xi_sigma_hat, domain_y_sigma_hat = domains_sep_hyperboles_inside(
    #     borders, hyperbole, hyperbole, {"const": (1.0 - sigma) / sqrt(q)}, {"const": 1.0 / sqrt(q)}
    # )
    domain_xi_sigma_hat, domain_y_sigma_hat = domain_xi_2, domain_y_2
    # print("sigma integral")
    integral_value_sigma_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_sigma_hat, domain_y_sigma_hat):
        integral_value_sigma_hat += dblquad(
            sigma_integral_Hinge_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]
    sigma_hat = -alpha * integral_value_sigma_hat

    return m_hat, q_hat, sigma_hat
