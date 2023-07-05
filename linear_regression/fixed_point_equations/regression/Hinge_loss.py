from numba import njit
import numpy as np
from math import sqrt
from scipy.integrate import dblquad
from scipy.optimize import root_scalar
from ...utils.integration_utils import (
    divide_integration_borders_multiple_grid,
    find_integration_borders_square,
    domains_sep_hyperboles_inside,
    domains_sep_hyperboles_above,
)
from ...aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
    f_out_Bayes_decorrelated_noise,
    Z_out_Bayes_single_noise,
    f_out_Bayes_single_noise,
    f_out_hinge,
    Df_out_hinge,
)

N_GRID = 5


@njit(error_model="numpy", fastmath=True)
def m_integral_Hinge_single_noise(y, xi, q, m, sigma, delta):
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * f_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * f_out_hinge(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Hinge_single_noise(y, xi, q, m, sigma, delta):
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * (f_out_hinge(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Hinge_single_noise(y, xi, q, m, sigma, delta):
    eta = m**2 / q
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_single_noise(y, np.sqrt(eta) * xi, 1 - eta, delta)
        * Df_out_hinge(y, np.sqrt(q) * xi, sigma)
    )


# -----------------------------------


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


def f_hat_Hinge_single_noise(m, q, sigma, alpha, delta):
    borders = find_integration_borders_square(
        m_integral_Hinge_single_noise,
        1 * np.sqrt(1 + delta),
        1.0,
        args=(q, m, sigma, delta),
    )

    domain_xi_1, domain_y_1 = domains_sep_hyperboles_inside(
        borders, hyperbole, hyperbole, {"const": (1.0 - sigma) / sqrt(q)}, {"const": 1.0 / sqrt(q)}
    )
    domain_xi_2, domain_y_2 = domains_sep_hyperboles_above(borders, hyperbole, {"const": (1.0 - sigma) / sqrt(q)})

    # --- m hat integral ---
    domain_xi_m_hat, domain_y_m_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
    integral_value_m_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_m_hat, domain_y_m_hat):
        integral_value_m_hat += dblquad(
            m_integral_Hinge_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
        )[0]
    m_hat = alpha * integral_value_m_hat

    #  --- q hat integral ---
    domain_xi_q_hat, domain_y_q_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
    integral_value_q_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_q_hat, domain_y_q_hat):
        integral_value_q_hat += dblquad(
            q_integral_Hinge_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
        )[0]
    q_hat = alpha * integral_value_q_hat

    # --- Sigma hat integral ---
    domain_xi_sigma_hat, domain_y_sigma_hat = domain_xi_1, domain_y_1
    integral_value_sigma_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_sigma_hat, domain_y_sigma_hat):
        integral_value_sigma_hat += dblquad(
            sigma_integral_Hinge_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
        )[0]
    sigma_hat = -alpha * integral_value_sigma_hat

    return m_hat, q_hat, sigma_hat


def f_hat_Hinge_decorrelated_noise(m, q, sigma, alpha, delta_in, delta_out, percentage, beta):
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

    # --- m hat integral ---
    domain_xi_m_hat, domain_y_m_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
    integral_value_m_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_m_hat, domain_y_m_hat):
        integral_value_m_hat += dblquad(
            m_integral_Hinge_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]
    m_hat = alpha * integral_value_m_hat

    #  --- q hat integral ---
    domain_xi_q_hat, domain_y_q_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
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
    domain_xi_sigma_hat, domain_y_sigma_hat = domain_xi_2, domain_y_2
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
