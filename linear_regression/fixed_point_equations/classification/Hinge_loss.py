from numba import njit
import numpy as np
from math import sqrt, exp, pi, erf
from scipy.integrate import quad, dblquad
from ...aux_functions.misc import gaussian
from ...utils.integration_utils import (
    find_integration_borders_square,
    domains_sep_hyperboles_inside,
    domains_sep_hyperboles_above,
    line_borders_hinge_above,
    line_borders_hinge_inside,
)
from ...aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_single_noise_classif,
    Z_out_Bayes_f_out_Bayes_single_noise_classif,
    f_out_Hinge,
    Df_out_Hinge,
)


@njit(error_model="numpy", fastmath=False)
def m_int_Hinge_probit_classif(ξ, y, q, m, Σ, delta):
    η = m**2 / q
    return y * gaussian(ξ, 0, 1) * gaussian(sqrt(η) * ξ, 0, 1 - η + delta) * f_out_Hinge(y, sqrt(q) * ξ, Σ)


@njit(error_model="numpy", fastmath=False)
def q_int_Hinge_probit_classif(ξ, y, q, m, Σ, delta):
    η = m**2 / q
    return 0.5 * (1 + erf(y * sqrt(0.5 * η / (1 - η + delta)) * ξ)) * (f_out_Hinge(y, sqrt(q) * ξ, Σ)) ** 2


@njit(error_model="numpy", fastmath=False)
def Σ_int_Hinge_probit_classif(ξ, y, q, m, Σ, delta):
    η = m**2 / q
    return 0.5 * (1 + erf(y * sqrt(0.5 * η / (1 - η + delta)) * ξ)) * Df_out_Hinge(y, sqrt(q) * ξ, Σ)


# ------------------------------------


@njit(error_model="numpy", fastmath=False)
def m_int_Hinge_no_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    return y * (
        gaussian(ξ, 0, 1) / sqrt(2.0 * pi * (1 - η)) * exp(-0.5 * η * ξ**2 / (1 - η)) * f_out_Hinge(y, sqrt(q) * ξ, Σ)
    )


@njit(error_model="numpy", fastmath=False)
def q_int_Hinge_no_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    return (
        0.5 * gaussian(ξ, 0, 1) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (f_out_Hinge(y, sqrt(q) * ξ, Σ) ** 2)
    )


@njit(error_model="numpy", fastmath=False)
def Σ_int_Hinge_no_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    return 0.5 * gaussian(ξ, 0, 1) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * Df_out_Hinge(y, sqrt(q) * ξ, Σ)


# ------------------------------------


@njit(error_model="numpy", fastmath=False)
def m_int_Hinge_single_noise_classif(y, ξ, q, m, Σ, delta):
    η = m**2 / q
    return (
        gaussian(ξ, 0, 1)
        * Z_out_Bayes_f_out_Bayes_single_noise_classif(y, sqrt(η) * ξ, 1 - η, delta)
        * f_out_Hinge(y, sqrt(q) * ξ, Σ)
    )


@njit(error_model="numpy", fastmath=False)
def q_int_Hinge_single_noise_classif(y, ξ, q, m, Σ, delta):
    η = m**2 / q
    return (
        gaussian(ξ, 0, 1)
        * Z_out_Bayes_single_noise_classif(y, sqrt(η) * ξ, 1 - η, delta)
        * (f_out_Hinge(y, sqrt(q) * ξ, Σ) ** 2)
    )


@njit(error_model="numpy", fastmath=False)
def Σ_int_Hinge_single_noise_classif(y, ξ, q, m, Σ, delta):
    η = m**2 / q
    return (
        gaussian(ξ, 0, 1)
        * Z_out_Bayes_single_noise_classif(y, sqrt(η) * ξ, 1 - η, delta)
        * Df_out_Hinge(y, sqrt(q) * ξ, Σ)
    )


# ------------------------------------


def f_hat_Hinge_probit_classif(m, q, Σ, alpha, delta):
    domains_internal = line_borders_hinge_inside(m, q, Σ)
    domains_external = line_borders_hinge_above(m, q, Σ)

    integral_value_m_hat = 0.0
    for y_val, domain in domains_internal + domains_external:
        integral_value_m_hat += quad(m_int_Hinge_probit_classif, domain[0], domain[1], args=(y_val, q, m, Σ, delta))[0]
    m_hat = alpha * integral_value_m_hat

    integral_value_q_hat = 0.0
    for y_val, domain in domains_internal + domains_external:
        integral_value_q_hat += quad(q_int_Hinge_probit_classif, domain[0], domain[1], args=(y_val, q, m, Σ, delta))[0]
    q_hat = alpha * integral_value_q_hat

    integral_value_Σ_hat = 0.0
    for y_val, domain in domains_internal:
        integral_value_Σ_hat += quad(Σ_int_Hinge_probit_classif, domain[0], domain[1], args=(y_val, q, m, Σ, delta))[0]
    Σ_hat = -alpha * integral_value_Σ_hat

    return m_hat, q_hat, Σ_hat


def f_hat_Hinge_no_noise_classif(m, q, Σ, alpha):
    domains_internal = line_borders_hinge_inside(m, q, Σ)
    domains_external = line_borders_hinge_above(m, q, Σ)

    integral_value_m_hat = 0.0
    for y_val, domain in domains_internal + domains_external:
        integral_value_m_hat += quad(m_int_Hinge_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, Σ))[0]
    m_hat = alpha * integral_value_m_hat

    integral_value_q_hat = 0.0
    for y_val, domain in domains_internal + domains_external:
        integral_value_q_hat += quad(q_int_Hinge_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, Σ))[0]
    q_hat = alpha * integral_value_q_hat

    integral_value_Σ_hat = 0.0
    for y_val, domain in domains_internal:
        integral_value_Σ_hat += quad(Σ_int_Hinge_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, Σ))[0]
    Σ_hat = -alpha * integral_value_Σ_hat

    return m_hat, q_hat, Σ_hat


@njit(error_model="numpy")
def hyperbole(x, const):
    return const / x


def f_hat_Hinge_single_noise_classif(m, q, Σ, alpha, delta):
    borders = find_integration_borders_square(
        m_int_Hinge_single_noise_classif, sqrt(1 + delta), 1, args=(q, m, Σ, delta), mult=10
    )

    domain_xi_1, domain_y_1 = domains_sep_hyperboles_inside(
        borders, hyperbole, hyperbole, {"const": (1 - Σ) / sqrt(q)}, {"const": 1 / sqrt(q)}
    )
    domain_xi_2, domain_y_2 = domains_sep_hyperboles_above(borders, hyperbole, {"const": (1 - Σ) / sqrt(q)})

    # --- m hat integral ---
    domain_xi_m_hat, domain_y_m_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
    integral_value_m_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_m_hat, domain_y_m_hat):
        integral_value_m_hat += dblquad(
            m_int_Hinge_single_noise_classif,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, Σ, delta),
        )[0]
    m_hat = alpha * integral_value_m_hat

    #  --- q hat integral ---
    domain_xi_q_hat, domain_y_q_hat = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
    integral_value_q_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_q_hat, domain_y_q_hat):
        integral_value_q_hat += dblquad(
            q_int_Hinge_single_noise_classif,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, Σ, delta),
        )[0]
    q_hat = alpha * integral_value_q_hat

    # --- Σ hat integral ---
    domain_xi_Σ_hat, domain_y_Σ_hat = domain_xi_1, domain_y_1
    integral_value_Σ_hat = 0.0
    for xi_funs, y_funs in zip(domain_xi_Σ_hat, domain_y_Σ_hat):
        integral_value_Σ_hat += dblquad(
            Σ_int_Hinge_single_noise_classif,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, Σ, delta),
        )[0]
    Σ_hat = -alpha * integral_value_Σ_hat

    return m_hat, q_hat, Σ_hat
