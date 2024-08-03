from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar
from ...aux_functions.misc import gaussian
from ...aux_functions.loss_functions import logistic_loss, DDz_logistic_loss

BIG_NUMBER = 20


@njit(error_model="numpy", fastmath=False)
def moreau_loss_adv(x, y, omega, V, P, eps_t):
    return (x - omega) ** 2 / (2 * V) + logistic_loss(y, x - y * sqrt(P) * eps_t)


# -----------------------------------
def m_int_Adv_Logistic_no_noise_classif(ξ, y, q, m, V, P, eps_t):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_adv, args=(y, sqrt(q) * ξ, V, P, eps_t))["x"]
    return (
        y
        * gaussian(ξ, 0, 1)
        * exp(-0.5 * η * ξ**2 / (1 - η))
        / sqrt(2 * pi * (1 - η))
        * (proximal - sqrt(q) * ξ)
        / V
    )


def q_int_Adv_Logistic_no_noise_classif(ξ, y, q, m, V, P, eps_t):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_adv, args=(y, sqrt(q) * ξ, V, P, eps_t))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


def V_int_Adv_Logistic_no_noise_classif(ξ, y, q, m, V, P, eps_t):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_adv, args=(y, sqrt(q) * ξ, V, P, eps_t))["x"]
    Dproximal = 1 / (1 + V * DDz_logistic_loss(y, proximal))
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (Dproximal - 1)
        / V
    )


def P_int_Adv_Logistic_no_noise_classif(ξ, y, q, m, V, P, eps_t):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_adv, args=(y, sqrt(q) * ξ, V, P, eps_t))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * y
        * (proximal - sqrt(q) * ξ)
        / V
    )


# -----------------------------------


def f_hat_Logistic_no_noise_classif(m, q, V, P, eps_t, alpha):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(
            m_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, eps_t),
        )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(
            q_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, eps_t),
        )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = 0.0
    for y_val, domain in domains:
        int_value_V_hat += quad(
            V_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, eps_t),
        )[0]
    V_hat = -alpha * int_value_V_hat

    int_value_P_hat = 0.0
    for y_val, domain in domains:
        int_value_P_hat += quad(
            P_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, eps_t),
        )[0]
    P_hat = 0.5 * eps_t * alpha / sqrt(P) * int_value_P_hat

    return m_hat, q_hat, V_hat, P_hat
