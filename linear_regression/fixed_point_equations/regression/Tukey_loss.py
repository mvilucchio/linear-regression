from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad, dblquad
from ...aux_functions.moreau_proximals import proximal_Logistic_loss, Dω_proximal_Logistic_loss
from ...aux_functions.misc import gaussian
from ...aux_functions.loss_functions import logistic_loss, DDz_logistic_loss

BIG_NUMBER = 20


def m_int_Tukey_decorrelated_noise(
    ξ: float,
    y: float,
    q: float,
    m: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    τ: float,
):
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return y * gaussian(ξ, 0, 1) * exp(-0.5 * η * ξ**2 / (1 - η)) * (proximal - sqrt(q) * ξ) / V


def q_int_Tukey_decorrelated_noise(
    ξ: float,
    y: float,
    q: float,
    m: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    τ: float,
):
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


def V_int_Tukey_decorrelated_noise(
    ξ: float,
    y: float,
    q: float,
    m: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    τ: float,
):
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    Dproximal = (1 + V * DDz_logistic_loss(y, proximal)) ** (-1)
    return gaussian(ξ, 0, 1) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (Dproximal - 1) / V


# -----------------------------------


def f_hat_Tukey_decorrelated_noise(m, q, V, alpha, delta_in, delta_out, percentage, beta, a):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(
            m_int_Tukey_decorrelated_noise, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(
            q_int_Tukey_decorrelated_noise, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = 0.0
    for y_val, domain in domains:
        int_value_V_hat += quad(
            V_int_Tukey_decorrelated_noise, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    V_hat = -alpha * int_value_V_hat

    return m_hat, q_hat, V_hat
