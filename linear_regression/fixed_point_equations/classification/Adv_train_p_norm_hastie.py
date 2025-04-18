from numba import njit
import numpy as np
from numpy import sign as np_sign
from math import exp, erf, sqrt, pi, log, exp
from scipy.integrate import quad, dblquad
from ...aux_functions.misc import gaussian
from ...aux_functions.moreau_proximals import (
    proximal_Logistic_adversarial,
    Dω_proximal_Logistic_adversarial,
)

BIG_NUMBER = 35


# -----------------------------------
@njit(error_model="numpy", fastmath=False)
def m_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, V: float, P: float, ε: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_adversarial(y, sqrt(q) * ξ, V, P, ε)
    return (
        y
        * gaussian(ξ, 0, 1)
        * exp(-0.5 * η * ξ**2 / (1 - η))
        / sqrt(2 * pi * (1 - η))
        * (proximal - sqrt(q) * ξ)
        / V
    )


@njit(error_model="numpy", fastmath=False)
def q_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, V: float, P: float, ε: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_adversarial(y, sqrt(q) * ξ, V, P, ε)
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


@njit(error_model="numpy", fastmath=False)
def V_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, V: float, P: float, ε: float
) -> float:
    η = m**2 / q
    Dproximal = Dω_proximal_Logistic_adversarial(y, sqrt(q) * ξ, V, P, ε)
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (Dproximal - 1)
        / V
    )


@njit(error_model="numpy", fastmath=False)
def P_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, V: float, P: float, ε: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_adversarial(y, sqrt(q) * ξ, V, P, ε)
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * y
        * (proximal - sqrt(q) * ξ)
        / V
    )


def f_hat_Logistic_no_noise_Linf_adv_classif(m, q, V, P, ε, alpha, gamma):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(
            m_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, ε),
        )[0]
    m_hat = alpha * sqrt(gamma) * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(
            q_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, ε),
        )[0]
    q_hat = alpha * gamma * int_value_q_hat

    int_value_V_hat = 0.0
    for y_val, domain in domains:
        int_value_V_hat += quad(
            V_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, ε),
        )[0]
    V_hat = -alpha * gamma * int_value_V_hat

    int_value_P_hat = 0.0
    for y_val, domain in domains:
        int_value_P_hat += quad(
            P_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, V, P, ε),
        )[0]
    P_hat = ε * alpha * gamma * int_value_P_hat

    return m_hat, q_hat, V_hat, P_hat
