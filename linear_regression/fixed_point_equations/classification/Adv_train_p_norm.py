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

BIG_NUMBER = 55


# -----------------------------------
@njit(error_model="numpy", fastmath=False)
def m_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, Σ: float, P: float, eps_t: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_adversarial(y, sqrt(q) * ξ, Σ, P, eps_t)
    return (
        y
        * gaussian(ξ, 0, 1)
        * exp(-0.5 * η * ξ**2 / (1 - η))
        / sqrt(2 * pi * (1 - η))
        * (proximal - sqrt(q) * ξ)
        / Σ
    )


@njit(error_model="numpy", fastmath=False)
def q_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, Σ: float, P: float, eps_t: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_adversarial(y, sqrt(q) * ξ, Σ, P, eps_t)
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (proximal - sqrt(q) * ξ) ** 2
        / Σ**2
    )


@njit(error_model="numpy", fastmath=False)
def Σ_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, Σ: float, P: float, eps_t: float
) -> float:
    η = m**2 / q
    Dproximal = Dω_proximal_Logistic_adversarial(y, sqrt(q) * ξ, Σ, P, eps_t)
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (Dproximal - 1)
        / Σ
    )


@njit(error_model="numpy", fastmath=False)
def P_int_Adv_Logistic_no_noise_classif(
    ξ: float, y: float, q: float, m: float, Σ: float, P: float, eps_t: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_adversarial(y, sqrt(q) * ξ, Σ, P, eps_t)
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * y
        * (proximal - sqrt(q) * ξ)
        / Σ
    )


def f_hat_Logistic_no_noise_Linf_adv_classif(m, q, Σ, P, eps_t, alpha):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(
            m_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, Σ, P, eps_t),
        )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(
            q_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, Σ, P, eps_t),
        )[0]
    q_hat = alpha * int_value_q_hat

    int_value_Σ_hat = 0.0
    for y_val, domain in domains:
        int_value_Σ_hat += quad(
            Σ_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, Σ, P, eps_t),
        )[0]
    Σ_hat = -alpha * int_value_Σ_hat

    int_value_P_hat = 0.0
    for y_val, domain in domains:
        int_value_P_hat += quad(
            P_int_Adv_Logistic_no_noise_classif,
            domain[0],
            domain[1],
            args=(y_val, q, m, Σ, P, eps_t),
        )[0]
    P_hat = eps_t * alpha * int_value_P_hat

    return m_hat, q_hat, Σ_hat, P_hat
