from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar
from ...aux_functions.moreau_proximals import proximal_Logistic_loss, Dω_proximal_Logistic_loss
from ...aux_functions.misc import gaussian
from ...aux_functions.loss_functions import logistic_loss, DDz_logistic_loss

BIG_NUMBER = 20


# -----------------------------------
@njit(error_model="numpy", fastmath=False)
def m_int_Logistic_probit_classif(
    ξ: float, y: float, q: float, m: float, V: float, delta: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return (
        y
        * gaussian(ξ, 0, 1)
        * gaussian(sqrt(η) * ξ, 0, 1 - η + delta)
        * (proximal - sqrt(q) * ξ)
        / V
    )


@njit(error_model="numpy", fastmath=False)
def q_int_Logistic_probit_classif(
    ξ: float, y: float, q: float, m: float, V: float, delta: float
) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + erf(y * sqrt(0.5 * η / (1 - η + delta)) * ξ))
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


@njit(error_model="numpy", fastmath=False)
def V_int_Logistic_probit_classif(
    ξ: float, y: float, q: float, m: float, V: float, delta: float
) -> float:
    η = m**2 / q
    Dproximal = Dω_proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + erf(y * sqrt(0.5 * η / (1 - η + delta)) * ξ))
        * (Dproximal - 1)
        / V
    )


# -----------------------------------
@njit(error_model="numpy", fastmath=False)
def m_int_Logistic_no_noise_classif(ξ: float, y: float, q: float, m: float, V: float) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return (
        y
        * gaussian(ξ, 0, 1)
        * exp(-0.5 * η * ξ**2 / (1 - η))
        / sqrt(2 * pi * (1 - η))
        * (proximal - sqrt(q) * ξ)
        / V
    )


@njit(error_model="numpy", fastmath=False)
def q_int_Logistic_no_noise_classif(ξ: float, y: float, q: float, m: float, V: float) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


@njit(error_model="numpy", fastmath=False)
def V_int_Logistic_no_noise_classif(ξ: float, y: float, q: float, m: float, V: float) -> float:
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    Dproximal = 1 / (1 + V * DDz_logistic_loss(y, proximal))
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (Dproximal - 1)
        / V
    )


# -----------------------------------
def m_int_Logistic_single_noise_classif(ξ: float, y: float, q: float, m: float, V: float):
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return y * gaussian(ξ, 0, 1) * exp(-0.5 * η * ξ**2 / (1 - η)) * (proximal - sqrt(q) * ξ) / V


def q_int_Logistic_single_noise_classif(ξ: float, y: float, q: float, m: float, V: float):
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


def V_int_Logistic_single_noise_classif(ξ: float, y: float, q: float, m: float, V: float):
    η = m**2 / q
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    Dproximal = (1 + V * DDz_logistic_loss(y, proximal)) ** (-1)
    return gaussian(ξ, 0, 1) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (Dproximal - 1) / V


# -----------------------------------


def f_hat_Logistic_probit_classif(m, q, V, alpha, delta):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(
            m_int_Logistic_probit_classif, domain[0], domain[1], args=(y_val, q, m, V, delta)
        )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(
            q_int_Logistic_probit_classif, domain[0], domain[1], args=(y_val, q, m, V, delta)
        )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = 0.0
    for y_val, domain in domains:
        int_value_V_hat += quad(
            V_int_Logistic_probit_classif, domain[0], domain[1], args=(y_val, q, m, V, delta)
        )[0]
    V_hat = -alpha * int_value_V_hat

    return m_hat, q_hat, V_hat


def f_hat_Logistic_no_noise_classif(m, q, V, alpha):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(
            m_int_Logistic_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(
            q_int_Logistic_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = 0.0
    for y_val, domain in domains:
        int_value_V_hat += quad(
            V_int_Logistic_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    V_hat = -alpha * int_value_V_hat

    return m_hat, q_hat, V_hat


def f_hat_Logistic_single_noise_classif(m, q, V, alpha, delta):
    raise NotImplementedError
