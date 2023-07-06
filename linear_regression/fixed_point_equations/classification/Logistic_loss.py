from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar
from ...aux_functions.misc import gaussian
from ...aux_functions.loss_functions import logistic_loss, DDz_logistic_loss

BIG_NUMBER = 20


@njit(error_model="numpy", fastmath=False)
def moreau_loss(x, y, omega, V):
    return (x - omega) ** 2 / (2 * V) + logistic_loss(y, x)


# -----------------------------------
def m_int_Logistic_no_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss, args=(y, sqrt(q) * ξ, Σ))["x"]
    return (
        y
        * gaussian(ξ, 0.0, 1.0)
        * exp(-0.5 * η * ξ**2 / (1 - η))
        / sqrt(2 * pi * (1 - η))
        * (proximal - sqrt(q) * ξ)
        / Σ
    )


def q_int_Logistic_no_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss, args=(y, sqrt(q) * ξ, Σ))["x"]
    return 0.5 * (
        gaussian(ξ, 0.0, 1.0) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (proximal - sqrt(q) * ξ) ** 2 / Σ**2
    )


def Σ_int_Logistic_no_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss, args=(y, sqrt(q) * ξ, Σ))["x"]
    Dproximal = 1 / (1 + Σ * DDz_logistic_loss(y, proximal))
    return 0.5 * gaussian(ξ, 0.0, 1.0) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (Dproximal - 1) / Σ


# -----------------------------------
def m_int_Logistic_single_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss, args=(y, sqrt(q) * ξ, Σ))["x"]
    return y * gaussian(ξ, 0.0, 1.0) * exp(-0.5 * η * ξ**2 / (1 - η)) * (proximal - sqrt(q) * ξ) / Σ


def q_int_Logistic_single_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss, args=(y, sqrt(q) * ξ, Σ))["x"]
    return (
        gaussian(ξ, 0.0, 1.0) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (proximal - sqrt(q) * ξ) ** 2 / Σ**2
    )


def Σ_int_Logistic_single_noise_classif(ξ, y, q, m, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss, args=(y, sqrt(q) * ξ, Σ))["x"]
    Dproximal = (1 + Σ * DDz_logistic_loss(y, proximal)) ** (-1)
    return gaussian(ξ, 0.0, 1.0) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (Dproximal - 1) / Σ


# -----------------------------------


def f_hat_Logistic_no_noise_classif(m, q, Σ, alpha):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(m_int_Logistic_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, Σ))[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(q_int_Logistic_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, Σ))[0]
    q_hat = alpha * int_value_q_hat

    int_value_Σ_hat = 0.0
    for y_val, domain in domains:
        int_value_Σ_hat += quad(Σ_int_Logistic_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, Σ))[0]
    Σ_hat = -alpha * int_value_Σ_hat

    return m_hat, q_hat, Σ_hat


def f_hat_Logistic_single_noise_classif(m, q, Σ, alpha, delta):
    raise NotImplementedError
