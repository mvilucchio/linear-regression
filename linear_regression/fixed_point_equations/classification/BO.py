from numba import njit
import numpy as np
from scipy.integrate import quad, dblquad
from math import sqrt, exp, pi, erf, erfc
from ...aux_functions.misc import gaussian

BIG_NUMBER = 3
SMALL_EPS = 1e-12


@njit(error_model="numpy", fastmath=True)
def f_BO(m_hat, q_hat, V_hat):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


# -----------------------------------
def q_int_BO_probit_classif(ξ, y, q, m, V, delta):
    A = gaussian(ξ, 0, 1) / (pi * (1 - q + delta))
    B = exp(-q * ξ**2 / (1 - q + delta))
    C = 1 + erf(y * sqrt(0.5 * q / (1 - q + delta)) * ξ) + SMALL_EPS
    return A * B / C


def q_int_BO_no_noise_classif(ξ, y, q, m, V):
    A = gaussian(ξ, 0, 1) / (pi * (1 - q))
    B = exp(-q * ξ**2 / (1 - q))
    # C = 1 - erf(-y * sqrt(q) * ξ / sqrt(2 * (1 - q)))
    C = erfc(-y * sqrt(q) * ξ / sqrt(2 * (1 - q))) + SMALL_EPS
    return A * B / C


def q_int_BO_flip_sign_classif(ξ, y, q, m, V, eps):
    raise NotImplementedError


# -----------------------------------
def f_hat_BO_probit_classif(m, q, V, alpha, delta):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    integral_value_q_hat = 0.0
    for y_val, domain in domains:
        integral_value_q_hat += quad(
            q_int_BO_probit_classif, domain[0], domain[1], args=(y_val, q, m, V, delta)
        )[0]
    q_hat = alpha * integral_value_q_hat

    return q_hat, q_hat, 1 - q_hat


def f_hat_BO_no_noise_classif(m, q, V, alpha):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    integral_value_q_hat = 0.0
    for y_val, domain in domains:
        integral_value_q_hat += quad(
            q_int_BO_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    q_hat = alpha * integral_value_q_hat

    return q_hat, q_hat, 1 - q_hat
