from numba import njit
from numpy import pi
from math import erf, erfc, exp, log, sqrt


# --------------------------- single gaussian noise -------------------------- #


@njit(error_model="numpy", fastmath=True)
def f_hat_superHuber_single_noise(m, q, V, alpha, delta, a):
    arg_sqrt = 1 + q + delta - 2 * m
    erf_arg = (a * (V + 1)) / sqrt(2 * arg_sqrt)

    m_hat = (alpha / (1 + V)) * erf(erf_arg)
    q_hat = (alpha / (1 + V) ** 2) * (
        arg_sqrt * erf(erf_arg)
        + a**2 * (1 + V) ** 2 * erfc(erf_arg)
        - a * (1 + V) * sqrt(2 / pi) * sqrt(arg_sqrt) * exp(-(erf_arg**2))
    )
    V_hat = (alpha / (1 + V)) * erf(erf_arg)
    return m_hat, q_hat, V_hat
