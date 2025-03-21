from numba import njit
from numpy import pi
from math import erf, erfc, exp, log, sqrt
from ...aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_single_noise,
    f_out_Bayes_single_noise,
    f_out_phase_retrival_simple,
    Df_out_phase_retrival_simple,
)
from scipy.integrate import dblquad

BIG_NUMBER = 5.0


@njit(error_model="numpy", fastmath=True)
def m_integral_phase_retrival_single_noise(y, xi, q, m, V, delta):
    eta = m**2 / q
    return (
        exp(-(xi**2) / 2)
        / sqrt(2 * pi)
        * Z_out_Bayes_single_noise(y, sqrt(eta) * xi, 1 - eta, delta)
        * f_out_Bayes_single_noise(y, sqrt(eta) * xi, 1 - eta, delta)
        * f_out_phase_retrival_simple(y, sqrt(q) * xi, V)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_phase_retrival_single_noise(y, xi, q, m, V, delta):
    eta = m**2 / q
    return (
        exp(-(xi**2) / 2)
        / sqrt(2 * pi)
        * Z_out_Bayes_single_noise(y, sqrt(eta) * xi, 1 - eta, delta)
        * (f_out_phase_retrival_simple(y, sqrt(q) * xi, V) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def V_integral_phase_retrival_single_noise(y, xi, q, m, V, delta):
    eta = m**2 / q
    return (
        exp(-(xi**2) / 2)
        / sqrt(2 * pi)
        * Z_out_Bayes_single_noise(y, sqrt(eta) * xi, 1 - eta, delta)
        * Df_out_phase_retrival_simple(y, sqrt(q) * xi, V)
    )


def f_hat_phase_retrival_single_noise(
    m: float, q: float, V: float, alpha: float, delta: float
) -> float:
    integral_value = dblquad(
        m_integral_phase_retrival_single_noise,
        -BIG_NUMBER,
        BIG_NUMBER,
        -BIG_NUMBER,
        BIG_NUMBER,
        args=(q, m, V, delta),
    )[0]
    m_hat = alpha * integral_value

    integral_value = dblquad(
        q_integral_phase_retrival_single_noise,
        -BIG_NUMBER,
        BIG_NUMBER,
        -BIG_NUMBER,
        BIG_NUMBER,
        args=(q, m, V, delta),
    )[0]
    q_hat = alpha * integral_value

    integral_value = dblquad(
        V_integral_phase_retrival_single_noise,
        -BIG_NUMBER,
        BIG_NUMBER,
        -BIG_NUMBER,
        BIG_NUMBER,
        args=(q, m, V, delta),
    )[0]
    V_hat = -alpha * integral_value

    return m_hat, q_hat, V_hat
