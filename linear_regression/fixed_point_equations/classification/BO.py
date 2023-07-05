from numba import njit
import numpy as np
from scipy.integrate import quad, dblquad
from math import sqrt, exp, pi, erf
from ...aux_functions.misc import gaussian
from ...utils.integration_utils import (
    divide_integration_borders_multiple_grid,
    find_integration_borders_square,
)
from ...aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
    f_out_Bayes_decorrelated_noise,
)

BIG_NUMBER = 50


@njit(error_model="numpy", fastmath=True)
def f_BO(m_hat, q_hat, sigma_hat):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


def q_int_BO_no_noise_classif(ξ, y, q, m, Σ):
    return (
        gaussian(ξ, 0, 1) / (pi * (1 - q)) * exp(-q * ξ**2 / (1 - q)) / (1 + y * erf(sqrt(q) * ξ / sqrt(2 * (1 - q))))
    )


def f_hat_BO_no_noise_classif(m, q, Σ, alpha):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    integral_value_q_hat = 0.0
    for y_val, domain in domains:
        integral_value_q_hat += quad(q_int_BO_no_noise_classif, domain[0], domain[1], args=(y_val, q, m, Σ))[0]
    q_hat = alpha * integral_value_q_hat

    return q_hat, q_hat, 1 - q_hat
