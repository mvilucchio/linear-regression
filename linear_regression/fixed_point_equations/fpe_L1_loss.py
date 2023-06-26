from numba import njit
from numpy import pi
# from math import exp, log, sqrt # erf, erfc, 
from scipy.special import erf, erfc
from numpy import exp, log, sqrt


@njit(error_model="numpy", fastmath=False)
def var_hat_func_L1_single_noise(m, q, sigma, alpha, delta):
    sqrt_arg = 1 + q + delta - 2 * m
    erf_arg = sigma / sqrt(2 * sqrt_arg)

    m_hat = (alpha / sigma) * erf(erf_arg)
    q_hat = (alpha / sigma**2) * (
        sqrt_arg * erf(erf_arg)
        + sigma**2 * erfc(erf_arg)
        - sigma * sqrt(2 / pi) * sqrt(sqrt_arg) * exp(-(erf_arg**2))
    )
    sigma_hat = (alpha / sigma) * erf(erf_arg)
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=False)
def var_hat_func_L1_double_noise(m, q, sigma, alpha, delta_in, delta_out, percentage):
    small_sqrt = delta_in - 2 * m + q + 1
    large_sqrt = delta_out - 2 * m + q + 1

    small_exp = -(sigma**2) / (2 * small_sqrt)
    large_exp = -(sigma**2) / (2 * large_sqrt)

    small_erf = sigma / sqrt(2 * small_sqrt)
    large_erf = sigma / sqrt(2 * large_sqrt)

    # probabily should change it
    m_hat = (alpha / sigma) * ((1 - percentage) * erf(small_erf) + percentage * erf(large_erf))
    q_hat = alpha * ((1 - percentage) * erfc(small_erf) + percentage * erfc(large_erf)) + alpha / sigma**2 * (
        ((1 - percentage) * (small_sqrt) * erf(small_erf) + percentage * (large_sqrt) * erf(large_erf))
        - exp(
            log(sigma)
            + 0.5 * log(2)
            - 0.5 * log(pi)
            + 0.5 * log(large_sqrt)
            + log((1 - percentage) * sqrt(small_sqrt / large_sqrt) * exp(small_exp) + percentage * exp(large_exp))
        )
    )
    sigma_hat = (alpha / sigma) * ((1 - percentage) * erf(small_erf) + percentage * erf(large_erf))
    return m_hat, q_hat, sigma_hat


# @njit(error_model="numpy", fastmath=False)
def var_hat_func_L1_decorrelated_noise(m, q, sigma, alpha, delta_in, delta_out, percentage, beta):
    small_sqrt = delta_in - 2 * m + q + 1
    large_sqrt = delta_out - 2 * m * beta + q + beta**2
    small_exp = -(sigma**2) / (2 * small_sqrt)
    large_exp = -(sigma**2) / (2 * large_sqrt)
    small_erf = sigma / sqrt(2 * small_sqrt)
    large_erf = sigma / sqrt(2 * large_sqrt)

    m_hat = (alpha / sigma) * ((1 - percentage) * erf(small_erf) + beta * percentage * erf(large_erf))
    q_hat = alpha * ((1 - percentage) * erfc(small_erf) + percentage * erfc(large_erf)) + alpha / sigma**2 * (
        ((1 - percentage) * (small_sqrt) * erf(small_erf) + percentage * (large_sqrt) * erf(large_erf))
        - exp(
            log(sigma)
            + 0.5 * log(2)
            - 0.5 * log(pi)
            + log((1 - percentage) * sqrt(small_sqrt) * exp(small_exp) + percentage * sqrt(large_sqrt) * exp(large_exp))
        )
    )
    sigma_hat = (alpha / sigma) * ((1 - percentage) * erf(small_erf) + percentage * erf(large_erf))
    return m_hat, q_hat, sigma_hat


@njit
def x_next_plateau_L1(x, delta_in, delta_out, percentage, beta):
    y_in = delta_in + percentage**2 * x**2
    y_out = delta_out + (percentage * x + 1) ** 2
    return -1 / ((1 - percentage) * sqrt(y_out / y_in) + percentage)
