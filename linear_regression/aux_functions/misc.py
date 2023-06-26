from math import exp, sqrt, acos
from numpy import pi, arccos
import numpy as np
from numba import vectorize, njit


@vectorize("float64(float64, float64, float64)")
def std_gaussian(x: float, mu: float, sigma_2: float) -> float:
    return exp(-0.5 * pow(x - mu, 2.0) / sigma_2) / sqrt(2 * pi * sigma_2)


@vectorize("float64(float64, float64, float64)")
def damped_update(new, old, damping):
    """
    Damped update of old value with new value.
    the opertation that is performed is:
    damping * new + (1 - damping) * old
    """
    return damping * new + (1 - damping) * old


# @njit(error_model="numpy", fastmath=True)
def estimation_error(m, q, sigma, **args):
    return 1 + q - 2.0 * m


# @njit
def angle_teacher_student(m, q, sigma, **args):
    return np.arccos(m / np.sqrt(q)) / pi


def gen_error(m, q, sigma, delta_in, delta_out, percentage, beta):
    return q - 2 * m * (1 + (-1 + beta) * percentage) + 1 + percentage * (-1 + beta**2)


def excess_gen_error(m, q, sigma, delta_in, delta_out, percentage, beta):
    gen_err_BO_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (1 - percentage) ** 2 * (
        beta - 1
    ) ** 2
    return gen_error(m, q, sigma, delta_in, delta_out, percentage, beta) - gen_err_BO_alpha_inf


def excess_gen_error_oracle_rescaling(m, q, sigma, delta_in, delta_out, percentage, beta):
    oracle_norm = 1 - percentage + percentage * beta
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return excess_gen_error(m_prime, q_prime, sigma, delta_in, delta_out, percentage, beta)


def estimation_error_rescaled(m, q, sigma, delta_in, delta_out, percentage, beta, norm_const):
    m = m / norm_const
    q = q / (norm_const**2)

    return estimation_error(m, q, sigma)


def estimation_error_oracle_rescaling(m, q, sigma, delta_in, delta_out, percentage, beta):
    oracle_norm = 1.0 # abs(1 - percentage + percentage * beta)
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return estimation_error(m_prime, q_prime, sigma)


def gen_error_BO(m, q, sigma, delta_in, delta_out, percentage, beta):
    return (1 + percentage * (-1 + beta**2) - (1 + percentage * (-1 + beta)) ** 2 * q) - (
        (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (1 - percentage) ** 2 * (beta - 1) ** 2
    )


def gen_error_BO_old(m, q, sigma, delta_in, delta_out, percentage, beta):
    q = (1 - percentage + percentage * beta) ** 2 * q
    m = (1 - percentage + percentage * beta) * m

    return estimation_error(m, q, sigma, tuple())
