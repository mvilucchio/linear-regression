from math import sqrt, pi, erf, erfc, acos, exp
from scipy.integrate import quad
import numpy as np


# ---------------------------- overlap parameters ---------------------------- #
def m_overlap(m, q, sigma, **args):
    return m


def q_overlap(m, q, sigma, **args):
    return q


def sigma_overlap(m, q, sigma, **args):
    return sigma


def angle_teacher_student(m, q, sigma, **args):
    return m / sqrt(q)


# --------------------------- errors classification -------------------------- #


def classification_adversarial_error(m, q, P, eps, pstar):
    Iminus = quad(
        lambda x: exp(-0.5 * x**2 / q) * erfc(m * x / sqrt(2 * q * (q - m**2))),
        -eps * P ** (1 / pstar),
        np.inf,
    )[0]
    Iplus = quad(
        lambda x: exp(-0.5 * x**2 / q) * (1 + erf(m * x / sqrt(2 * q * (q - m**2)))),
        -np.inf,
        eps * P ** (1 / pstar),
    )[0]
    return 0.5 * (Iminus + Iplus) / sqrt(2 * pi * q)


def classifiction_error(m, q, sigma, **args):
    return acos(m / sqrt(q)) / pi


# ----------------------------- errors regression ---------------------------- #
# @njit(error_model="numpy", fastmath=True)
def estimation_error(m, q, sigma, **args):
    return 1 + q - 2.0 * m


def margin_probit_classif(m, q, sigma, delta):
    return (4 * m * sqrt(2 * pi**3)) / sqrt(delta + 1)


# errors
def gen_error(m, q, sigma, delta_in, delta_out, percentage, beta):
    return q - 2 * m * (1 + (-1 + beta) * percentage) + 1 + percentage * (-1 + beta**2)


def excess_gen_error(m, q, sigma, delta_in, delta_out, percentage, beta):
    gen_err_BO_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (
        1 - percentage
    ) ** 2 * (beta - 1) ** 2
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
    oracle_norm = 1.0  # abs(1 - percentage + percentage * beta)
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return estimation_error(m_prime, q_prime, sigma)


def gen_error_BO(m, q, sigma, delta_in, delta_out, percentage, beta):
    return (1 + percentage * (-1 + beta**2) - (1 + percentage * (-1 + beta)) ** 2 * q) - (
        (1 - percentage) * percentage**2 * (1 - beta) ** 2
        + percentage * (1 - percentage) ** 2 * (beta - 1) ** 2
    )


def gen_error_BO_old(m, q, sigma, delta_in, delta_out, percentage, beta):
    q = (1 - percentage + percentage * beta) ** 2 * q
    m = (1 - percentage + percentage * beta) * m

    return estimation_error(m, q, sigma, tuple())
