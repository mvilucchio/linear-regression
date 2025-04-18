from math import sqrt, pi, erf, erfc, acos, exp
from scipy.integrate import quad
import numpy as np


# ---------------------------- overlap parameters ---------------------------- #
def m_overlap(m, q, V, **args):
    return m


def q_overlap(m, q, V, **args):
    return q


def V_overlap(m, q, V, **args):
    return V


def angle_teacher_student(m, q, V, **args):
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


def classifiction_error(m, q, V, **args):
    return acos(m / sqrt(q)) / pi


# ----------------------------- errors regression ---------------------------- #
# @njit(error_model="numpy", fastmath=True)
def estimation_error(m, q, V, **args):
    return 1 + q - 2.0 * m


def margin_probit_classif(m, q, V, Δ):
    return (4 * m * sqrt(2 * pi**3)) / sqrt(Δ + 1)


# errors
def gen_error(m, q, V, Δ_in, Δ_out, ε, β):
    return q - 2 * m * (1 + (-1 + β) * ε) + 1 + ε * (-1 + β**2)


def excess_gen_error(m, q, V, Δ_in, Δ_out, ε, β):
    gen_err_BO_alpha_inf = (1 - ε) * ε**2 * (1 - β) ** 2 + ε * (1 - ε) ** 2 * (β - 1) ** 2
    return gen_error(m, q, V, Δ_in, Δ_out, ε, β) - gen_err_BO_alpha_inf


def excess_gen_error_oracle_rescaling(m, q, V, Δ_in, Δ_out, ε, β):
    oracle_norm = 1 - ε + ε * β
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return excess_gen_error(m_prime, q_prime, V, Δ_in, Δ_out, ε, β)


def estimation_error_rescaled(m, q, V, Δ_in, Δ_out, ε, β, norm_const):
    m = m / norm_const
    q = q / (norm_const**2)

    return estimation_error(m, q, V)


def estimation_error_oracle_rescaling(m, q, V, Δ_in, Δ_out, ε, β):
    oracle_norm = 1.0  # abs(1 - ε + ε * β)
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return estimation_error(m_prime, q_prime, V)


def gen_error_BO(m, q, V, Δ_in, Δ_out, ε, β):
    return (1 + ε * (-1 + β**2) - (1 + ε * (-1 + β)) ** 2 * q) - (
        (1 - ε) * ε**2 * (1 - β) ** 2 + ε * (1 - ε) ** 2 * (β - 1) ** 2
    )


def gen_error_BO_old(m, q, V, Δ_in, Δ_out, ε, β):
    q = (1 - ε + ε * β) ** 2 * q
    m = (1 - ε + ε * β) * m

    return estimation_error(m, q, V, tuple())
