from math import exp, sqrt, erf, erfc, inf, pi, erf
from numpy import max as np_max
from numpy import abs as np_abs
import numpy as np
from numpy.linalg import norm, det, inv
from numpy.random import normal
from scipy.integrate import quad, dblquad
from math import gamma as gamma_fun
from scipy.integrate import quad, dblquad
import numpy as np


# ---------------------------------------------------------------------------- #
#                                    errors                                    #
# ---------------------------------------------------------------------------- #


# ----------------------------- errors regression ---------------------------- #


# @njit(error_model="numpy", fastmath=True)
def estimation_error(m, q, V, **args):
    return 1 + q - 2.0 * m


# @njit
def angle_teacher_student(m, q, V, **args):
    return np.arccos(m / np.sqrt(q)) / pi


def margin_probit_classif(m, q, V, delta):
    return (4 * m * sqrt(2 * pi**3)) / sqrt(delta + 1)


# errors
def gen_error(m, q, V, delta_in, delta_out, percentage, beta):
    return q - 2 * m * (1 + (-1 + beta) * percentage) + 1 + percentage * (-1 + beta**2)


def excess_gen_error(m, q, V, delta_in, delta_out, percentage, beta):
    gen_err_BO_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (
        1 - percentage
    ) ** 2 * (beta - 1) ** 2
    return gen_error(m, q, V, delta_in, delta_out, percentage, beta) - gen_err_BO_alpha_inf


def excess_gen_error_oracle_rescaling(m, q, V, delta_in, delta_out, percentage, beta):
    oracle_norm = 1 - percentage + percentage * beta
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return excess_gen_error(m_prime, q_prime, V, delta_in, delta_out, percentage, beta)


def estimation_error_rescaled(m, q, V, delta_in, delta_out, percentage, beta, norm_const):
    m = m / norm_const
    q = q / (norm_const**2)

    return estimation_error(m, q, V)


def estimation_error_oracle_rescaling(m, q, V, delta_in, delta_out, percentage, beta):
    oracle_norm = 1.0  # abs(1 - percentage + percentage * beta)
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return estimation_error(m_prime, q_prime, V)


def gen_error_BO(m, q, V, delta_in, delta_out, percentage, beta):
    return (1 + percentage * (-1 + beta**2) - (1 + percentage * (-1 + beta)) ** 2 * q) - (
        (1 - percentage) * percentage**2 * (1 - beta) ** 2
        + percentage * (1 - percentage) ** 2 * (beta - 1) ** 2
    )


def gen_error_BO_old(m, q, V, delta_in, delta_out, percentage, beta):
    q = (1 - percentage + percentage * beta) ** 2 * q
    m = (1 - percentage + percentage * beta) * m

    return estimation_error(m, q, V, tuple())


# --------------------------- errors classification -------------------------- #


# def classification_adversarial_error(m, q, P, eps, pstar):
#     Iminus = quad(
#         lambda x: np.exp(-0.5 * x**2 / q) * erfc(m * x / np.sqrt(2 * q * (q - m**2))),
#         -eps * P ** (1 / pstar),
#         np.inf,
#         epsabs=1e-10,
#     )[0]
#     Iplus = quad(
#         lambda x: np.exp(-0.5 * x**2 / q) * (1 + erf(m * x / np.sqrt(2 * q * (q - m**2)))),
#         -np.inf,
#         eps * P ** (1 / pstar),
#         epsabs=1e-10,
#     )[0]
#     return 0.5 * (Iminus + Iplus) / np.sqrt(2 * pi * q)


def classification_adversarial_error(m, q, P, eps, pstar):
    if float(pstar) not in (1.0, 2.0):
        raise ValueError("pstar must be 1 or 2 for this function")
    rho = 1.0
    if pstar == 1.0:
        AA = eps * np.sqrt(q) * np.sqrt(2 / pi)
    else:
        AA = (
            eps
            * np.sqrt(q)
            * np.sqrt(2)
            / np.sqrt(pi) ** (1 / pstar)
            * (np.sqrt(pi) / 2) ** (1 / pstar)
        )
    return dblquad(
        lambda nu, lamb: (
            exp((-2 * m * lamb * nu + q * nu**2 + lamb**2 * rho) / (2.0 * (m**2 - q * rho)))
            * np.heaviside(+AA - lamb * np.sign(nu), 0.0)
        )
        / (2.0 * np.pi * sqrt(-(m**2) + q * rho)),
        -np.inf,
        np.inf,
        lambda nu: -np.inf,
        lambda nu: np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


def boundary_error_direct_space(m, q, P, eps, pstar):
    # if pstar != 1.0:
    #     raise ValueError("pstar must be 1 for this function")
    # rho = 1.0
    # AA = eps * np.sqrt(q - m**2 / rho) * np.sqrt(2 / pi)
    if float(pstar) not in (1.0, 2.0):
        raise ValueError("pstar must be 1 or 2 for this function")
    rho = 1.0
    if pstar == 1.0:
        AA = eps * np.sqrt(q - m**2) * np.sqrt(2 / pi)
    else:
        AA = (
            eps
            * np.sqrt(q - m**2)
            * np.sqrt(2)
            / np.sqrt(pi) ** (1 / pstar)
            * (np.sqrt(pi) / 2) ** (1 / pstar)
        )
    return dblquad(
        lambda nu, lamb: (
            exp((-2 * m * lamb * nu + q * nu**2 + lamb**2 * rho) / (2.0 * (m**2 - q * rho)))
            * np.heaviside(+AA - lamb * np.sign(nu), 0.0)
            * np.heaviside(np.sign(nu) * np.sign(lamb), 0.0)
        )
        / (2.0 * np.pi * sqrt(-(m**2) + q * rho)),
        -np.inf,
        np.inf,
        lambda nu: -np.inf,
        lambda nu: np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


def misclassification_error_direct_space(m, q, P, eps, pstar):
    # if pstar != 1.0:
    #     raise ValueError("pstar must be 1 for this function")
    # rho = 1.0
    # AA = eps * np.sqrt(q - m**2 / rho) * np.sqrt(2 / pi)
    if float(pstar) not in (1.0, 2.0):
        raise ValueError("pstar must be 1 or 2 for this function")
    rho = 1.0
    if pstar == 1.0:
        AA = eps * np.sqrt(q - m**2) * np.sqrt(2 / pi)
    else:
        AA = (
            eps
            * np.sqrt(q - m**2)
            * np.sqrt(2)
            / np.sqrt(pi) ** (1 / pstar)
            * (np.sqrt(pi) / 2) ** (1 / pstar)
        )
    return dblquad(
        lambda nu, lamb: (
            exp((-2 * m * lamb * nu + q * nu**2 + lamb**2 * rho) / (2.0 * (m**2 - q * rho)))
            * np.heaviside(+AA - lamb * np.sign(nu), 0.0)
        )
        / (2.0 * np.pi * sqrt(-(m**2) + q * rho)),
        -np.inf,
        np.inf,
        lambda nu: -np.inf,
        lambda nu: np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


def flipped_error_direct_space(m, q, P, eps, pstar):
    # if pstar != 1.0:
    #     raise ValueError("pstar must be 1 for this function")
    # rho = 1.0
    # AA = eps * np.sqrt(q - m**2 / rho) * np.sqrt(2 / pi)
    if float(pstar) not in (1.0, 2.0):
        raise ValueError("pstar must be 1 or 2 for this function")
    rho = 1.0
    if pstar == 1.0:
        AA = eps * np.sqrt(q) * np.sqrt(2 / pi)
    else:
        AA = (
            eps
            * np.sqrt(q)
            * np.sqrt(2)
            / np.sqrt(pi) ** (1 / pstar)
            * (np.sqrt(pi) / 2) ** (1 / pstar)
        )
    return quad(
        lambda lamb: (exp(-0.5 * lamb**2 / q) * np.heaviside(+AA - lamb * np.sign(lamb), 0.0))
        / sqrt(2.0 * np.pi * q),
        -np.inf,
        np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


def classification_adversarial_error_latent(m, q, q_features, q_latent, rho, P, eps, gamma, pstar):
    if float(pstar) not in (1.0, 2.0):
        raise ValueError("pstar must be 1 for this function")

    # if gamma <= 1:
    #     AA = eps * np.sqrt(q_latent) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    # else:
    #     AA = eps * np.sqrt(q_features) / np.sqrt(gamma) * np.sqrt(2 / np.pi)
    # if gamma <= 1:
    #     first_term = np.sqrt(q_latent) * np.sqrt(gamma)
    # else:
    #     first_term = np.sqrt(q_features) / np.sqrt(gamma)

    # if pstar == 1.0:
    #     second_term = np.sqrt(2 / pi)
    # elif pstar == 2.0:
    #     second_term = np.sqrt(2) / np.sqrt(pi) ** (1 / pstar) * (np.sqrt(pi) / 2) ** (1 / pstar)

    # AA = eps * first_term * second_term

    if gamma <= 1 and float(pstar) == 1.0:
        AA = np.sqrt(q_latent) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    elif gamma <= 1 and float(pstar) == 2.0:
        AA = np.sqrt(q_latent) * np.sqrt(gamma)
    elif gamma > 1 and float(pstar) == 1.0:
        AA = np.sqrt(q_features) * np.sqrt(2 / np.pi) / np.sqrt(gamma)
    elif gamma > 1 and float(pstar) == 2.0:
        AA = np.sqrt(q_latent) * np.sqrt(gamma)

    AA = eps * AA
    return dblquad(
        lambda nu, lamb: (
            exp((-2 * m * lamb * nu + q * nu**2 + lamb**2 * rho) / (2.0 * (m**2 - q * rho)))
            * np.heaviside(+AA - lamb * np.sign(nu), 0.0)
        )
        / (2.0 * np.pi * sqrt(-(m**2) + q * rho)),
        -np.inf,
        np.inf,
        lambda nu: -np.inf,
        lambda nu: np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


# ------------------------------- Direct Space ------------------------------- #


def percentage_misclassified_direct_space(
    m: float, q: float, rho: float, epsilon: float, p
) -> float:
    η = m**2 / (q * rho)
    if p == "inf":
        A = epsilon / sqrt(pi) * sqrt(2 * q) * sqrt(1 - η)
    else:
        pstar = 1 / (1 - 1 / p)
        A = (
            (gamma_fun((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
            * epsilon
            * sqrt(2 * q)
            * sqrt(1 - η)
        )

    int_val_1 = quad(
        lambda x: exp(-(x**2) / (2 * q))
        / sqrt(2 * pi * q)
        * erfc(m * x / sqrt(2 * q * (q * rho - m**2)))
        * np.heaviside(-A - x, 0),
        -np.inf,
        np.inf,
    )[0]
    int_val_2 = quad(
        lambda x: exp(-(x**2) / (2 * q))
        / sqrt(2 * pi * q)
        * (1 + erf(m * x / sqrt(2 * q * (q * rho - m**2))))
        * np.heaviside(x - A, 0),
        -np.inf,
        np.inf,
    )[0]
    return 1 - 0.5 * (int_val_1 + int_val_2)


def percentage_flipped_direct_space(m: float, q: float, rho: float, epsilon: float, p) -> float:
    η = m**2 / (q * rho)
    if p == "inf":
        C = 1 / sqrt(pi)
        return erf(epsilon * sqrt(1 - η) * C)
    else:
        pstar = 1 / (1 - 1 / p)
        C = (gamma_fun((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
        return erf(epsilon * sqrt(1 - η) * C)


# ------------------------------- Hastie Model ------------------------------- #


def boundary_error_fair_hastie_model(
    m: float,
    q: float,
    q_latent: float,
    q_features: float,
    rho: float,
    epsilon: float,
    gamma: float,
    p,
) -> float:
    # if float(p) == inf:
    # if gamma <= 1:
    #     AA = epsilon * np.sqrt(q_latent - m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    # else:
    #     AA = epsilon * np.sqrt(q_features - m**2 / gamma) * np.sqrt(2 / np.pi) / np.sqrt(gamma)

    if gamma <= 1 and float(p) == inf:
        AA = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    elif gamma <= 1 and float(p) == 2.0:
        AA = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(gamma)
    elif gamma > 1 and float(p) == inf:
        AA = np.sqrt(q_features - m**2 / gamma) * np.sqrt(2 / np.pi) / np.sqrt(gamma)
    elif gamma > 1 and float(p) == 2.0:
        AA = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(gamma)

    AA = epsilon * AA
    # if float(p) == inf:
    #     second_term = np.sqrt(2 / pi) / np.sqrt(gamma)
    # elif float(p) == 2.0:
    #     pstar = 2.0
    #     second_term = np.sqrt(2) / np.sqrt(pi) ** (1 / pstar) * (np.sqrt(pi) / 2) ** (1 / pstar)

    # AA = epsilon * first_term * second_term
    # if gamma <= 1:
    #     first_term = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(gamma)
    # else:
    #     # first_term = np.sqrt(q_features - m**2 / gamma) / np.sqrt(gamma)
    #     first_term = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(gamma)

    # if float(p) == inf:
    #     second_term = np.sqrt(2 / pi) / np.sqrt(gamma)
    # elif float(p) == 2.0:
    #     pstar = 2.0
    #     second_term = np.sqrt(2) / np.sqrt(pi) ** (1 / pstar) * (np.sqrt(pi) / 2) ** (1 / pstar)

    # AA = epsilon * first_term * second_term
    return dblquad(
        lambda nu, lamb: (
            exp((-2 * m * lamb * nu + q * nu**2 + lamb**2 * rho) / (2.0 * (m**2 - q * rho)))
            * np.heaviside(+AA - lamb * np.sign(nu), 0.0)
            * np.heaviside(np.sign(lamb) * np.sign(nu), 0.0)
        )
        / (2.0 * np.pi * sqrt(-(m**2) + q * rho)),
        -np.inf,
        np.inf,
        lambda nu: -np.inf,
        lambda nu: np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


def percentage_flipped_hastie_model(
    m: float,
    q: float,
    q_latent: float,
    q_features: float,
    rho: float,
    epsilon: float,
    gamma: float,
    p,
) -> float:
    # if float(p) == inf:
    if gamma <= 1:
        AA = epsilon * np.sqrt(q_latent - m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    else:
        AA = epsilon * np.sqrt(q_features - m**2 / gamma) * np.sqrt(2 / np.pi) / np.sqrt(gamma)

    return quad(
        lambda x: np.exp(-(x**2) / (2 * q))
        / np.sqrt(2 * np.pi * q)
        * np.heaviside(+AA - np.sign(x) * x, 0),
        -np.inf,
        np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


def percentage_misclassified_hastie_model(
    m: float,
    q: float,
    q_latent: float,
    q_features: float,
    rho: float,
    epsilon: float,
    gamma: float,
    p,
) -> float:
    # if gamma <= 1:
    #     AA = epsilon * np.sqrt(q_latent - m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    # else:
    #     AA = epsilon * np.sqrt(q_features - m**2 / gamma) / np.sqrt(gamma) * np.sqrt(2 / np.pi)

    # if gamma <= 1:
    #     first_term = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(gamma)
    # else:
    #     first_term = np.sqrt(q_features - m**2 / gamma) / np.sqrt(gamma)

    # if float(p) == inf:
    #     second_term = np.sqrt(2 / pi)
    # elif float(p) == 2.0:
    #     pstar = 2.0
    #     second_term = np.sqrt(2) / np.sqrt(pi) ** (1 / pstar) * (np.sqrt(pi) / 2) ** (1 / pstar)

    # AA = epsilon * first_term * second_term

    if gamma <= 1 and float(p) == inf:
        AA = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    elif gamma <= 1 and float(p) == 2.0:
        AA = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(gamma)
    elif gamma > 1 and float(p) == inf:
        AA = np.sqrt(q_features - m**2 / gamma) * np.sqrt(2 / np.pi) / np.sqrt(gamma)
    elif gamma > 1 and float(p) == 2.0:
        AA = np.sqrt(q_latent - m**2 / gamma) * np.sqrt(gamma)

    AA = epsilon * AA

    return dblquad(
        lambda nu, lamb: (
            exp((-2 * m * lamb * nu + q * nu**2 + lamb**2 * rho) / (2.0 * (m**2 - q * rho)))
            * np.heaviside(+AA - lamb * np.sign(nu), 0.0)
        )
        / (2.0 * np.pi * sqrt(-(m**2) + q * rho)),
        -np.inf,
        np.inf,
        lambda nu: -np.inf,
        lambda nu: np.inf,
        epsabs=1e-7,
        epsrel=1e-7,
    )[0]


# ------------------------------ Linear Features ----------------------------- #


def percentage_flipped_linear_features(m: float, q: float, rho: float, epsilon: float, p) -> float:
    if p == "inf":
        C = 1 / sqrt(pi)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)
    else:
        pstar = 1 / (1 - 1 / p)
        C = (gamma_fun((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)


def percentage_misclassified_linear_features(
    m: float, q: float, rho: float, epsilon: float, p
) -> float:
    if p == "inf":
        A = epsilon / sqrt(pi) * sqrt(2 * q) * sqrt(1 - m**2 / (q * rho))
    else:
        pstar = 1 / (1 - 1 / p)
        A = (
            (gamma_fun((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
            * epsilon
            * sqrt(2 * q)
            * sqrt(1 - m**2 / (q * rho))
        )

    int_val_1 = quad(
        lambda x: exp(-(x**2) / (2 * q))
        / sqrt(2 * pi * q)
        * erfc(m * x / sqrt(2 * q * (q * rho - m**2)))
        * np.heaviside(-A - x, 0),
        -np.inf,
        np.inf,
    )[0]
    int_val_2 = quad(
        lambda x: exp(-(x**2) / (2 * q))
        / sqrt(2 * pi * q)
        * (1 + erf(m * x / sqrt(2 * q * (q * rho - m**2))))
        * np.heaviside(x - A, 0),
        -np.inf,
        np.inf,
    )[0]
    return 1 - 0.5 * (int_val_1 + int_val_2)


# ---------------------------- Non-Linear Features --------------------------- #
def percentage_flipped_nonlinear_features(
    m: float, q: float, rho: float, epsilon: float, p
) -> float:
    raise NotImplementedError


# ---------------------------------------------------------------------------- #
#                                   overlaps                                   #
# ---------------------------------------------------------------------------- #
def m_overlap(m, q, V, **args):
    return m


def q_overlap(m, q, V, **args):
    return q


def V_overlap(m, q, V, **args):
    return V
