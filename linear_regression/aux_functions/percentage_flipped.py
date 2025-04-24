from math import pi, sqrt, erf, exp, erfc, inf
from math import gamma as gamma_fun
from scipy.integrate import quad
import numpy as np
from numba import njit


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
    if float(p) == inf:
        if gamma <= 1:
            return erf(epsilon / sqrt(pi) * sqrt(q_latent - m**2) / sqrt(q))
        else:
            return erf(epsilon / sqrt(pi) * sqrt(q_features - m**2) / gamma / sqrt(q))
    else:
        raise NotImplementedError


def percentage_misclassified_hastie_model(
    m: float, q: float, rho: float, epsilon: float, p
) -> float:
    η = m**2 / (q * rho)
    if float(p) == inf:
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
    if p == "inf":
        a
    else:
        pstar = 1 / (1 - 1 / p)
        C = (gamma_fun((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)
