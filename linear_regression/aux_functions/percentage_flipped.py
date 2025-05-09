from math import pi, sqrt, erf, exp, erfc, inf
from math import gamma as gamma_fun
from scipy.integrate import quad, dblquad
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
    if gamma <= 1:
        AA = epsilon * np.sqrt(q_latent - m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    else:
        AA = epsilon * np.sqrt(q_features - m**2 / gamma) * np.sqrt(2 / np.pi) / np.sqrt(gamma)

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
    # if gamma <= 1:
    #     return erf(
    #         epsilon
    #         * np.sqrt(q_latent - m**2 / gamma)
    #         * np.sqrt(1 / np.pi)
    #         / np.sqrt(q)
    #         * np.sqrt(gamma)
    #     )
    # else:
    #     return erf(
    #         epsilon
    #         * np.sqrt(q_features - m**2 / gamma)
    #         / np.sqrt(gamma)
    #         * np.sqrt(1 / np.pi)
    #         / np.sqrt(q)
    #     )
    # else:
    #     raise NotImplementedError


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
    if gamma <= 1:
        AA = epsilon * np.sqrt(q_latent - m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
    else:
        AA = epsilon * np.sqrt(q_features - m**2 / gamma) / np.sqrt(gamma) * np.sqrt(2 / np.pi)

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

    int_val_1 = quad(
        lambda x: np.exp(-(x**2) / (2 * q))
        / np.sqrt(2 * np.pi * q)
        * erfc(m / np.sqrt(gamma) * x / np.sqrt(2 * q * (q * rho - m**2 / gamma)))
        * np.heaviside(-AA - x, 0),
        -np.inf,
        np.inf,
    )[0]
    int_val_2 = quad(
        lambda x: np.exp(-(x**2) / (2 * q))
        / np.sqrt(2 * np.pi * q)
        * (1 + erf(m / np.sqrt(gamma) * x / np.sqrt(2 * q * (q * rho - m**2 / gamma))))
        * np.heaviside(x - AA, 0),
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
