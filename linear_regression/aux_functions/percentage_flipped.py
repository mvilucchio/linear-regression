from math import pi, atan, sqrt, erf
from scipy.special import owens_t
from math import gamma
from scipy.integrate import dblquad, quad
import numpy as np
from numba import njit


# ------------------------------- Direct Space ------------------------------- #


def percentage_flipped_direct_space_true_label(
    m: float, q: float, rho: float, epsilon: float
) -> float:
    raise NotImplementedError("Not implemented")
    η = m**2 / (q * rho)
    return (
        atan(sqrt(η) / sqrt(1 - η)) / pi
        + 0.5 * erf(epsilon * sqrt(0.5 * (1 - η)))
        + 2 * owens_t(epsilon * sqrt((1 - η)), sqrt(η / (1 - η)))
    )


def percentage_flipped_direct_space(m: float, q: float, rho: float, epsilon: float, p) -> float:
    η = m**2 / (q * rho)
    if p == "inf":
        C = 1 / sqrt(pi)
        return erf(epsilon * sqrt(1 - η) * C)
    else:
        pstar = p / (p - 1)
        C = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
        return erf(epsilon * sqrt(1 - η) * C)


# ------------------------------ Linear Features ----------------------------- #


def percentage_flipped_linear_features(
    m: float, q: float, rho: float, epsilon: float, p, gamma_val: float
) -> float:
    if p == "inf":
        C = 1 / sqrt(pi)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)
    else:
        pstar = p / (p - 1)
        C = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)


@njit(error_model="numpy", fastmath=False)
def gaussian_pdf(x, y, cov):
    a = np.array([x, y])
    return np.exp(-0.5 * a @ np.linalg.inv(cov) @ a) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def percentage_misclassified_linear_features(
    m: float, q: float, rho: float, epsilon: float, p, gamma_val: float
) -> float:
    if p == "inf":
        C = 1 / sqrt(pi) * gamma_val ** (1 / 2)
        # return (
        #     0.5
        #     - 0.5
        #     * quad(
        #         lambda x: np.exp(-(x**2) / (2 * q))
        #         / np.sqrt(2 * np.pi * q)
        #         * erf(m * x / sqrt(2 * q * (q * rho - m**2))),
        #         epsilon * sqrt(2 * q) * sqrt(1 - m**2 / (q * rho)) * C,
        #         np.inf,
        #     )[0]
        # )
        # return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)
        mat = np.array([[rho, m], [m, q]])
        int_val = dblquad(
            lambda x, y: gaussian_pdf(x, y, mat)
            * np.heaviside(
                -np.sign(x) * y + epsilon * sqrt(q) * sqrt(1 - m**2 / (q * rho)) * C * np.sqrt(2),
                1,
            ),
            -np.inf,
            np.inf,
            lambda x: -np.inf,
            lambda x: np.inf,
        )[0]
        return int_val
    else:
        pstar = p / (p - 1)
        C = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar) * gamma_val ** (1 / 2 - 1 / p)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)


# ---------------------------- Non-Linear Features --------------------------- #
