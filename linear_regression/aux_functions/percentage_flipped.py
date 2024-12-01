from math import pi, sqrt, erf, exp, erfc
from math import gamma
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
        pstar = p / (p - 1)
        A = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar) * epsilon * sqrt(2 * q) * sqrt(1 - η)

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
        pstar = p / (p - 1)
        C = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
        return erf(epsilon * sqrt(1 - η) * C)


# ------------------------------ Linear Features ----------------------------- #


def percentage_flipped_linear_features(m: float, q: float, rho: float, epsilon: float, p) -> float:
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
    m: float, q: float, rho: float, epsilon: float, p
) -> float:
    if p == "inf":
        A = epsilon / sqrt(pi) * sqrt(2 * q) * sqrt(1 - m**2 / (q * rho))
    else:
        pstar = p / (p - 1)
        A = (
            (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
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


# def percentage_misclassified_linear_features(
#     m: float, q: float, rho: float, epsilon: float, p, gamma_val: float
# ) -> float:
#     if p == "inf":
#         C = 1 / sqrt(pi)
#         int_val_1 = quad(
#             lambda x: exp(-(x**2) / (2 * q))
#             / sqrt(2 * pi * q)
#             * erfc(m * x / sqrt(2 * q * (q * rho - m**2)))
#             * np.heaviside(epsilon * sqrt(2 * q) * sqrt(1 - m**2 / (q * rho)) * C - x, 1),
#             -np.inf,
#             np.inf,
#         )[0]
#         int_val_2 = quad(
#             lambda x: exp(-(x**2) / (2 * q))
#             / sqrt(2 * pi * q)
#             * erfc(-m * x / sqrt(2 * q * (q * rho - m**2)))
#             * np.heaviside(x - epsilon * sqrt(2 * q) * sqrt(1 - m**2 / (q * rho)) * C, 1),
#             -np.inf,
#             np.inf,
#         )[0]
#         return 0.5 * (int_val_1 + int_val_2)
#         # return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)
#         mat = np.array([[rho, m], [m, q]])
#         int_val = dblquad(
#             lambda x, y: gaussian_pdf(x, y, mat)
#             * np.heaviside(
#                 -np.sign(x) * y + epsilon * sqrt(q) * sqrt(1 - m**2 / (q * rho)) * C * np.sqrt(2),
#                 1,
#             ),
#             -np.inf,
#             np.inf,
#             lambda x: -np.inf,
#             lambda x: np.inf,
#         )[0]
#         return int_val
#     else:
#         pstar = p / (p - 1)
#         C = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar) * gamma_val ** (1 / 2 - 1 / p)
#         return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * C)


# ---------------------------- Non-Linear Features --------------------------- #
