from numpy import pi
from math import sqrt, exp, erfc, erf
from numba import njit, vectorize
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from ..aux_functions.misc import gaussian
from ..aux_functions.loss_functions import logistic_loss, exponential_loss
from ..aux_functions.moreau_proximals import moreau_loss_Logistic, moreau_loss_Exponential
from ..utils.integration_utils import line_borders_hinge_above

import warnings
from scipy.integrate import IntegrationWarning

BIG_NUMBER = 20

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Training Errors for Regression                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@vectorize(["float64(float64, float64, float64, float64, float64, float64, float64)"])
def training_error_l2_loss(m, q, sigma, delta_in, delta_out, percentage, beta):
    return (
        1
        + q
        + delta_in
        - delta_in * percentage
        - 2 * m * (1 + (-1 + beta) * percentage)
        + percentage * (-1 + beta**2 + delta_out)
    ) / (2 * (1 + sigma) ** 2)


@vectorize(["float64(float64, float64, float64, float64, float64, float64, float64)"])
def training_error_l1_loss(m, q, sigma, delta_in, delta_out, percentage, beta):
    return (
        sqrt(2 / pi)
        * (
            -(
                (sqrt(1.0 - 2.0 * m + q + delta_in) * (-1 + percentage))
                / exp(sigma**2 / (2.0 * (1.0 - 2.0 * m + q + delta_in)))
            )
            + (percentage * sqrt(q - 2 * m * beta + beta**2 + delta_out))
            / exp(sigma**2 / (2.0 * (q - 2 * m * beta + beta**2 + delta_out)))
        )
        + sigma * (-1 + percentage) * erfc(sigma / (sqrt(2) * sqrt(1.0 - 2.0 * m + q + delta_in)))
        - sigma
        * percentage
        * erfc(sigma / (sqrt(2) * sqrt(q - 2 * m * beta + beta**2 + delta_out)))
    )


@vectorize(["float64(float64, float64, float64, float64, float64, float64, float64, float64)"])
def training_error_huber_loss(m, q, sigma, delta_in, delta_out, percentage, beta, a):
    return (
        -(
            (
                a
                * sqrt(2 / pi)
                * (1 + sigma)
                * (1 + 2 * sigma)
                * (
                    exp(
                        (
                            a**2
                            * (1 + sigma) ** 2
                            * (
                                -(1 / (1 - 2 * m + q + delta_in))
                                + 1 / (q - 2 * m * beta + beta**2 + delta_out)
                            )
                        )
                        / 2.0
                    )
                    * sqrt(1 - 2 * m + q + delta_in)
                    * (-1 + percentage)
                    - percentage * sqrt(q - 2 * m * beta + beta**2 + delta_out)
                )
            )
            / exp((a**2 * (1 + sigma) ** 2) / (2.0 * (q - 2 * m * beta + beta**2 + delta_out)))
        )
        + (
            1
            + q
            + delta_in
            + 2 * m * (-1 + percentage)
            - (1 + q + a**2 * (1 + sigma) ** 2 * (1 + 2 * sigma) + delta_in) * percentage
        )
        * erf((a * (1 + sigma)) / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
        + percentage
        * (q + a**2 * (1 + sigma) ** 2 * (1 + 2 * sigma) - 2 * m * beta + beta**2 + delta_out)
        * erf((a * (1 + sigma)) / (sqrt(2) * sqrt(q - 2 * m * beta + beta**2 + delta_out)))
        - a**2
        * (1 + sigma) ** 2
        * (1 + 2 * sigma)
        * erfc((a * (1 + sigma)) / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
    ) / (2.0 * (1 + sigma) ** 2)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Training Errors for Classification                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # Probit model                            # # # # #


@njit(error_model="numpy", fastmath=False)
def integral_training_error_Hinge_probit(ξ, y, m, q, Σ, delta):
    η = m**2 / q
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + erf(y * sqrt(0.5 * η / (1 - η + delta)) * ξ))
        * (1 - y * sqrt(q) * ξ - Σ)
    )


def training_error_Hinge_loss_probit(m: float, q: float, Σ: float, delta: float) -> float:
    domains = line_borders_hinge_above(m, q, Σ)

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Hinge_probit, *domain, args=(y_val, m, q, Σ, delta)
        )[0]

    return int_value


# -----
def integral_training_error_Logistic_probit(ξ, y, m, q, Σ, delta):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Logistic, args=(y, sqrt(q) * ξ, Σ))["x"]
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η + delta))))
        * logistic_loss(y, proximal)
    )


def training_error_Logistic_loss_probit(m, q, Σ, delta):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Logistic_probit, *domain, args=(y_val, m, q, Σ, delta)
        )[0]

    return int_value


# -----


def integral_training_error_Exponential_probit(ξ, y, m, q, Σ, delta):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Exponential, args=(y, sqrt(q) * ξ, Σ))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η + delta))))
        * exponential_loss(y, proximal)
    )


def training_error_Exponential_loss_probit(m, q, Σ, delta):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Exponential_probit, *domain, args=(y_val, m, q, Σ, delta)
        )[0]

    return int_value


# # # # # No Noise model                          # # # # #


@njit(error_model="numpy", fastmath=False)
def integral_training_error_Hinge_no_noise(ξ, y, m, q, Σ):
    η = m**2 / q
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + erf(y * sqrt(0.5 * η / (1 - η)) * ξ))
        * (1 - y * sqrt(q) * ξ - Σ)
    )


def training_error_Hinge_loss_no_noise(m, q, Σ):
    domains = line_borders_hinge_above(m, q, Σ)

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Hinge_no_noise, domain[0], domain[1], args=(y_val, m, q, Σ)
        )[0]

    return int_value


# -----
def integral_training_error_Logistic_no_noise(ξ, y, m, q, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Logistic, args=(y, sqrt(q) * ξ, Σ))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1) * (1 + erf(y * sqrt(0.5 * η / (1 - η)) * ξ)) * logistic_loss(y, proximal)
    )


def training_error_Logistic_loss_no_noise(m, q, Σ):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Logistic_no_noise, *domain, args=(y_val, m, q, Σ)
        )[0]

    return int_value


# -----
def integral_training_error_Exponential_no_noise(ξ, y, m, q, Σ):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Exponential, args=(y, sqrt(q) * ξ, Σ))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * exponential_loss(y, proximal)
    )


def training_error_Exponential_loss_no_noise(m, q, Σ):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Exponential_no_noise, *domain, args=(y_val, m, q, Σ)
        )[0]

    return int_value


# # # # # Noise model                             # # # # #


def integral_training_error_Hinge_loss_single_noise(ξ, y, m, q, Σ, delta):
    raise NotImplementedError


def training_error_Hinge_loss_single_noise(m, q, sigma, delta):
    raise NotImplementedError
