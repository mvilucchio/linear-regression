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
def training_error_l2_loss(m, q, V, delta_in, delta_out, percentage, beta):
    return (
        1
        + q
        + delta_in
        - delta_in * percentage
        - 2 * m * (1 + (-1 + beta) * percentage)
        + percentage * (-1 + beta**2 + delta_out)
    ) / (2 * (1 + V) ** 2)


@vectorize(["float64(float64, float64, float64, float64, float64, float64, float64)"])
def training_error_l1_loss(m, q, V, delta_in, delta_out, percentage, beta):
    return (
        sqrt(2 / pi)
        * (
            -(
                (sqrt(1.0 - 2.0 * m + q + delta_in) * (-1 + percentage))
                / exp(V**2 / (2.0 * (1.0 - 2.0 * m + q + delta_in)))
            )
            + (percentage * sqrt(q - 2 * m * beta + beta**2 + delta_out))
            / exp(V**2 / (2.0 * (q - 2 * m * beta + beta**2 + delta_out)))
        )
        + V * (-1 + percentage) * erfc(V / (sqrt(2) * sqrt(1.0 - 2.0 * m + q + delta_in)))
        - V * percentage * erfc(V / (sqrt(2) * sqrt(q - 2 * m * beta + beta**2 + delta_out)))
    )


@vectorize(["float64(float64, float64, float64, float64, float64, float64, float64, float64)"])
def training_error_huber_loss(m, q, V, delta_in, delta_out, percentage, beta, a):
    return (
        -(
            (
                a
                * sqrt(2 / pi)
                * (1 + V)
                * (1 + 2 * V)
                * (
                    exp(
                        (
                            a**2
                            * (1 + V) ** 2
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
            / exp((a**2 * (1 + V) ** 2) / (2.0 * (q - 2 * m * beta + beta**2 + delta_out)))
        )
        + (
            1
            + q
            + delta_in
            + 2 * m * (-1 + percentage)
            - (1 + q + a**2 * (1 + V) ** 2 * (1 + 2 * V) + delta_in) * percentage
        )
        * erf((a * (1 + V)) / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
        + percentage
        * (q + a**2 * (1 + V) ** 2 * (1 + 2 * V) - 2 * m * beta + beta**2 + delta_out)
        * erf((a * (1 + V)) / (sqrt(2) * sqrt(q - 2 * m * beta + beta**2 + delta_out)))
        - a**2
        * (1 + V) ** 2
        * (1 + 2 * V)
        * erfc((a * (1 + V)) / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
    ) / (2.0 * (1 + V) ** 2)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Training Errors for Classification                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # Probit model                            # # # # #


@njit(error_model="numpy", fastmath=False)
def integral_training_error_Hinge_probit(ξ, y, m, q, V, delta):
    η = m**2 / q
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + erf(y * sqrt(0.5 * η / (1 - η + delta)) * ξ))
        * (1 - y * sqrt(q) * ξ - V)
    )


def training_error_Hinge_loss_probit(m: float, q: float, V: float, delta: float) -> float:
    domains = line_borders_hinge_above(m, q, V)

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Hinge_probit, *domain, args=(y_val, m, q, V, delta)
        )[0]

    return int_value


# -----
def integral_training_error_Logistic_probit(ξ, y, m, q, V, delta):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Logistic, args=(y, sqrt(q) * ξ, V))["x"]
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η + delta))))
        * logistic_loss(y, proximal)
    )


def training_error_Logistic_loss_probit(m, q, V, delta):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Logistic_probit, *domain, args=(y_val, m, q, V, delta)
        )[0]

    return int_value


# -----


def integral_training_error_Exponential_probit(ξ, y, m, q, V, delta):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Exponential, args=(y, sqrt(q) * ξ, V))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η + delta))))
        * exponential_loss(y, proximal)
    )


def training_error_Exponential_loss_probit(m, q, V, delta):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Exponential_probit, *domain, args=(y_val, m, q, V, delta)
        )[0]

    return int_value


# # # # # No Noise model                          # # # # #


@njit(error_model="numpy", fastmath=False)
def integral_training_error_Hinge_no_noise(ξ, y, m, q, V):
    η = m**2 / q
    return (
        0.5
        * gaussian(ξ, 0, 1)
        * (1 + erf(y * sqrt(0.5 * η / (1 - η)) * ξ))
        * (1 - y * sqrt(q) * ξ - V)
    )


def training_error_Hinge_loss_no_noise(m, q, V):
    domains = line_borders_hinge_above(m, q, V)

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Hinge_no_noise, domain[0], domain[1], args=(y_val, m, q, V)
        )[0]

    return int_value


# -----
def integral_training_error_Logistic_no_noise(ξ, y, m, q, V):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Logistic, args=(y, sqrt(q) * ξ, V))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1) * (1 + erf(y * sqrt(0.5 * η / (1 - η)) * ξ)) * logistic_loss(y, proximal)
    )


def training_error_Logistic_loss_no_noise(m, q, V):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Logistic_no_noise, *domain, args=(y_val, m, q, V)
        )[0]

    return int_value


# -----
def integral_training_error_Exponential_no_noise(ξ, y, m, q, V):
    η = m**2 / q
    proximal = minimize_scalar(moreau_loss_Exponential, args=(y, sqrt(q) * ξ, V))["x"]
    return 0.5 * (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * exponential_loss(y, proximal)
    )


def training_error_Exponential_loss_no_noise(m, q, V):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value = 0.0
    for y_val, domain in domains:
        int_value += quad(
            integral_training_error_Exponential_no_noise, *domain, args=(y_val, m, q, V)
        )[0]

    return int_value


# # # # # Noise model                             # # # # #


def integral_training_error_Hinge_loss_single_noise(ξ, y, m, q, V, delta):
    raise NotImplementedError


def training_error_Hinge_loss_single_noise(m, q, V, delta):
    raise NotImplementedError
