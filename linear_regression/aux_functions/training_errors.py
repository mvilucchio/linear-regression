from numpy import pi
from math import sqrt, exp, erfc, erf
from numba import njit


@njit
def training_error_l2_loss(m, q, sigma, delta_in, delta_out, percentage, beta):
    return (
        1
        + q
        + delta_in
        - delta_in * percentage
        - 2 * m * (1 + (-1 + beta) * percentage)
        + percentage * (-1 + beta**2 + delta_out)
    ) / (
        2 * (1 + sigma) ** 2
    )


@njit
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


@njit
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
    ) / (
        2.0 * (1 + sigma) ** 2
    )
