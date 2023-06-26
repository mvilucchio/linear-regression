from numba import njit
import numpy as np
from scipy.integrate import dblquad
from ..utils.integration_utils import (
    divide_integration_borders_multiple_grid,
    find_integration_borders_square,
)
from ..aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
    f_out_Bayes_decorrelated_noise,
)


@njit(error_model="numpy", fastmath=True)
def order_parameters_BO_single_noise(alpha: float, delta_in: float):
    q = 0.5 * (
        1
        + alpha
        + delta_in
        - np.sqrt((alpha - 1) ** 2 + 2 * (1 + alpha) * delta_in + delta_in**2)
    )
    q_hat = (
        -1 + alpha - delta_in + np.sqrt(4 * alpha * delta_in + (1 - alpha + delta_in) ** 2)
    ) / (2 * delta_in)
    return q, q, 1 - q, q_hat, q_hat, 1 - q_hat


@njit(error_model="numpy", fastmath=True)
def var_func_BO(m_hat, q_hat, sigma_hat, reg_param):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


@njit(error_model="numpy", fastmath=True)
def var_hat_func_BO_single_noise(m, q, sigma, alpha, delta):
    q_hat = alpha / (1 + delta - q)
    return q_hat, q_hat, 1 - q_hat


# --------------------------------


@njit(error_model="numpy", fastmath=True)
def q_integral_BO_decorrelated_noise(y, xi, q, m, sigma, delta_in, delta_out, percentage, beta):
    return (
        np.exp(-(xi**2) / 2)
        / np.sqrt(2 * np.pi)
        * Z_out_Bayes_decorrelated_noise(
            y, np.sqrt(q) * xi, 1 - q, delta_in, delta_out, percentage, beta
        )
        * (
            f_out_Bayes_decorrelated_noise(
                y, np.sqrt(q) * xi, 1 - q, delta_in, delta_out, percentage, beta
            )
            ** 2
        )
    )


def var_hat_func_BO_num_double_noise(m, q, sigma, alpha, delta_in, delta_out, percentage):
    borders = find_integration_borders_square(
        q_integral_BO_decorrelated_noise,
        np.sqrt((1 + max(delta_in, delta_out))),
        1.0,
        (q, m, sigma, delta_in, delta_out, percentage, 1.0),
    )

    # args = {"m": m, "q": q, "sigma": sigma}
    # domain_xi, domain_y = domains_line_constraint(
    #     borders, border_BO, test_fun_BO, args, args
    # )

    if delta_out <= 0.11 * delta_in:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=10)
    elif delta_out <= 0.5 * delta_in:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=7)
    elif delta_out <= delta_in:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=5)
    else:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=3)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_BO_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, 1.0),
        )[0]

    q_hat = alpha * integral_value
    return q_hat, q_hat, 1 - q_hat


# --------------------------------


def var_hat_func_BO_num_decorrelated_noise(
    m, q, sigma, alpha, delta_in, delta_out, percentage, beta
):
    borders = find_integration_borders_square(
        q_integral_BO_decorrelated_noise,
        np.sqrt((1 + max(delta_in, delta_out))),
        1.0,
        args=(q, m, sigma, delta_in, delta_out, percentage, beta),
    )

    # args = {"m": m, "q": q, "sigma": sigma}
    # domain_xi, domain_y = domains_line_constraint(
    #     borders, border_BO, test_fun_BO, args, args
    # )
    if delta_out <= 0.11 * delta_in:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=10)
    elif delta_out <= 0.5 * delta_in:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=7)
    elif delta_out <= delta_in:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=5)
    else:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=3)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_BO_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_in, delta_out, percentage, beta),
        )[0]

    q_hat = alpha * integral_value
    return q_hat, q_hat, 1 - q_hat
