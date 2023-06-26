from numpy import sqrt
from numba import njit


@njit(error_model="numpy", fastmath=True)
def order_parameters_ridge(alpha, reg_param, delta_in, delta_out, percentage, beta):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    t = sqrt((alpha + reg_param - 1) ** 2 + 4 * reg_param)

    Gamma = 1 + percentage * (beta - 1)
    Lambda = 1 + delta_eff + percentage * (beta**2 - 1)

    sigma = (t - reg_param - alpha + 1) / (2 * reg_param)
    sigma_hat = 2 * reg_param * alpha / (t + reg_param - alpha + 1)

    m = 2 * alpha * Gamma / (t + reg_param + alpha + 1)
    m_hat = 2 * alpha * Gamma * reg_param / (t + reg_param - alpha + 1)

    q = (4 * alpha * (alpha * Gamma**2 * (alpha + reg_param + t - 3) + Lambda * (alpha + reg_param + t + 1))) / (
        (t + reg_param + alpha + 1) * ((alpha + reg_param) ** 2)
        - 2 * alpha
        + 2 * reg_param
        + t**2
        + 2 * t * (alpha + reg_param + 1)
        + 1
    )
    q_hat = (
        4
        * alpha
        * reg_param**2
        * (Lambda * (alpha + reg_param + t + 1) ** 2 - 4 * alpha * Gamma**2 * (reg_param + t + 1))
    ) / (
        (t + reg_param - alpha + 1) ** 2 * ((alpha + reg_param) ** 2)
        - 2 * alpha
        + 2 * reg_param
        + t**2
        + 2 * t * (alpha + reg_param + 1)
        + 1
    )

    return m, q, sigma, m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L2_single_noise(m, q, sigma, alpha, delta):
    m_hat = alpha / (1 + sigma)
    q_hat = alpha * (1 + q + delta - 2 * abs(m)) / ((1 + sigma) ** 2)
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L2_double_noise(m, q, sigma, alpha, delta_in, delta_out, percentage):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    m_hat = alpha / (1 + sigma)
    q_hat = alpha * (1 + q + delta_eff - 2 * abs(m)) / ((1 + sigma) ** 2)
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


# @njit(error_model="numpy", fastmath=True)
def var_hat_func_L2_decorrelated_noise(m, q, sigma, alpha, delta_in, delta_out, percentage, beta):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    intermediate_val = 1 + percentage * (beta - 1)

    m_hat = alpha * intermediate_val / (1 + sigma)
    q_hat = (
        alpha * (1 + q + delta_eff + percentage * (beta**2 - 1) - 2 * abs(m) * intermediate_val) / ((1 + sigma) ** 2)
    )
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_decorrelated_noise_rescaled_data(m, q, sigma, alpha, delta_in, delta_out, percentage, beta):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    intermediate_val = 1 + percentage * (beta - 1)
    normal_const = sqrt(1 + percentage + percentage * beta**2 + delta_eff) / sqrt(1 + delta_eff)

    m_hat = alpha * intermediate_val / (1 + sigma)
    q_hat = (
        alpha
        * (1 + q + delta_eff + percentage * (beta**2 - 1) - 2 * abs(m) * intermediate_val)
        / ((1 + sigma) ** 2)
    )
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat
