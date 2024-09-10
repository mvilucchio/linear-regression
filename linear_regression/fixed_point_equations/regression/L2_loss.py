from numpy import sqrt
from numba import njit


@njit(error_model="numpy", fastmath=True)
def order_parameters_ridge(alpha, reg_param, delta_in, delta_out, percentage, beta):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    t = sqrt((alpha + reg_param - 1) ** 2 + 4 * reg_param)

    Gamma = 1 + percentage * (beta - 1)
    Lambda = 1 + delta_eff + percentage * (beta**2 - 1)

    V = (t - reg_param - alpha + 1) / (2 * reg_param)
    V_hat = 2 * reg_param * alpha / (t + reg_param - alpha + 1)

    m = 2 * alpha * Gamma / (t + reg_param + alpha + 1)
    m_hat = 2 * alpha * Gamma * reg_param / (t + reg_param - alpha + 1)

    q = (
        4
        * alpha
        * (alpha * Gamma**2 * (alpha + reg_param + t - 3) + Lambda * (alpha + reg_param + t + 1))
    ) / (
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

    return m, q, V, m_hat, q_hat, V_hat


@njit(error_model="numpy", fastmath=True)
def f_hat_L2_single_noise(m, q, V, alpha, delta):
    m_hat = alpha / (1 + V)
    q_hat = alpha * (1 + q + delta - 2 * m) / ((1 + V) ** 2)
    V_hat = alpha / (1 + V)
    return m_hat, q_hat, V_hat


@njit(error_model="numpy", fastmath=True)
def f_hat_L2_single_noise_new(m, q, V, alpha, delta, rho):
    m_hat = alpha / (1 + V)
    q_hat = alpha * (rho + q + delta - 2 * abs(m)) / ((1 + V) ** 2)
    V_hat = alpha / (1 + V)
    return m_hat, q_hat, V_hat


@njit(error_model="numpy", fastmath=True)
def f_hat_L2_double_noise(m, q, V, alpha, delta_in, delta_out, percentage):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    m_hat = alpha / (1 + V)
    q_hat = alpha * (1 + q + delta_eff - 2 * abs(m)) / ((1 + V) ** 2)
    V_hat = alpha / (1 + V)
    return m_hat, q_hat, V_hat


# @njit(error_model="numpy", fastmath=True)
def f_hat_L2_decorrelated_noise(m, q, V, alpha, delta_in, delta_out, percentage, beta):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    intermediate_val = 1 + percentage * (beta - 1)

    m_hat = alpha * intermediate_val / (1 + V)
    q_hat = (
        alpha
        * (1 + q + delta_eff + percentage * (beta**2 - 1) - 2 * abs(m) * intermediate_val)
        / ((1 + V) ** 2)
    )
    V_hat = alpha / (1 + V)
    return m_hat, q_hat, V_hat


def f_hat_L2_decorrelated_noise_rescaled_data(
    m, q, V, alpha, delta_in, delta_out, percentage, beta
):
    delta_eff = (1 - percentage) * delta_in + percentage * delta_out
    intermediate_val = 1 + percentage * (beta - 1)
    normal_const = sqrt(1 + percentage + percentage * beta**2 + delta_eff) / sqrt(1 + delta_eff)

    m_hat = alpha * intermediate_val / (1 + V)
    q_hat = (
        alpha
        * (1 + q + delta_eff + percentage * (beta**2 - 1) - 2 * abs(m) * intermediate_val)
        / ((1 + V) ** 2)
    )
    V_hat = alpha / (1 + V)
    return m_hat, q_hat, V_hat
