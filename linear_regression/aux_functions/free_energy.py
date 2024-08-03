from numpy import pi
from math import sqrt, exp, erf, erfc, log, sinh
from numba import njit


# these functions need to be checked
# @njit
def free_energy(
    Psi_w,
    Psi_out,
    alpha: float,
    m: float,
    q: float,
    V: float,
    m_hat: float,
    q_hat: float,
    V_hat: float,
    Psi_w_args: tuple = (),
    Psi_out_args: tuple = (),
):
    # Q_hat = V_hat - q_hat
    # Q = V + q
    first_term = (
        -0.5 * V * V_hat - 0.5 * (q * V_hat - q_hat * V) + m * m_hat
    )  # m * m_hat - 0.5 * q * q_hat - 0.5 * Q * Q_hat
    # first_term = (
    #     -0.5 * (q * V_hat - q_hat * V) + m * m_hat
    # )  # m * m_hat - 0.5 * q * q_hat - 0.5 * Q * Q_hat
    second_term = -Psi_w(m_hat, q_hat, V_hat, *Psi_w_args)
    third_term = alpha * Psi_out(m, q, V, *Psi_out_args)
    # print("here", first_term, second_term, third_term)
    return first_term + second_term + third_term


@njit
def Psi_w_L2_reg(m_hat: float, q_hat: float, V_hat: float, reg_param: float) -> float:
    reg_param_combination = V_hat + reg_param
    # return 0.5 * (q_hat + m_hat**2) / reg_param_combination
    return 0.5 * ((q_hat + m_hat**2) / reg_param_combination - log(reg_param_combination))


def Psi_w_projection_denoising(Q_hat: float, m_hat: float, q_hat: float, q_fixed: float) -> float:
    # V_hat = Q_hat + q_hat
    return 0.5 * sqrt(q_fixed * q_hat / (q_hat + m_hat**2))


@njit
def Psi_out_L2(
    m: float,
    q: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
) -> float:
    return (
        1
        + q
        + delta_in
        - delta_in * percentage
        - 2 * m * (1 + (-1 + beta) * percentage)
        + percentage * (-1 + beta**2 + delta_out)
    ) / (2.0 * (1 + V))


@njit
def Psi_out_L1(
    m: float,
    q: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
) -> float:
    comb_in = 1 - 2 * m + q + delta_in
    comb_out = q - 2 * m * beta + beta**2 + delta_out
    return (
        (
            sqrt(2 * pi)
            * (2 + V)
            * (
                exp((V**2 * (-(1 / (comb_in)) + 1 / (comb_out))) / 2.0)
                * (-1 + 2 * m - q - delta_in)
                * (-1 + percentage)
                + sqrt(comb_in) * percentage * sqrt(comb_out)
            )
        )
        / (exp(V**2 / (2.0 * (comb_out))) * sqrt(comb_in))
        + pi
        * (
            1
            + q
            + delta_in
            + 2 * m * (-1 + percentage)
            - (1 + q + V + V**2 + delta_in) * percentage
        )
        * erf(V / (sqrt(2) * sqrt(comb_in)))
        + pi
        * percentage
        * (q + V + V**2 - 2 * m * beta + beta**2 + delta_out)
        * erf(V / (sqrt(2) * sqrt(comb_out)))
        - pi * V * (1 + V) * erfc(V / (sqrt(2) * sqrt(comb_in)))
    ) / (2.0 * pi * (1 + V))


@njit
def Psi_out_Huber(
    m: float,
    q: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    a: float,
) -> float:
    # V = Q - q
    comb_in = 1 - 2 * m + q + delta_in
    comb_out = q - 2 * m * beta + beta**2 + delta_out
    return (
        -(
            (
                a
                * sqrt(2 / pi)
                * (1 + V)
                * (
                    exp((a**2 * (1 + V) ** 2 * (-(1 / (comb_in)) + 1 / (comb_out))) / 2.0)
                    * sqrt(comb_in)
                    * (-1 + percentage)
                    - percentage * sqrt(comb_out)
                )
            )
            / exp((a**2 * (1 + V) ** 2) / (2.0 * (comb_out)))
        )
        + (
            1
            + q
            + delta_in
            + 2 * m * (-1 + percentage)
            - (1 + q + a**2 * (1 + V) ** 2 + delta_in) * percentage
        )
        * erf((a * (1 + V)) / (sqrt(2) * sqrt(comb_in)))
        + percentage
        * (q + a**2 * (1 + V) ** 2 - 2 * m * beta + beta**2 + delta_out)
        * erf((a * (1 + V)) / (sqrt(2) * sqrt(comb_out)))
        - a**2 * (1 + V) ** 2 * erfc((a * (1 + V)) / (sqrt(2) * sqrt(comb_in)))
    ) / (2.0 * (1 + V))
