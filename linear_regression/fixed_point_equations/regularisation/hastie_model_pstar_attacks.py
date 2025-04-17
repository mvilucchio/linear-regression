from numba import njit
from math import sqrt
from scipy.integrate import quad
from ...aux_functions.misc import gaussian
from ...aux_functions.prior_regularization_funcs import (
    Z_w_Bayes_gaussian_prior,
    DZ_w_Bayes_gaussian_prior,
)
from ...aux_functions.moreau_proximals import (
    proximal_sum_absolute,
    DƔ_proximal_sum_absolute,
)

BIG_NUMBER = 15


@njit(error_model="numpy", fastmath=False)
def m_integral_hastie_L2_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    pstar: float,
    gamma: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + 1 / gamma
        return (
            sqrt(1 + gamma)
            * gaussian(ξ, 0, 1)
            * DZ_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_sum_absolute(
                sqrt(q_hat * gamma_tilde) * ξ,
                V_hat * gamma_tilde,
                reg_param,
                2.0,
                0.5 * P_hat,
                pstar,
            )
        )
    else:
        return (
            sqrt(2 / gamma)
            * gaussian(ξ, 0, 1)
            * DZ_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
            * proximal_sum_absolute(sqrt(2 * q_hat) * ξ, 2 * V_hat, reg_param, 2.0, P_hat, pstar)
        )


@njit(error_model="numpy", fastmath=False)
def q_integral_hastie_L2_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    pstar: float,
    gamma: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + (1 / gamma)
        first_term = (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * (
                proximal_sum_absolute(
                    sqrt(q_hat * gamma_tilde) * ξ,
                    V_hat * gamma_tilde,
                    reg_param,
                    2.0,
                    0.5 * P_hat,
                    pstar,
                )
                ** 2
            )
        )
        second_term = gaussian(ξ, 0, 1) * (
            proximal_sum_absolute(
                sqrt(q_hat * gamma_tilde) * ξ,
                V_hat * gamma_tilde,
                reg_param,
                2.0,
                0.5 * P_hat,
                pstar,
            )
            ** 2
        )
        return (1 + gamma) * first_term + (1 - gamma**2) / gamma * second_term
    else:
        return 2 * (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
            * (
                proximal_sum_absolute(sqrt(2 * q_hat) * ξ, 2 * V_hat, reg_param, 2.0, P_hat, pstar)
                ** 2
            )
        )


@njit(error_model="numpy", fastmath=False)
def V_integral_hastie_L2_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    pstar: float,
    gamma: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + 1 / gamma
        first_term = (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * DƔ_proximal_sum_absolute(
                sqrt(q_hat * gamma_tilde) * ξ,
                V_hat * gamma_tilde,
                reg_param,
                2.0,
                0.5 * P_hat,
                pstar,
            )
        )
        second_term = gaussian(ξ, 0, 1) * DƔ_proximal_sum_absolute(
            sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, reg_param, 2.0, 0.5 * P_hat, pstar
        )
        return (1 + gamma) * first_term + (1 - gamma**2) / gamma * second_term
    else:
        return 2 * (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
            * DƔ_proximal_sum_absolute(sqrt(2 * q_hat) * ξ, 2 * V_hat, reg_param, 2.0, P_hat, pstar)
        )


@njit(error_model="numpy", fastmath=False)
def P_integral_hastie_L2_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    pstar: float,
    gamma: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + 1 / gamma
        first_term = (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * abs(
                proximal_sum_absolute(
                    sqrt(q_hat * gamma_tilde) * ξ,
                    V_hat * gamma_tilde,
                    reg_param,
                    2.0,
                    0.5 * P_hat,
                    pstar,
                )
            )
            ** pstar
        )
        second_term = (
            gaussian(ξ, 0, 1)
            * abs(
                proximal_sum_absolute(
                    sqrt(q_hat * gamma_tilde) * ξ,
                    V_hat * gamma_tilde,
                    reg_param,
                    2.0,
                    0.5 * P_hat,
                    pstar,
                )
            )
            ** pstar
        )
        return gamma * first_term + (1 - gamma) * second_term
    else:
        return (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
            * abs(
                proximal_sum_absolute(sqrt(2 * q_hat) * ξ, 2 * V_hat, reg_param, 2.0, P_hat, pstar)
            )
            ** pstar
        )


# -----------------------------------
def f_hastie_L2_reg_Lp_attack(
    m_hat: float,
    q_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    pstar: float,
    gamma: float,
):
    η_hat = m_hat**2 / q_hat

    domains = [
        (
            -BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
            BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
        )
    ]

    int_value_m = 0.0
    for domain in domains:
        int_value_m += quad(
            m_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, pstar, gamma),
        )[0]
    m = int_value_m

    int_value_q = 0.0
    for domain in domains:
        int_value_q += quad(
            q_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, pstar, gamma),
        )[0]
    q = int_value_q

    int_value_V = 0.0
    for domain in domains:
        int_value_V += quad(
            V_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, pstar, gamma),
        )[0]
    V = int_value_V

    int_value_P = 0.0
    for domain in domains:
        int_value_P += quad(
            P_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, pstar, gamma),
        )[0]
    P = int_value_P

    return m, q, V, P
