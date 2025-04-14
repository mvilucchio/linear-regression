from numba import njit
from math import sqrt
from scipy.integrate import quad
from ...aux_functions.misc import gaussian
from ...aux_functions.prior_regularization_funcs import (
    Z_w_Bayes_gaussian_prior,
    f_w_Bayes_gaussian_prior,
    DZ_w_Bayes_gaussian_prior,
    gauss_Z_w_Bayes_gaussian_prior,
)
from ...aux_functions.moreau_proximals import (
    proximal_sum_absolute,
    DƔ_proximal_sum_absolute,
)

BIG_NUMBER = 55


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
            sqrt(gamma / (1 + gamma))
            * gaussian(ξ, 0, 1)
            * DZ_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_sum_absolute(
                sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, reg_param, 2.0, P_hat, pstar
            )
        )
    else:
        raise sqrt(2) * gaussian(ξ, 0, 1) * DZ_w_Bayes_gaussian_prior(
            sqrt(η_hat) * ξ, η_hat, 0, 1
        ) * proximal_sum_absolute(sqrt(2 * q_hat) * ξ, 2 * V_hat, reg_param, 2.0, P_hat, pstar)


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
    if gamma <= 1:
        raise NotImplementedError
    else:
        raise NotImplementedError

    proximal = proximal_sum_absolute(sqrt(q_hat) * ξ, V_hat, reg_param, reg_order, P_hat, pstar)
    return gauss_Z_w_Bayes_gaussian_prior(ξ, m_hat, q_hat, 0, 1) * (proximal**2)


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
                sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, reg_param, 2.0, P_hat, pstar
            )
        )
        second_term = gaussian(ξ, 0, 1) * DƔ_proximal_sum_absolute(
            sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, reg_param, 2.0, P_hat, pstar
        )
        return gamma * first_term + (1 - gamma) * second_term
    else:
        raise gaussian(ξ, 0, 1) * DZ_w_Bayes_gaussian_prior(
            sqrt(η_hat) * ξ, η_hat, 0, 1
        ) * DƔ_proximal_sum_absolute(sqrt(2 * q_hat) * ξ, 2 * V_hat, reg_param, 2.0, P_hat, pstar)


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
                    sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, reg_param, 2.0, P_hat, pstar
                )
            )
            ** pstar
        )
        second_term = (
            gaussian(ξ, 0, 1)
            * abs(
                proximal_sum_absolute(
                    sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, reg_param, 2.0, P_hat, pstar
                )
            )
            ** pstar
        )
        return gamma * first_term + (1 - gamma) * second_term
    else:
        raise gaussian(ξ, 0, 1) * DZ_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * abs(
            proximal_sum_absolute(sqrt(2 * q_hat) * ξ, 2 * V_hat, reg_param, 2.0, P_hat, pstar)
        ) ** pstar


# -----------------------------------
def f_hastie_L2_reg_Lp_attack(
    m_hat: float,
    q_hat: float,
    V_hat: float,
    P_hat: float,
    reg_order: float,
    reg_param: float,
    pstar: float,
):
    if reg_order == 1 and pstar == 1:
        domains = [
            (-BIG_NUMBER, -(reg_param + P_hat) / sqrt(q_hat)),
            ((reg_param + P_hat) / sqrt(q_hat), BIG_NUMBER),
        ]
    elif reg_order == 1:
        domains = [(-BIG_NUMBER, -reg_param / sqrt(q_hat)), (reg_param / sqrt(q_hat), BIG_NUMBER)]
    elif pstar == 1:
        domains = [(-BIG_NUMBER, -P_hat / sqrt(q_hat)), (P_hat / sqrt(q_hat), BIG_NUMBER)]
    else:
        domains = [(-BIG_NUMBER, BIG_NUMBER)]

    int_value_m = 0.0
    for domain in domains:
        int_value_m += quad(
            m_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    m = int_value_m

    int_value_q = 0.0
    for domain in domains:
        int_value_q += quad(
            q_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    q = int_value_q

    int_value_V = 0.0
    for domain in domains:
        int_value_V += quad(
            V_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    V = int_value_V

    int_value_P = 0.0
    for domain in domains:
        int_value_P += quad(
            P_integral_hastie_L2_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    P = int_value_P

    return m, q, V, P
