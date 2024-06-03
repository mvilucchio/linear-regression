from numba import njit
from math import exp, erf, erfc, sqrt, pi
import numpy as np
from numpy import sign as np_sign
from scipy.integrate import quad, dblquad
from ...aux_functions.misc import gaussian
from scipy.optimize import minimize_scalar
from ...aux_functions.prior_regularization_funcs import (
    Z_w_Bayes_gaussian_prior,
    f_w_Bayes_gaussian_prior,
    DZ_w_Bayes_gaussian_prior,
)
from ...aux_functions.moreau_proximals import (
    proximal_sum_absolute,
    DƔ_proximal_sum_absolute,
    Dlambdaq_moreau_loss_sum_absolute,
    proximal_L1,
    DƔ_proximal_L1,
    Dphat_proximal_L1,
)

BIG_NUMBER = 35


@njit(error_model="numpy", fastmath=False)
def m_integral_gaussian_weights_Lr_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_order: float,
    reg_param: float,
    pstar: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    proximal = proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, reg_param, reg_order, P_hat, pstar)
    return (
        gaussian(ξ, 0, 1)
        * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
        * f_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
        * proximal
    )


@njit(error_model="numpy", fastmath=False)
def q_integral_gaussian_weights_Lr_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_order: float,
    reg_param: float,
    pstar: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    proximal = proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, reg_param, reg_order, P_hat, pstar)
    return (
        gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * (proximal**2)
    )


@njit(error_model="numpy", fastmath=False)
def Σ_integral_gaussian_weights_Lr_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_order: float,
    reg_param: float,
    pstar: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    DƔ_prox = DƔ_proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, reg_param, reg_order, P_hat, pstar)
    return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DƔ_prox


@njit(error_model="numpy", fastmath=False)
def P_integral_gaussian_weights_Lr_reg_Lp_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_order: float,
    reg_param: float,
    pstar: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    DPhat_moreau = Dlambdaq_moreau_loss_sum_absolute(
        sqrt(q_hat) * ξ, Σ_hat, reg_param, reg_order, P_hat, pstar
    )
    return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DPhat_moreau


# -----------------------------------
def f_Lr_regularisation_Lpstar_attack(
    m_hat: float,
    q_hat: float,
    Σ_hat: float,
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
            m_integral_gaussian_weights_Lr_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, Σ_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    m = int_value_m

    int_value_q = 0.0
    for domain in domains:
        int_value_q += quad(
            q_integral_gaussian_weights_Lr_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, Σ_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    q = int_value_q

    int_value_Σ = 0.0
    for domain in domains:
        int_value_Σ += quad(
            Σ_integral_gaussian_weights_Lr_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, Σ_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    Σ = int_value_Σ

    int_value_P = 0.0
    for domain in domains:
        int_value_P += quad(
            P_integral_gaussian_weights_Lr_reg_Lp_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, Σ_hat, P_hat, reg_order, reg_param, pstar),
        )[0]
    P = int_value_P

    return m, q, Σ, P


# -------------- specification to when reg_order and pstar are 1 ------------- #
@njit(error_model="numpy", fastmath=False)
def m_integral_gaussian_weights_L1_attack_and_reg(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_param: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    proximal = proximal_L1(sqrt(q_hat) * ξ, Σ_hat, P_hat + reg_param)
    return (
        gaussian(ξ, 0, 1)
        * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
        * f_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
        * proximal
    )


@njit(error_model="numpy", fastmath=False)
def q_integral_gaussian_weights_L1_attack_and_reg(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_param: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    proximal = proximal_L1(sqrt(q_hat) * ξ, Σ_hat, P_hat + reg_param)
    return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * proximal**2


@njit(error_model="numpy", fastmath=False)
def Σ_integral_gaussian_weights_L1_attack_and_reg(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_param: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    DƔ_prox = DƔ_proximal_L1(sqrt(q_hat) * ξ, Σ_hat, P_hat + reg_param)
    return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DƔ_prox


@njit(error_model="numpy", fastmath=False)
def P_integral_gaussian_weights_L1_attack_and_reg(
    ξ: float,
    q_hat: float,
    m_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_param: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    DPhat_moreau = Dphat_proximal_L1(sqrt(q_hat) * ξ, Σ_hat, reg_param, P_hat)
    return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DPhat_moreau


# -----------------------------------
def f_Lr_regularisation_attack_1(
    m_hat: float,
    q_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_param: float,
):
    domains = [
        (-BIG_NUMBER, -(reg_param + P_hat) / sqrt(q_hat)),
        ((reg_param + P_hat) / sqrt(q_hat), BIG_NUMBER),
    ]

    # int_value_m = 0.0
    # for domain in domains:
    #     int_value_m += quad(
    #         m_integral_gaussian_weights_L1_attack_and_reg,
    #         domain[0],
    #         domain[1],
    #         args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
    #     )[0]
    # m = int_value_m
    m = (
        abs(m_hat)
        * (
            (sqrt(2 / pi) * (P_hat + reg_param))
            / (exp((P_hat + reg_param) ** 2 / (2.0 * (m_hat**2 + q_hat))) * sqrt(m_hat**2 + q_hat))
            + erfc((P_hat + reg_param) / (sqrt(2) * sqrt(m_hat**2 + q_hat)))
        )
    ) / Σ_hat

    # int_value_q = 0.0
    # for domain in domains:
    #     int_value_q += quad(
    #         q_integral_gaussian_weights_L1_attack_and_reg,
    #         domain[0],
    #         domain[1],
    #         args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
    #     )[0]
    # q = int_value_q
    q = (
        m_hat**2
        + q_hat
        + (sqrt(2 / pi) * sqrt(m_hat**2 + q_hat) * (P_hat + reg_param))
        / exp((P_hat + reg_param) ** 2 / (2.0 * (m_hat**2 + q_hat)))
        + (P_hat + reg_param) ** 2
        - (m_hat**2 + q_hat + (P_hat + reg_param) ** 2)
        * erf((P_hat + reg_param) / (sqrt(2) * sqrt(m_hat**2 + q_hat)))
    ) / Σ_hat**2

    # int_value_Σ = 0.0
    # for domain in domains:
    #     int_value_Σ += quad(
    #         Σ_integral_gaussian_weights_L1_attack_and_reg,
    #         domain[0],
    #         domain[1],
    #         args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
    #     )[0]
    # Σ = int_value_Σ
    Σ = erfc((P_hat + reg_param) / (sqrt(2) * sqrt(m_hat**2 + q_hat))) / Σ_hat

    # int_value_P = 0.0
    # for domain in domains:
    #     int_value_P += quad(
    #         P_integral_gaussian_weights_L1_attack_and_reg,
    #         domain[0],
    #         domain[1],
    #         args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
    #     )[0]
    # P = int_value_P
    P = (
        (sqrt(2 / pi) * sqrt(m_hat**2 + q_hat))
        / exp((P_hat + reg_param) ** 2 / (2.0 * (m_hat**2 + q_hat)))
        - (P_hat + reg_param) * erfc((P_hat + reg_param) / (sqrt(2) * sqrt(m_hat**2 + q_hat)))
    ) / Σ_hat

    return m, q, Σ, P


# # -------------------------------- other_stuff ------------------------------- #
# @njit(error_model="numpy", fastmath=False)
# def proximal_Lasso(gamma, Lambda, P_hat, reg_param):
#     return np_sign(gamma) * max(0.0, abs(gamma) - (reg_param + P_hat)) / Lambda


# # @njit(error_model="numpy", fastmath=False)
# def DƔ_proximal_Lasso(gamma, Lambda, P_hat, reg_param):
#     if abs(gamma) / Lambda != 0.0:
#         return np.heaviside(abs(gamma / Lambda) - (reg_param + P_hat) / Lambda, 0.0) / Lambda
#     else:
#         return (
#             np.heaviside(abs(gamma / Lambda) - (reg_param + P_hat) / Lambda, 0.0) / Lambda
#         ) + max(0.0, abs(gamma) - (reg_param + P_hat)) / Lambda


# def moreau_Lasso(gamma, Lambda, P_hat, reg_param):
#     if abs(gamma) / Lambda <= (reg_param + P_hat) / Lambda:
#         return 0.5 * gamma**2 / (Lambda * (reg_param + P_hat))
#     else:
#         return abs(gamma / Lambda) - 0.5 * (P_hat + reg_param) / Lambda


# def DP_hat_moreau_Lasso(gamma, Lambda, P_hat, reg_param):
#     if abs(gamma) / Lambda <= (reg_param + P_hat) / Lambda:
#         return -0.5 * gamma**2 / (Lambda * (reg_param + P_hat) ** 2)
#     else:
#         return -0.5 / Lambda


# # -----------------------------


# def m_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO(ξ, q_hat, m_hat, Σ_hat, P_hat, reg_param):
#     η_hat = m_hat**2 / q_hat
#     prox = proximal_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     return (
#         gaussian(ξ, 0, 1)
#         * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
#         * f_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
#         * prox
#     )


# def q_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO(ξ, q_hat, m_hat, Σ_hat, P_hat, reg_param):
#     η_hat = m_hat**2 / q_hat
#     prox = proximal_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * prox**2


# def Σ_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO(ξ, q_hat, m_hat, Σ_hat, P_hat, reg_param):
#     η_hat = m_hat**2 / q_hat
#     DƔ_prox = DƔ_proximal_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DƔ_prox


# def Dz_Lr_reg_Linf_attack_LASSO(z, P_hat, reg_param):
#     return (P_hat + reg_param) * np_sign(z)


# def DPhat_Lr_reg_Linf_attack_LASSO(z, P_hat, reg_param):
#     return abs(z)


# def P_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO(ξ, q_hat, m_hat, Σ_hat, P_hat, reg_param):
#     η_hat = m_hat**2 / q_hat
#     prox = proximal_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     # DP_hat_prox = (
#     #     (-1 if np.abs(sqrt(q_hat) * ξ) > P_hat + reg_param else 0)
#     #     * np_sign(sqrt(q_hat) * ξ)
#     #     / (Σ_hat)
#     # )
#     # DP_hat_prox = (
#     #     -np_sign(sqrt(q_hat) * ξ / Σ_hat)
#     #     * np.heaviside(np.abs(sqrt(q_hat) * ξ / Σ_hat) - (P_hat + reg_param) / Σ_hat, 0)
#     #     / Σ_hat
#     # )
#     # DP_hat_prox = DƔ_proximal_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     DP_hat_prox = (
#         -np.sign(prox) / (Σ_hat) if prox != 0 else -np.sign(prox) / (reg_param + P_hat + Σ_hat)
#     )
#     DPhat_moreau = (
#         (prox * Σ_hat - sqrt(q_hat) * ξ) * DP_hat_prox
#         + Dz_Lr_reg_Linf_attack_LASSO(prox, P_hat, reg_param) * DP_hat_prox
#         + DPhat_Lr_reg_Linf_attack_LASSO(prox, P_hat, reg_param)
#     )
#     # DPhat_moreau = DP_hat_moreau_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DPhat_moreau


# def _just_derivative_lasso(ξ, q_hat, m_hat, Σ_hat, P_hat, reg_param):
#     η_hat = m_hat**2 / q_hat
#     # prox = proximal_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     # DP_hat_prox = (
#     #     (-1 if np.abs(sqrt(q_hat) * ξ) > P_hat + reg_param else 0)
#     #     * np_sign(sqrt(q_hat) * ξ)
#     #     / (Σ_hat)
#     # )
#     # DP_hat_prox = (
#     #     -np_sign(sqrt(q_hat) * ξ / Σ_hat)
#     #     * np.heaviside(np.abs(sqrt(q_hat) * ξ / Σ_hat) - (P_hat + reg_param) / Σ_hat, 0)
#     #     / Σ_hat
#     # )
#     # DPhat_moreau = (
#     #     (prox * Σ_hat - sqrt(q_hat) * ξ) * DP_hat_prox
#     #     + Dz_Lr_reg_Linf_attack_LASSO(prox, P_hat, reg_param) * DP_hat_prox
#     #     + DPhat_Lr_reg_Linf_attack_LASSO(prox, P_hat, reg_param)
#     # )
#     prox = proximal_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     # DP_hat_prox = (
#     #     (-1 if np.abs(sqrt(q_hat) * ξ) > P_hat + reg_param else 0)
#     #     * np_sign(sqrt(q_hat) * ξ)
#     #     / (Σ_hat)
#     # )
#     # DP_hat_prox = (
#     #     -np_sign(sqrt(q_hat) * ξ / Σ_hat)
#     #     * np.heaviside(np.abs(sqrt(q_hat) * ξ / Σ_hat) - (P_hat + reg_param) / Σ_hat, 0)
#     #     / Σ_hat
#     # )
#     DP_hat_prox = (
#         -np.sign(prox) / (Σ_hat) if prox != 0 else -np.sign(prox) / (reg_param + P_hat + Σ_hat)
#     )
#     DPhat_moreau = (
#         (prox * Σ_hat - sqrt(q_hat) * ξ) * DP_hat_prox
#         + Dz_Lr_reg_Linf_attack_LASSO(prox, P_hat, reg_param) * DP_hat_prox
#         + DPhat_Lr_reg_Linf_attack_LASSO(prox, P_hat, reg_param)
#     )
#     # DPhat_moreau = DP_hat_moreau_Lasso(sqrt(q_hat) * ξ, Σ_hat, P_hat, reg_param)
#     return DPhat_moreau


# def f_Lr_reg_Linf_attack_Lasso(
#     m_hat: float,
#     q_hat: float,
#     Σ_hat: float,
#     P_hat: float,
#     reg_param: float,
# ):
#     domains = [
#         (-BIG_NUMBER, BIG_NUMBER),
#     ]

#     int_value_m = 0.0
#     for domain in domains:
#         int_value_m += quad(
#             m_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
#         )[0]
#     m = int_value_m

#     int_value_q = 0.0
#     for domain in domains:
#         int_value_q += quad(
#             q_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
#         )[0]
#     q = int_value_q

#     int_value_Σ = 0.0
#     for domain in domains:
#         int_value_Σ += quad(
#             Σ_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
#         )[0]
#     Σ = int_value_Σ

#     int_value_P = 0.0
#     for domain in domains:
#         int_value_P += quad(
#             P_integral_gaussian_weights_Lr_reg_Lp_attack_LASSO,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, reg_param),
#         )[0]
#     P = int_value_P

#     return m, q, Σ, P


# # -----------------------------------


# @njit(error_model="numpy", fastmath=False)
# def moreau_loss_adv(z, gamma, Lambda, P_hat, r, reg_param):
#     return 0.5 * (z - gamma / Lambda) ** 2 * Lambda + reg_param * abs(z) ** r + P_hat * abs(z)


# # supposing derivative of sign function is 0
# @njit(error_model="numpy", fastmath=False)
# def DDz_Lr_reg_Linf_attack(z, P_hat, r, reg_param):
#     if r > 1:
#         if np.isclose(z, 0.0, atol=1e-11):
#             additional_term = P_hat
#         else:
#             additional_term = 0.0
#         return reg_param * r * (r - 1) * abs(z) ** (r - 2) + additional_term
#     else:
#         if np.isclose(z, 0.0, atol=1e-11):
#             return reg_param + P_hat
#         return 0.0


# @njit(error_model="numpy", fastmath=False)
# def DzDPhat_Lr_reg_Linf_attack(z, P_hat, r, reg_param):
#     return np_sign(z)


# @njit(error_model="numpy", fastmath=False)
# def Dz_Lr_reg_Linf_attack(z, P_hat, r, reg_param):
#     if r > 1:
#         return np_sign(z) * (reg_param * r * abs(z) ** (r - 1) + P_hat)
#     else:
#         return np_sign(z) * (reg_param + P_hat)


# @njit(error_model="numpy", fastmath=False)
# def DPhat_Lr_reg_Linf_attack(z, P_hat, r, reg_param):
#     return abs(z)


# # -----------------------------


# def m_integral_gaussian_weights_Lr_reg_Lp_attack(ξ, q_hat, m_hat, Σ_hat, P_hat, r, reg_param):
#     η_hat = m_hat**2 / q_hat
#     proximal = minimize_scalar(
#         moreau_loss_adv, args=(sqrt(q_hat) * ξ, Σ_hat, P_hat, r, reg_param), tol=1e-11
#     )["x"]
#     # proximal = proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, P_hat, 1.0, reg_param, r)
#     return (
#         gaussian(ξ, 0, 1)
#         * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
#         * f_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1)
#         * proximal
#     )


# def q_integral_gaussian_weights_Lr_reg_Lp_attack(ξ, q_hat, m_hat, Σ_hat, P_hat, r, reg_param):
#     η_hat = m_hat**2 / q_hat
#     proximal = minimize_scalar(
#         moreau_loss_adv, args=(sqrt(q_hat) * ξ, Σ_hat, P_hat, r, reg_param), tol=1e-11
#     )["x"]
#     # proximal = proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, P_hat, 1.0, reg_param, r)
#     return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * proximal**2


# def Σ_integral_gaussian_weights_Lr_reg_Lp_attack(ξ, q_hat, m_hat, Σ_hat, P_hat, r, reg_param):
#     η_hat = m_hat**2 / q_hat
#     prox = minimize_scalar(
#         moreau_loss_adv, args=(sqrt(q_hat) * ξ, Σ_hat, P_hat, r, reg_param), tol=1e-11
#     )["x"]
#     DƔ_prox = (Σ_hat + DDz_Lr_reg_Linf_attack(prox, P_hat, r, reg_param)) ** (-1)
#     # DƔ_prox = DƔ_proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, P_hat, 1.0, reg_param, r)
#     return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DƔ_prox


# def P_integral_gaussian_weights_Lr_reg_Lp_attack(ξ, q_hat, m_hat, Σ_hat, P_hat, r, reg_param):
#     η_hat = m_hat**2 / q_hat
#     prox = minimize_scalar(
#         moreau_loss_adv, args=(sqrt(q_hat) * ξ, Σ_hat, P_hat, r, reg_param), tol=1e-11
#     )["x"]
#     DP_hat_prox = -DzDPhat_Lr_reg_Linf_attack(prox, P_hat, r, reg_param) / (
#         DDz_Lr_reg_Linf_attack(prox, P_hat, r, reg_param) + Σ_hat
#     )
#     DPhat_moreau = (
#         (prox * Σ_hat - sqrt(q_hat) * ξ) * DP_hat_prox
#         + Dz_Lr_reg_Linf_attack(prox, P_hat, r, reg_param) * DP_hat_prox
#         + DPhat_Lr_reg_Linf_attack(prox, P_hat, r, reg_param)
#     )
#     return gaussian(ξ, 0, 1) * Z_w_Bayes_gaussian_prior(sqrt(η_hat) * ξ, η_hat, 0, 1) * DPhat_moreau


# def _just_derivative(ξ, q_hat, m_hat, Σ_hat, P_hat, r, reg_param):
#     η_hat = m_hat**2 / q_hat
#     prox = minimize_scalar(
#         moreau_loss_adv, args=(sqrt(q_hat) * ξ, Σ_hat, P_hat, r, reg_param), tol=1e-11
#     )["x"]
#     DP_hat_prox = -DzDPhat_Lr_reg_Linf_attack(prox, P_hat, r, reg_param) / (
#         DDz_Lr_reg_Linf_attack(prox, P_hat, r, reg_param) + Σ_hat
#     )
#     DPhat_moreau = (
#         (prox * Σ_hat - sqrt(q_hat) * ξ) * DP_hat_prox
#         + Dz_Lr_reg_Linf_attack(prox, P_hat, r, reg_param) * DP_hat_prox
#         + DPhat_Lr_reg_Linf_attack(prox, P_hat, r, reg_param)
#     )
#     return DPhat_moreau


# # -----------------------------------
# def f_Lr_reg_Linf_attack_generic(
#     m_hat: float,
#     q_hat: float,
#     Σ_hat: float,
#     P_hat: float,
#     r: float,
#     reg_param: float,
# ):
#     domains = [(-BIG_NUMBER, BIG_NUMBER)]

#     int_value_m = 0.0
#     for domain in domains:
#         int_value_m += quad(
#             m_integral_gaussian_weights_Lr_reg_Lp_attack,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, r, reg_param),
#         )[0]
#     m = int_value_m

#     int_value_q = 0.0
#     for domain in domains:
#         int_value_q += quad(
#             q_integral_gaussian_weights_Lr_reg_Lp_attack,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, r, reg_param),
#         )[0]
#     q = int_value_q

#     int_value_Σ = 0.0
#     for domain in domains:
#         int_value_Σ += quad(
#             Σ_integral_gaussian_weights_Lr_reg_Lp_attack,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, r, reg_param),
#         )[0]
#     Σ = int_value_Σ

#     int_value_P = 0.0
#     for domain in domains:
#         int_value_P += quad(
#             P_integral_gaussian_weights_Lr_reg_Lp_attack,
#             domain[0],
#             domain[1],
#             args=(q_hat, m_hat, Σ_hat, P_hat, r, reg_param),
#         )[0]
#     P = int_value_P

#     return m, q, Σ, P


# def f_Lr_reg_Linf_attack(
#     m_hat: float,
#     q_hat: float,
#     Σ_hat: float,
#     P_hat: float,
#     r: float,
#     reg_param: float,
# ):
#     if r == 1:
#         # print("LASSO")
#         return f_Lr_reg_Linf_attack_Lasso(m_hat, q_hat, Σ_hat, P_hat, reg_param)
#     else:
#         return f_Lr_reg_Linf_attack_generic(m_hat, q_hat, Σ_hat, P_hat, r, reg_param)
