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
    Dlambdaq_moreau_loss_sum_absolute,
)

BIG_NUMBER = 55


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
        gauss_Z_w_Bayes_gaussian_prior(ξ, m_hat, q_hat, 0, 1)
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
    proximal = proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, reg_param, reg_order, P_hat, pstar)
    return gauss_Z_w_Bayes_gaussian_prior(ξ, m_hat, q_hat, 0, 1) * (proximal**2)


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
    DƔ_prox = DƔ_proximal_sum_absolute(sqrt(q_hat) * ξ, Σ_hat, reg_param, reg_order, P_hat, pstar)
    return gauss_Z_w_Bayes_gaussian_prior(ξ, m_hat, q_hat, 0, 1) * DƔ_prox


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
    DPhat_moreau = Dlambdaq_moreau_loss_sum_absolute(
        sqrt(q_hat) * ξ, Σ_hat, reg_param, reg_order, P_hat, pstar
    )
    return gauss_Z_w_Bayes_gaussian_prior(ξ, m_hat, q_hat, 0, 1) * DPhat_moreau


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
