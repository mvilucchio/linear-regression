from numba import njit
from math import sqrt
from scipy.integrate import quad
from ...aux_functions.misc import gaussian
from ...aux_functions.prior_regularization_funcs import (
    Z_w_Bayes_gaussian_prior,
    DZ_w_Bayes_gaussian_prior,
)
from ...aux_functions.moreau_proximals import (
    proximal_Elastic_net,
    DƔ_proximal_Elastic_net,
)

BIG_NUMBER = 15


@njit(error_model="numpy", fastmath=False)
def m_integral_hastie_L2_reg_Linf_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + (1 / gamma)
        return (
            sqrt(gamma)
            * gaussian(ξ, 0, 1)
            * DZ_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_Elastic_net(
                sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, 0.5 * P_hat, reg_param
            )
        )
    else:
        η_hat_red = η_hat / 2
        return (
            gaussian(ξ, 0, 1)
            * DZ_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_Elastic_net(sqrt(2 * q_hat) * ξ, 2 * V_hat, 0.5 * P_hat, reg_param)
        )


@njit(error_model="numpy", fastmath=False)
def q_integral_hastie_L2_reg_Linf_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
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
                proximal_Elastic_net(
                    sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, 0.5 * P_hat, reg_param
                )
                ** 2
            )
        )
        second_term = gaussian(ξ, 0, 1) * (
            proximal_Elastic_net(sqrt(q_hat) * ξ, V_hat, 0.5 * P_hat, reg_param) ** 2
        )
        return 0.5 * ((1 + gamma) * first_term + (1 - gamma) * second_term)
    else:
        η_hat_red = η_hat / 2
        return (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_Elastic_net(sqrt(2 * q_hat) * ξ, 2 * V_hat, 0.5 * P_hat, reg_param) ** 2
        )


@njit(error_model="numpy", fastmath=False)
def V_integral_hastie_L2_reg_Linf_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + (1 / gamma)
        first_term = (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * DƔ_proximal_Elastic_net(
                sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, 0.5 * P_hat, reg_param
            )
        )
        second_term = gaussian(ξ, 0, 1) * DƔ_proximal_Elastic_net(
            sqrt(q_hat) * ξ, V_hat, 0.5 * P_hat, reg_param
        )
        return 0.5 * ((1 + gamma) * first_term + (1 - gamma) * second_term)
    else:
        η_hat_red = η_hat / 2
        return (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * DƔ_proximal_Elastic_net(sqrt(2 * q_hat) * ξ, 2 * V_hat, 0.5 * P_hat, reg_param)
        )


@njit(error_model="numpy", fastmath=False)
def P_integral_hastie_L2_reg_Linf_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
) -> float:
    η_hat = m_hat**2 / q_hat
    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + (1 / gamma)
        first_term = (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * abs(
                proximal_Elastic_net(
                    sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, 0.5 * P_hat, reg_param
                )
            )
        )
        second_term = gaussian(ξ, 0, 1) * abs(
            proximal_Elastic_net(sqrt(q_hat) * ξ, V_hat, 0.5 * P_hat, reg_param)
        )
        return 0.5 * (gamma * first_term + (1 - gamma) * second_term)
    else:
        η_hat_red = η_hat / 2
        return 0.5 * (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * abs(proximal_Elastic_net(sqrt(2 * q_hat) * ξ, 2 * V_hat, 0.5 * P_hat, reg_param))
        )


# -----------------------------------
def f_hastie_L2_reg_Linf_attack(
    m_hat: float,
    q_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
):
    η_hat = m_hat**2 / q_hat

    if gamma <= 1:
        domains = [
            (
                -BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
                -0.5 * P_hat / sqrt(q_hat) / sqrt(1 + 1 / gamma),
            ),
            (
                0.5 * P_hat / sqrt(q_hat) / sqrt(1 + 1 / gamma),
                BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
            ),
        ]
    else:
        domains = [
            (
                -BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
                -0.5 * P_hat / sqrt(2 * q_hat),
            ),
            (0.5 * P_hat / sqrt(2 * q_hat), BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1))),
        ]

    int_value_m = 0.0
    for domain in domains:
        int_value_m += quad(
            m_integral_hastie_L2_reg_Linf_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, gamma),
        )[0]
    m = int_value_m / sqrt(gamma)

    int_value_q = 0.0
    for domain in domains:
        int_value_q += quad(
            q_integral_hastie_L2_reg_Linf_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, gamma),
        )[0]
    q = 2 * int_value_q

    int_value_V = 0.0
    for domain in domains:
        int_value_V += quad(
            V_integral_hastie_L2_reg_Linf_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, gamma),
        )[0]
    V = 2 * int_value_V

    int_value_P = 0.0
    for domain in domains:
        int_value_P += quad(
            P_integral_hastie_L2_reg_Linf_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, gamma),
        )[0]
    P = 2 * int_value_P

    return m, q, V, P


# -----------------------------------


def q_latent_integral_hastie_L2_reg_Linf_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
):
    η_hat = m_hat**2 / q_hat

    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + (1 / gamma)
        return 0.5 * (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_Elastic_net(
                sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, 0.5 * P_hat, reg_param
            )
            ** 2
        )
    else:
        η_hat_red = η_hat / 2
        return 0.5 * (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_Elastic_net(sqrt(2 * q_hat) * ξ, 2 * V_hat, 0.5 * P_hat, reg_param) ** 2
        )


def q_latent_hastie_L2_reg_Linf_attack(
    m_hat: float,
    q_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
):
    η_hat = m_hat**2 / q_hat

    if gamma <= 1:
        domains = [
            (
                -BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
                -0.5 * P_hat / sqrt(q_hat) / sqrt(1 + 1 / gamma),
            ),
            (
                0.5 * P_hat / sqrt(q_hat) / sqrt(1 + 1 / gamma),
                BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
            ),
        ]
    else:
        domains = [
            (
                -BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
                -0.5 * P_hat / sqrt(2 * q_hat),
            ),
            (0.5 * P_hat / sqrt(2 * q_hat), BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1))),
        ]

    int_value_q_latent = 0.0
    for domain in domains:
        int_value_q_latent += quad(
            q_latent_integral_hastie_L2_reg_Linf_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, gamma),
        )[0]
    q_latent = 2 / gamma * int_value_q_latent

    return q_latent


# -----------------------------------
def q_features_integral_hastie_L2_reg_Linf_attack(
    ξ: float,
    q_hat: float,
    m_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
):
    η_hat = m_hat**2 / q_hat

    if gamma <= 1:
        η_hat_red = η_hat / (1 + gamma)
        gamma_tilde = 1 + (1 / gamma)
        first_term = (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_Elastic_net(
                sqrt(q_hat * gamma_tilde) * ξ, V_hat * gamma_tilde, 0.5 * P_hat, reg_param
            )
            ** 2
        )
        second_term = (
            gaussian(ξ, 0, 1)
            * proximal_Elastic_net(sqrt(q_hat) * ξ, V_hat, 0.5 * P_hat, reg_param) ** 2
        )
        return 0.5 * (gamma * first_term + (1 - gamma) * second_term)
    else:
        η_hat_red = η_hat / 2
        return 0.5 * (
            gaussian(ξ, 0, 1)
            * Z_w_Bayes_gaussian_prior(sqrt(η_hat_red) * ξ, η_hat_red, 0, 1)
            * proximal_Elastic_net(sqrt(2 * q_hat) * ξ, 2 * V_hat, 0.5 * P_hat, reg_param) ** 2
        )


def q_features_hastie_L2_reg_Linf_attack(
    m_hat: float,
    q_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    gamma: float,
):
    η_hat = m_hat**2 / q_hat

    if gamma <= 1:
        domains = [
            (
                -BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
                -0.5 * P_hat / sqrt(q_hat) / sqrt(1 + 1 / gamma),
            ),
            (
                0.5 * P_hat / sqrt(q_hat) / sqrt(1 + 1 / gamma),
                BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
            ),
        ]
    else:
        domains = [
            (
                -BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1)),
                -0.5 * P_hat / sqrt(2 * q_hat),
            ),
            (0.5 * P_hat / sqrt(2 * q_hat), BIG_NUMBER * ((1 + sqrt(η_hat) / (η_hat + 1)) ** (-1))),
        ]

    int_value_q_features = 0.0
    for domain in domains:
        int_value_q_features += quad(
            q_features_integral_hastie_L2_reg_Linf_attack,
            domain[0],
            domain[1],
            args=(q_hat, m_hat, V_hat, P_hat, reg_param, gamma),
        )[0]
    q_features = 2 * int_value_q_features

    return q_features
