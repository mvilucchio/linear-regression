from numba import vectorize, njit
from math import exp, sqrt, cosh, sinh, pi, log
from numpy import sum


# ------------------------------ gaussian prior ------------------------------ #
@vectorize("float64(float64, float64, float64, float64)")
def Z_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, V: float) -> float:
    return exp(
        (V * gamma**2 + 2 * gamma * mu - Lambda * mu**2) / (2 * (Lambda * V + 1))
    ) / sqrt(Lambda * V + 1)


@vectorize("float64(float64, float64, float64, float64)")
def DZ_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, V: float) -> float:
    return (
        exp((2 * gamma * mu - Lambda * mu**2 + gamma**2 * V) / (2 + 2 * Lambda * V))
        * (mu + gamma * V)
    ) / (1 + Lambda * V) ** 1.5


@vectorize("float64(float64, float64, float64, float64)")
def f_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, V: float) -> float:
    return (gamma * V + mu) / (1 + V * Lambda)


@vectorize("float64(float64, float64, float64, float64)")
def Df_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, V: float) -> float:
    return V / (1 + V * Lambda)


def log_Z_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, V: float) -> float:
    return (V * gamma**2 + 2 * gamma * mu - Lambda * mu**2) / (
        2 * (Lambda * V + 1)
    ) - 0.5 * log(Lambda * V + 1)


# ----------------------------
@vectorize("float64(float64, float64, float64, float64, float64)")
def gauss_Z_w_Bayes_gaussian_prior(
    ξ: float, m_hat: float, q_hat: float, mu: float, V: float
) -> float:
    η_hat = m_hat**2 / q_hat
    return exp(
        -0.5 * ξ**2
        + (V * η_hat * ξ**2 + 2 * sqrt(η_hat) * ξ * mu - η_hat * mu**2) / (2 * (η_hat * V + 1))
    ) / sqrt(2 * pi * (η_hat * V + 1))


# ---------------------------- sparse binary prior --------------------------- #
@vectorize("float64(float64, float64, float64)")
def Z_w_Bayes_sparse_binary_weights(gamma: float, Lambda: float, rho: float) -> float:
    return rho + exp(-Lambda / 2) * (1 - rho) * cosh(gamma)


@vectorize("float64(float64, float64, float64)")
def f_w_Bayes_sparse_binary_weights(gamma: float, Lambda: float, rho: float) -> float:
    return (exp(-Lambda / 2) * (1 - rho) * sinh(gamma)) / (
        rho + exp(-Lambda / 2) * (1 - rho) * cosh(gamma)
    )


@vectorize("float64(float64, float64, float64)")
def Df_w_Bayes_sparse_binary_weights(gamma: float, Lambda: float, rho: float) -> float:
    return (exp(-Lambda / 2) * (1 - rho) * cosh(gamma)) / (
        rho + exp(-Lambda / 2) * (1 - rho) * cosh(gamma)
    )


# --------------------------


@vectorize("float64(float64, float64, float64, float64, float64)")
def Z_w_Bayes_Gauss_Bernoulli_weights(
    gamma: float, Lambda: float, rho: float, mu: float, V: float
) -> float:
    return (
        rho
        * exp((2 * gamma * mu - Lambda * mu**2 + gamma**2 * V) / (2 + 2 * Lambda * V))
        / sqrt(1 + Lambda * V)
    )


@vectorize("float64(float64, float64, float64, float64, float64)")
def f_w_Bayes_Gauss_Bernoulli_weights(
    gamma: float, Lambda: float, rho: float, mu: float, V: float
) -> float:
    return (exp((gamma * (2 * mu + gamma * V)) / (2 + 2 * Lambda * V)) * rho * (mu + gamma * V)) / (
        (1 + Lambda * V)
        * (
            exp((gamma * (2 * mu + gamma * V)) / (2 + 2 * Lambda * V)) * rho
            - exp((Lambda * mu**2) / (2 + 2 * Lambda * V)) * (-1 + rho) * sqrt(1 + Lambda * V)
        )
    )


@vectorize("float64(float64, float64, float64, float64, float64)")
def Df_w_Bayes_Gauss_Bernoulli_weights(
    gamma: float, Lambda: float, rho: float, mu: float, V: float
) -> float:
    return (
        exp((gamma * (2 * mu + gamma * V)) / (2 + 2 * Lambda * V))
        * rho
        * (
            exp((gamma * (2 * mu + gamma * V)) / (2 + 2 * Lambda * V)) * rho * V * (1 + Lambda * V)
            - exp((Lambda * mu**2) / (2 + 2 * Lambda * V))
            * (-1 + rho)
            * sqrt(1 + Lambda * V)
            * (mu**2 + V + 2 * gamma * mu * V + (gamma**2 + Lambda) * V**2)
        )
    ) / (
        (1 + Lambda * V) ** 2
        * (
            exp((gamma * (2 * mu + gamma * V)) / (2 + 2 * Lambda * V)) * rho
            - exp((Lambda * mu**2) / (2 + 2 * Lambda * V)) * (-1 + rho) * sqrt(1 + Lambda * V)
        )
        ** 2
    )


# --------------------------


@vectorize("float64(float64, float64, float64)")
def Z_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    return exp((gamma**2 * Lambda) / (2 * (reg_param + Lambda) ** 2))


@vectorize("float64(float64, float64, float64)")
def f_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    return gamma / (reg_param + Lambda)


@vectorize("float64(float64, float64, float64)")
def Df_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    return 1.0 / (reg_param + Lambda)


# -------------------------


# not sure about this one
@vectorize("float64(float64, float64, float64)")
def f_w_L1_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    if gamma > reg_param:
        return (gamma - reg_param) / Lambda
    elif gamma + reg_param < 0:
        return (gamma + reg_param) / Lambda
    else:
        return 0.0


@vectorize("float64(float64, float64, float64)")
def Df_w_L1_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    if abs(gamma) > reg_param:
        return 1 / Lambda
    else:
        return 0.0


# -------------------------


# @vectorize("float64(float64, float64, float64, float64)")
def Z_w_ElasticNet_regularization(
    gamma: float, Lambda: float, reg_param_1: float, reg_param_2: float
) -> float:
    raise NotImplementedError


@vectorize("float64(float64, float64, float64, float64)")
def f_w_ElasticNet_regularization(
    gamma: float, Lambda: float, reg_param_1: float, reg_param_2: float
) -> float:
    l2_factor = Lambda / (reg_param_2 + Lambda)
    if gamma > reg_param_1:
        return l2_factor * (gamma - reg_param_1) / Lambda
    elif gamma + reg_param_1 < 0:
        return l2_factor * (gamma + reg_param_1) / Lambda
    else:
        return 0.0


@vectorize("float64(float64, float64, float64, float64)")
def Df_w_ElasticNet_regularization(
    gamma: float, Lambda: float, reg_param_1: float, reg_param_2: float
) -> float:
    l2_factor = Lambda / (reg_param_2 + Lambda)
    if abs(gamma) > reg_param_1:
        return l2_factor / Lambda
    else:
        return 0.0


# -------------------------


# @vectorize("float64(float64, float64, float64)")
# def Z_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
#     return exp(gamma**2 / (2.0 * (reg_param + Lambda))) / sqrt(reg_param + Lambda)


# @vectorize("float64(float64, float64, float64)")
# def f_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
#     return gamma / (reg_param + Lambda)


# @vectorize("float64(float64, float64, float64)")
# def Df_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
#     return 1.0 / (reg_param + Lambda)


# -------------------------


@njit
def f_w_projection_on_sphere(gamma, Lambda: float, q_fixed: float):
    d = gamma.shape[0]
    gamma_norm = sqrt(sum(gamma**2))
    return gamma * sqrt(d * q_fixed) / gamma_norm


@njit
def Df_w_projection_on_sphere(gamma, Lambda: float, q_fixed: float):
    d = gamma.shape[0]
    gamma_norm = sqrt(sum(gamma**2))
    return sqrt(d * q_fixed) / gamma_norm
