from numba import vectorize, njit
from math import exp, sqrt, cosh, sinh, pi, log
from numpy import sum


# ------------------------------ gaussian prior ------------------------------ #
@vectorize("float64(float64, float64, float64, float64)")
def Z_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return exp(
        (sigma * gamma**2 + 2 * gamma * mu - Lambda * mu**2) / (2 * (Lambda * sigma + 1))
    ) / sqrt(Lambda * sigma + 1)


@vectorize("float64(float64, float64, float64, float64)")
def DZ_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return (
        exp((2 * gamma * mu - Lambda * mu**2 + gamma**2 * sigma) / (2 + 2 * Lambda * sigma))
        * (mu + gamma * sigma)
    ) / (1 + Lambda * sigma) ** 1.5


@vectorize("float64(float64, float64, float64, float64)")
def f_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return (gamma * sigma + mu) / (1 + sigma * Lambda)


@vectorize("float64(float64, float64, float64, float64)")
def Df_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return sigma / (1 + sigma * Lambda)


def log_Z_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return (sigma * gamma**2 + 2 * gamma * mu - Lambda * mu**2) / (
        2 * (Lambda * sigma + 1)
    ) - 0.5 * log(Lambda * sigma + 1)


# ----------------------------
@vectorize("float64(float64, float64, float64, float64, float64)")
def gauss_Z_w_Bayes_gaussian_prior(
    ξ: float, m_hat: float, q_hat: float, mu: float, sigma: float
) -> float:
    η_hat = m_hat**2 / q_hat
    return exp(
        -0.5 * ξ**2
        + (sigma * η_hat * ξ**2 + 2 * sqrt(η_hat) * ξ * mu - η_hat * mu**2)
        / (2 * (η_hat * sigma + 1))
    ) / sqrt(2 * pi * (η_hat * sigma + 1))


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
    gamma: float, Lambda: float, rho: float, mu: float, sigma: float
) -> float:
    return (
        rho
        * exp((2 * gamma * mu - Lambda * mu**2 + gamma**2 * sigma) / (2 + 2 * Lambda * sigma))
        / sqrt(1 + Lambda * sigma)
    )


@vectorize("float64(float64, float64, float64, float64, float64)")
def f_w_Bayes_Gauss_Bernoulli_weights(
    gamma: float, Lambda: float, rho: float, mu: float, sigma: float
) -> float:
    return (
        exp((gamma * (2 * mu + gamma * sigma)) / (2 + 2 * Lambda * sigma))
        * rho
        * (mu + gamma * sigma)
    ) / (
        (1 + Lambda * sigma)
        * (
            exp((gamma * (2 * mu + gamma * sigma)) / (2 + 2 * Lambda * sigma)) * rho
            - exp((Lambda * mu**2) / (2 + 2 * Lambda * sigma))
            * (-1 + rho)
            * sqrt(1 + Lambda * sigma)
        )
    )


@vectorize("float64(float64, float64, float64, float64, float64)")
def Df_w_Bayes_Gauss_Bernoulli_weights(
    gamma: float, Lambda: float, rho: float, mu: float, sigma: float
) -> float:
    return (
        exp((gamma * (2 * mu + gamma * sigma)) / (2 + 2 * Lambda * sigma))
        * rho
        * (
            exp((gamma * (2 * mu + gamma * sigma)) / (2 + 2 * Lambda * sigma))
            * rho
            * sigma
            * (1 + Lambda * sigma)
            - exp((Lambda * mu**2) / (2 + 2 * Lambda * sigma))
            * (-1 + rho)
            * sqrt(1 + Lambda * sigma)
            * (mu**2 + sigma + 2 * gamma * mu * sigma + (gamma**2 + Lambda) * sigma**2)
        )
    ) / (
        (1 + Lambda * sigma) ** 2
        * (
            exp((gamma * (2 * mu + gamma * sigma)) / (2 + 2 * Lambda * sigma)) * rho
            - exp((Lambda * mu**2) / (2 + 2 * Lambda * sigma))
            * (-1 + rho)
            * sqrt(1 + Lambda * sigma)
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
