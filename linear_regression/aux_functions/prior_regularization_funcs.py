from numba import vectorize, njit
from math import exp, sqrt, cosh, sinh
from numpy import sum


@vectorize("float64(float64, float64, float64, float64)")
def Z_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return exp(
        (sigma * gamma**2 + 2 * gamma * mu - Lambda * mu**2) / (2 * (Lambda * sigma + 1))
    ) / sqrt(Lambda * sigma + 1)


@vectorize("float64(float64, float64, float64, float64)")
def f_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return (gamma * sigma + mu) / (1 + sigma * Lambda)


@vectorize("float64(float64, float64, float64, float64)")
def Df_w_Bayes_gaussian_prior(gamma: float, Lambda: float, mu: float, sigma: float) -> float:
    return sigma / (1 + sigma * Lambda)


# --------------------------


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


@vectorize("float64(float64, float64, float64)")
def Z_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    return exp(gamma**2 / (2.0 * (reg_param + Lambda))) / sqrt(reg_param + Lambda)


@vectorize("float64(float64, float64, float64)")
def f_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    return gamma / (reg_param + Lambda)


@vectorize("float64(float64, float64, float64)")
def Df_w_L2_regularization(gamma: float, Lambda: float, reg_param: float) -> float:
    return 1.0 / (reg_param + Lambda)


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
