from numba import vectorize, njit
from numpy import sign as np_sign
from math import exp, log, tanh, cosh, sqrt, pow, pi
from numpy import log1p


@vectorize("float64(float64, float64, float64)")
def power_regularisation(x: float, alpha: float, reg_param: float) -> float:
    return reg_param * pow(abs(x), alpha)


@vectorize("float64(float64, float64, float64)")
def Dx_power_regularisation(x: float, alpha: float, reg_param: float) -> float:
    if alpha == 1:
        return reg_param * np_sign(x)
    return reg_param * alpha * pow(abs(x), alpha - 1) * np_sign(x)


@vectorize("float64(float64, float64, float64)")
def DDx_power_regularisation(x: float, alpha: float, reg_param: float) -> float:
    if alpha == 1:  # and x == 0:
        return 0.0  # 2 * reg_param
    elif alpha == 2:
        return reg_param * alpha
    return reg_param * alpha * (alpha - 1) * pow(abs(x), alpha - 2)


@vectorize("float64(float64, float64, float64)")
def DxDreg_param_power_regularisation(x: float, alpha: float, reg_param: float) -> float:
    if alpha == 1:
        return np_sign(x)
    return alpha * pow(abs(x), alpha - 1) * np_sign(x)


@vectorize("float64(float64, float64, float64)")
def Dreg_param_power_regularisation(x: float, alpha: float, reg_param: float) -> float:
    return pow(abs(x), alpha)
