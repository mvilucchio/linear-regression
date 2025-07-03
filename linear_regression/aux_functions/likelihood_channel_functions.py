from numba import vectorize
from math import exp, sqrt, pow, erf, pi, log
from ..aux_functions.misc import gaussian
import numpy as np
from numba import njit


@vectorize("float64(float64, float64, float64, float64)")
def Z_out_Bayes_single_noise_classif(y: float, omega: float, V: float, delta: float) -> float:
    return 0.5 * (
        gaussian(y, 1, delta) * (1 + erf(omega / sqrt(2 * V)))
        + gaussian(y, -1, delta) * (1 - erf(omega / sqrt(2 * V)))
    )


@vectorize("float64(float64, float64, float64, float64)")
def Z_out_Bayes_f_out_Bayes_single_noise_classif(
    y: float, omega: float, V: float, delta: float
) -> float:
    return (
        exp(-0.5 * (1 + y) ** 2 / delta - omega**2 / (2.0 * V)) * (-1 + exp((2 * y) / delta))
    ) / (2.0 * pi * sqrt(V * delta))


# -----------------------------------
def Z_out_Bayes_sign_flip_classif(y: float, omega: float, V: float, eps: float) -> float:
    return 0.5 * (1 + (1 - 2 * eps) * erf(y * omega / sqrt(2 * V)))


def f_out_Bayes_sign_flip_classif(y: float, omega: float, V: float, eps: float) -> float:
    num = y * sqrt(2 / pi) * exp(-(omega**2) / V) * (1 - 2 * eps)
    denom = sqrt(V) * (1 + (1 - 2 * eps) * erf(y * omega / sqrt(2 * V)))
    return num / denom


# -----------------------------------


@vectorize("float64(float64, float64, float64, float64)")
def Z_out_Bayes_single_noise(y: float, omega: float, V: float, delta: float) -> float:
    return exp(-((y - omega) ** 2) / (2 * (V + delta))) / sqrt(2 * pi * (V + delta))


@vectorize("float64(float64, float64, float64, float64)")
def f_out_Bayes_single_noise(y: float, omega: float, V: float, delta: float) -> float:
    return (y - omega) / (V + delta)


# -----------------------------------


@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")
def Z_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_in: float,
    delta_out: float,
    eps: float,
    beta: float,
) -> float:
    return (1 - eps) * exp(-((y - omega) ** 2) / (2 * (V + delta_in))) / sqrt(
        2 * pi * (V + delta_in)
    ) + eps * exp(-((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_out))) / sqrt(
        2 * pi * (beta**2 * V + delta_out)
    )


@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")
def DZ_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_in: float,
    delta_out: float,
    eps: float,
    beta: float,
) -> float:
    small_exponential = exp(-((y - omega) ** 2) / (2 * (V + delta_in))) / sqrt(2 * pi)
    large_exponential = exp(-((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_out))) / sqrt(
        2 * pi
    )

    return (1 - eps) * small_exponential * (y - omega) / pow(
        V + delta_in, 1.5
    ) + eps * beta * large_exponential * (y - beta * omega) / pow(beta**2 * V + delta_out, 1.5)


@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")
def f_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_in: float,
    delta_out: float,
    eps: float,
    beta: float,
) -> float:
    exp_in = exp(-((y - omega) ** 2) / (2 * (V + delta_in)))
    exp_out = exp(-((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_out)))
    return (
        (y - omega) * (1 - eps) * exp_in / pow(V + delta_in, 3 / 2)
        + eps * beta * (y - beta * omega) * exp_out / pow(beta**2 * V + delta_out, 3 / 2)
    ) / (
        (1 - eps) * exp_in / pow(V + delta_in, 1 / 2)
        + eps * exp_out / pow(beta**2 * V + delta_out, 1 / 2)
    )


@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")
def Df_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_in: float,
    delta_out: float,
    eps: float,
    beta: float,
) -> float:
    f_out_2 = -f_out_Bayes_decorrelated_noise(y, omega, V, delta_in, delta_out, eps, beta) ** 2

    exp_in = exp(-((y - omega) ** 2) / (2 * (V + delta_in)))
    exp_out = exp(-((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_out)))

    return f_out_2 + (
        (1 - eps) * (y - omega) ** 2 * exp_in / pow(V + delta_in, 2.5)
        + exp_out * (y - beta * omega) ** 2 * (beta**2 * eps) / pow(V * beta**2 + delta_out, 2.5)
        - (1 - eps) * exp_in / pow(V + delta_in, 1.5)
        - exp_out * beta**2 * eps / pow(V * beta**2 + delta_out, 1.5)
    ) / (
        (1 - eps) * exp_in / pow(V + delta_in, 0.5)
        + eps * exp_out / pow(beta**2 * V + delta_out, 0.5)
    )


@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")
def log_Z_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_in: float,
    delta_out: float,
    eps: float,
    beta: float,
) -> float:
    term1_exponent = -((y - omega) ** 2) / (2 * (V + delta_in))
    term2_exponent = -((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_out))

    max_exponent = max(term1_exponent, term2_exponent)

    term1 = (1 - eps) * exp(term1_exponent - max_exponent) / sqrt(2 * pi * (V + delta_in))
    term2 = eps * exp(term2_exponent - max_exponent) / sqrt(2 * pi * (beta**2 * V + delta_out))

    result = max_exponent + log(term1 + term2)

    return result

@vectorize("float64(float64, float64, float64)")
def f_out_L2(y: float, omega: float, V: float) -> float:
    return (y - omega) / (1 + V)


@vectorize("float64(float64, float64, float64)")
def Df_out_L2(y: float, omega: float, V: float) -> float:
    return -1.0 / (1 + V)


# -----------------------------------


# @njit(error_model="numpy", fastmath=True)
@vectorize("float64(float64, float64, float64)")
def f_out_L1(y: float, omega: float, V: float) -> float:
    if y - omega < -V:
        return -1.0
    elif y - omega > V:
        return 1.0
    else:
        return (y - omega) / V


@vectorize(["float64(float64, float64, float64)"])
def Df_out_L1(y: float, omega: float, V: float) -> float:
    if abs(y - omega) > V:
        return 0.0
    else:
        return -1.0 / V


# -----------------------------------
@vectorize(["float64(float64, float64, float64, float64)"])
def f_out_Huber(y: float, omega: float, V: float, a: float) -> float:
    if a + a * V + omega < y:
        return a
    elif abs(y - omega) <= a + a * V:
        return (y - omega) / (1 + V)
    elif omega > a + a * V + y:
        return -a
    else:
        return 0.0


@vectorize(["float64(float64, float64, float64, float64)"])
def Df_out_Huber(y: float, omega: float, V: float, a: float) -> float:
    if (y < omega and a + a * V + y < omega) or (a + a * V + omega < y):
        return 0.0
    else:
        return -1.0 / (1 + V)


# -----------------------------------
@vectorize("float64(float64, float64, float64)")
def f_out_Hinge(y: float, omega: float, V: float) -> float:
    if y * omega < 1.0 - V:
        return y
    elif y * omega < 1.0:
        return (y - omega) / V
    else:
        return 0.0


@vectorize(["float64(float64, float64, float64)"])
def Df_out_Hinge(y: float, omega: float, V: float) -> float:
    if (y * omega < 1.0) and (y * omega > 1.0 - V):
        return -1.0 / V
    else:
        return 0.0


# -----------------------------------
# @vectorize("float64(float64, float64, float64)")
def f_out_Logistic(y: float, omega: float, V: float) -> float:
    raise NotImplementedError


# @vectorize(["float64(float64, float64, float64)"])
def Df_out_Logistic(y: float, omega: float, V: float) -> float:
    raise NotImplementedError


# -----------------------------------
@vectorize("float64(float64, float64, float64)")
def f_out_phase_retrival_simple(y: float, omega: float, V: float) -> float:
    return 2 * omega * y / (1 + 2 * y * V)


@vectorize(["float64(float64, float64, float64)"])
def Df_out_phase_retrival_simple(y: float, omega: float, V: float) -> float:
    return 2 * y / (1 + 2 * y * V)
