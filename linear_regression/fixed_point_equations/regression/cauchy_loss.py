from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad, dblquad
from ...aux_functions.moreau_proximals import proximal_Cauchy
from ...aux_functions.misc import gaussian
from ...aux_functions.loss_functions import DDz_cauchy_loss
from ...aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
    f_out_Bayes_decorrelated_noise,
    DZ_out_Bayes_decorrelated_noise,
)

BIG_NUMBER = 15


@njit(error_model="numpy", fastmath=False)
def m_int_Cauchy_decorrelated_noise(
    ξ: float,
    y: float,
    q: float,
    m: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    τ: float,
):
    η = m**2 / q
    proximal = proximal_Cauchy(y, sqrt(q) * ξ, V, τ)
    return (
        gaussian(ξ, 0, 1)
        * DZ_out_Bayes_decorrelated_noise(
            y, sqrt(η) * ξ, 1 - η, delta_in, delta_out, percentage, beta
        )
        * (proximal - sqrt(q) * ξ)
        / V
    )


@njit(error_model="numpy", fastmath=False)
def q_int_Cauchy_decorrelated_noise(
    ξ: float,
    y: float,
    q: float,
    m: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    τ: float,
):
    η = m**2 / q
    proximal = proximal_Cauchy(y, sqrt(q) * ξ, V, τ)
    return (
        gaussian(ξ, 0, 1)
        * Z_out_Bayes_decorrelated_noise(
            y, sqrt(η) * ξ, 1 - η, delta_in, delta_out, percentage, beta
        )
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


@njit(error_model="numpy", fastmath=False)
def V_int_Cauchy_decorrelated_noise(
    ξ: float,
    y: float,
    q: float,
    m: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    τ: float,
):
    η = m**2 / q
    proximal = proximal_Cauchy(y, sqrt(q) * ξ, V, τ)
    Dproximal = (1 + V * DDz_cauchy_loss(y, proximal, τ)) ** (-1)
    return (
        gaussian(ξ, 0, 1)
        * Z_out_Bayes_decorrelated_noise(
            y, sqrt(η) * ξ, 1 - η, delta_in, delta_out, percentage, beta
        )
        * (Dproximal - 1)
        / V
    )


# -----------------------------------


def f_hat_Cauchy_decorrelated_noise(
    m,
    q,
    V,
    alpha,
    delta_in,
    delta_out,
    percentage,
    beta,
    tau,
):
    y_dom = [-BIG_NUMBER, BIG_NUMBER]
    xi_dom = [-BIG_NUMBER, BIG_NUMBER]

    int_value_m_hat = dblquad(
        m_int_Cauchy_decorrelated_noise,
        xi_dom[0],
        xi_dom[1],
        lambda x: y_dom[0],
        lambda x: y_dom[1],
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau),
    )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = dblquad(
        q_int_Cauchy_decorrelated_noise,
        xi_dom[0],
        xi_dom[1],
        lambda x: y_dom[0],
        lambda x: y_dom[1],
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau),
    )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = dblquad(
        V_int_Cauchy_decorrelated_noise,
        xi_dom[0],
        xi_dom[1],
        lambda x: y_dom[0],
        lambda x: y_dom[1],
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau),
    )[0]
    V_hat = -alpha * int_value_V_hat

    return m_hat, q_hat, V_hat
