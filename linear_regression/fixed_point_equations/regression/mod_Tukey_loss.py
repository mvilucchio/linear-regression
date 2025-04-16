from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad, dblquad
from ...aux_functions.moreau_proximals import proximal_Tukey_modified_quad
from ...aux_functions.misc import gaussian
from ...aux_functions.loss_functions import DDz_mod_tukey_loss_cubic
from ...aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
    f_out_Bayes_decorrelated_noise,
    DZ_out_Bayes_decorrelated_noise,
)

BIG_NUMBER = 15


@njit(error_model="numpy", fastmath=False)
def m_int_mod_Tukey_decorrelated_noise(
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
    c: float,
):
    η = m**2 / q
    proximal = proximal_Tukey_modified_quad(y, sqrt(q) * ξ, V, τ, c)
    return (
        gaussian(ξ, 0, 1)
        * DZ_out_Bayes_decorrelated_noise(
            y, sqrt(η) * ξ, 1 - η, delta_in, delta_out, percentage, beta
        )
        * (proximal - sqrt(q) * ξ)
        / V
    )


@njit(error_model="numpy", fastmath=False)
def q_int_mod_Tukey_decorrelated_noise(
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
    c: float,
):
    η = m**2 / q
    proximal = proximal_Tukey_modified_quad(y, sqrt(q) * ξ, V, τ, c)
    return (
        gaussian(ξ, 0, 1)
        * Z_out_Bayes_decorrelated_noise(
            y, sqrt(η) * ξ, 1 - η, delta_in, delta_out, percentage, beta
        )
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


@njit(error_model="numpy", fastmath=False)
def V_int_mod_Tukey_decorrelated_noise(
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
    c: float,
):
    η = m**2 / q
    proximal = proximal_Tukey_modified_quad(y, sqrt(q) * ξ, V, τ, c)
    Dproximal = (1 + V * DDz_mod_tukey_loss_cubic(y, proximal, τ, c)) ** (-1)
    return (
        gaussian(ξ, 0, 1)
        * Z_out_Bayes_decorrelated_noise(
            y, sqrt(η) * ξ, 1 - η, delta_in, delta_out, percentage, beta
        )
        * (Dproximal - 1)
        / V
    )


# -----------------------------------


def f_hat_mod_Tukey_decorrelated_noise(
    m,
    q,
    V,
    alpha,
    delta_in,
    delta_out,
    percentage,
    beta,
    tau,
    c,
):
    y_dom = [-BIG_NUMBER, BIG_NUMBER]
    xi_dom = [-BIG_NUMBER, BIG_NUMBER]

    int_value_m_hat = dblquad(
        m_int_mod_Tukey_decorrelated_noise,
        xi_dom[0],
        xi_dom[1],
        lambda x: y_dom[0],
        lambda x: y_dom[1],
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),
    )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = dblquad(
        q_int_mod_Tukey_decorrelated_noise,
        xi_dom[0],
        xi_dom[1],
        lambda x: y_dom[0],
        lambda x: y_dom[1],
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),
    )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = dblquad(
        V_int_mod_Tukey_decorrelated_noise,
        xi_dom[0],
        xi_dom[1],
        lambda x: y_dom[0],
        lambda x: y_dom[1],
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),
    )[0]
    V_hat = -alpha * int_value_V_hat

    return m_hat, q_hat, V_hat


# Change of variable to (u,v)

@njit(error_model="numpy", fastmath=False)
def V_prime(q, m, V,delta_in, delta_out, percentage, beta, tau):
    """Donne la borne d'intégration pour l'intégrale sur u"""
    eta = m**2 / q
    J_beta_out = beta*sqrt(eta) - sqrt(q) # Jacobien de la transformation
    delta_prime_beta_out = (1-eta)*beta**2 + delta_out
    V_prime_out = delta_prime_beta_out*J_beta_out**2 / (delta_prime_beta_out+J_beta_out**2)

    J_beta_in = 1.0*sqrt(eta) - sqrt(q) # Jacobien de la transformation
    delta_prime_beta_in = (1-eta) + delta_in
    V_prime_in = delta_prime_beta_in*J_beta_in**2 / (delta_prime_beta_in+J_beta_in**2)

    return sqrt(max(V_prime_in, V_prime_out))

@njit(error_model="numpy", fastmath=False)
def gaussian_2d_uv_Tukey(u, v, J_beta, delta_prime):
    """Calcule la densité de la gaussienne 2D en (u,v)."""
    exponent = -0.5 * (
        + u**2 / delta_prime
        + (1.0 / J_beta**2) * (u-v)**2
    )
    normalisation = 1 / (2 * pi * sqrt(delta_prime)*J_beta)
    # La constante de normalisation est gérée par Z_norm dans les intégrandes
    return normalisation*exp(exponent)

SMALL_NUMBER = 1e-10

@njit(error_model="numpy", fastmath=False)
def m_star_int_uv_Tukey(u, v, q, m, V, delta, beta, tau, c):
    """Intégrande pour le calcul de m_hat_star dans les coordonnées (u,v)"""
    if beta ==0 : return 0
    eta = m**2 / q
    J_beta = beta*sqrt(eta) - sqrt(q) # Jacobien de la transformation

    if abs(J_beta) < SMALL_NUMBER: 
        print("J_beta < SMALL_NUMBER, retour de 0.0")
        return 0.0

    delta_prime = (1-eta)*beta**2 + delta

    weight = gaussian_2d_uv_Tukey(u, v, J_beta, delta_prime)* beta*u/delta_prime
    
    xi = (v - u) / J_beta
    y = (beta * sqrt(eta) * v - sqrt(q) * u) / J_beta
    omega = sqrt(q) * xi

    prox = proximal_Tukey_modified_quad(y, omega, V, tau, c)
    f_out = (prox - omega) / V
    return  weight*f_out

@njit(error_model="numpy", fastmath=False)
def m_int_uv_Tukey(u, v, q, m, V, delta_in, delta_out, percentage, beta, tau, c):
    return (1-percentage)*m_star_int_uv_Tukey(u, v, q, m, V, delta_in, 1, tau, c) + percentage*m_star_int_uv_Tukey(u, v, q, m, V, delta_out, beta, tau, c)

@njit(error_model="numpy", fastmath=False)
def q_star_int_uv_Tukey(u, v, q, m, V, delta, beta, tau, c):
    """Intégrande pour le calcul de q_hat_star dans les coordonnées (u,v)"""
    eta = m**2 / q
    J_beta = beta*sqrt(eta) - sqrt(q) # Jacobien de la transformation

    if abs(J_beta) < SMALL_NUMBER: 
        print("J_beta < SMALL_NUMBER, retour de 0.0")
        return 0.0

    delta_prime = (1-eta)*beta**2 + delta

    weight = gaussian_2d_uv_Tukey(u, v, J_beta, delta_prime)
    
    xi = (v - u) / J_beta
    y = (beta * sqrt(eta) * v - sqrt(q) * u) / J_beta
    omega = sqrt(q) * xi

    prox = proximal_Tukey_modified_quad(y, omega, V, tau, c)
    f_out = (prox - omega) / V
    return  weight*f_out**2

@njit(error_model="numpy", fastmath=False)
def q_int_uv_Tukey(u, v, q, m, V, delta_in, delta_out, percentage, beta, tau, c):
    return (1-percentage)*q_star_int_uv_Tukey(u, v, q, m, V, delta_in, 1, tau, c) + percentage*q_star_int_uv_Tukey(u, v, q, m, V, delta_out, beta, tau, c)

@njit(error_model="numpy", fastmath=False)
def V_star_int_uv_Tukey(u, v, q, m, V, delta, beta, tau, c):
    """Intégrande pour le calcul de q_hat_star dans les coordonnées (u,v)"""
    eta = m**2 / q
    J_beta = beta*sqrt(eta) - sqrt(q) # Jacobien de la transformation

    if abs(J_beta) < SMALL_NUMBER: 
        print("J_beta < SMALL_NUMBER, retour de 0.0")
        return 0.0

    delta_prime = (1-eta)*beta**2 + delta

    weight = gaussian_2d_uv_Tukey(u, v, J_beta, delta_prime)
    
    xi = (v - u) / J_beta
    y = (beta * sqrt(eta) * v - sqrt(q) * u) / J_beta
    omega = sqrt(q) * xi

    prox = proximal_Tukey_modified_quad(y, omega, V, tau, c)
    Dproximal = (1 + V * DDz_mod_tukey_loss_cubic(y, prox, tau, c)) ** (-1)

    return  weight*(Dproximal - 1)/ V

@njit(error_model="numpy", fastmath=False)
def V_int_uv_Tukey(u, v, q, m, V, delta_in, delta_out, percentage, beta, tau, c):
    return (1-percentage)*V_star_int_uv_Tukey(u, v, q, m, V, delta_in, 1, tau, c) + percentage*V_star_int_uv_Tukey(u, v, q, m, V, delta_out, beta, tau, c)

DEFAULT_N_STD_U = 10.0 # Nombre d'écarts-types pour l'intégration en u

def f_hat_uv_mod_Tukey_decorrelated_noise(
    m,
    q,
    V,
    alpha,
    delta_in,
    delta_out,
    percentage,
    beta,
    tau,
    c,
):
    v_dom = [-tau, tau]
    u_var = DEFAULT_N_STD_U *V_prime(q, m, V, delta_in, delta_out, percentage, beta, tau)

    int_value_m_hat = dblquad(
        m_int_uv_Tukey,
        v_dom[0],
        v_dom[1],
        lambda x: x - u_var,
        lambda x: x + u_var,
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),epsabs=1e-12
    )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = dblquad(
        q_int_uv_Tukey,
        v_dom[0],
        v_dom[1],
        lambda x: x - u_var,
        lambda x: x + u_var,
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),epsabs=1e-12
    )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = dblquad(
        V_int_uv_Tukey,
        v_dom[0],
        v_dom[1],
        lambda x: x - u_var,
        lambda x: x + u_var,
        args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),epsabs=1e-12
    )[0]
    V_hat = -alpha * int_value_V_hat

    return m_hat, q_hat, V_hat
