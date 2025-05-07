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


# Change of variable to (xi, gamma)

@njit(error_model="numpy", fastmath=False)
def mu(q, m, V,delta_in, delta_out, percentage, beta, tau):
    """Donne la variance pour l'intégrale sur xi"""
    eta = m**2 / q
    J_beta_out = beta*sqrt(eta) - sqrt(q) # Jacobien de la transformation
    delta_prime_beta_out = (1-eta)*beta**2 + delta_out
    mu_out = 1+J_beta_out**2 / delta_prime_beta_out

    J_beta_in = 1.0*sqrt(eta) - sqrt(q) # Jacobien de la transformation
    delta_prime_beta_in = (1-eta) + delta_in
    mu_in = 1+J_beta_in**2 / delta_prime_beta_in

    return mu_in, mu_out, J_beta_in, J_beta_out, delta_prime_beta_in, delta_prime_beta_out

@njit(error_model="numpy", fastmath=False)
def gaussian_2d_xigamma_Tukey(xi, gamma, J_beta, delta_prime):
    """Calcule la densité de la gaussienne 2D en (xi,gamma)."""
    exponent = -0.5 * (
        + xi**2 
        + (J_beta*xi-gamma)**2 / delta_prime
    )
    normalisation = 1 / (2 * pi * sqrt(delta_prime))
    return normalisation*exp(exponent)

@njit(error_model="numpy", fastmath=False)
def m_star_int_xigamma_Tukey(xi, gamma, q, m, V, delta, beta, tau, c):
    """Intégrande pour le calcul de m_hat_star dans les coordonnées (xi,gamma)"""
    if beta ==0 : return 0
    eta = m**2 / q
    J_beta = beta*sqrt(eta) - sqrt(q) # Jacobien

    delta_prime = (1-eta)*beta**2 + delta
    y = gamma +sqrt(q) * xi

    weight = gaussian_2d_xigamma_Tukey(xi, gamma, J_beta, delta_prime)* beta*(y-beta*sqrt(eta)*xi)/delta_prime

    prox = proximal_Tukey_modified_quad(y, sqrt(q)*xi, V, tau, c)
    f_out = (prox - sqrt(q)*xi) / V
    return  weight*f_out

@njit(error_model="numpy", fastmath=False)
def q_star_int_xigamma_Tukey(xi, gamma, q, m, V, delta, beta, tau, c):
    """Intégrande pour le calcul de q_hat_star dans les coordonnées (xi,gamma)"""
    eta = m**2 / q
    J_beta = beta*sqrt(eta) - sqrt(q) # Jacobien

    delta_prime = (1-eta)*beta**2 + delta
    y = gamma +sqrt(q) * xi

    weight = gaussian_2d_xigamma_Tukey(xi, gamma, J_beta, delta_prime)
    prox = proximal_Tukey_modified_quad(y, sqrt(q)*xi, V, tau, c)
    f_out = (prox - sqrt(q)*xi) / V
    return  weight*f_out**2

@njit(error_model="numpy", fastmath=False)
def V_star_int_xigamma_Tukey(xi, gamma, q, m, V, delta, beta, tau, c):
    """Intégrande pour le calcul de V_hat_star dans les coordonnées (xi,gamma)"""
    eta = m**2 / q
    J_beta = beta*sqrt(eta) - sqrt(q) # Jacobien

    delta_prime = (1-eta)*beta**2 + delta
    y = gamma +sqrt(q) * xi

    weight = gaussian_2d_xigamma_Tukey(xi, gamma, J_beta, delta_prime)
    prox = proximal_Tukey_modified_quad(y, sqrt(q)*xi, V, tau, c)
    Dproximal = (1 + V * DDz_mod_tukey_loss_cubic(y, prox, tau, c)) ** (-1)

    return  weight*(Dproximal - 1)/ V

DEFAULT_N_STD = 7 # Nombre d'écarts-types pour l'intégration en u

def f_hat_xigamma_mod_Tukey_decorrelated_noise(
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
    integration_bound=DEFAULT_N_STD,
    epsabs=1e-12,
    epsrel=1e-8,
):

    mu_in, mu_out, J_beta_in, J_beta_out,delta_prime_in,delta_prime_out = mu(q, m, V, delta_in, delta_out, percentage, beta, tau)
    var_in = integration_bound *sqrt(mu_in)**(-1) # Nombre d'écarts-types pour l'intégration en xi
    var_out = integration_bound *sqrt(mu_out)**(-1) # Nombre d'écarts-types pour l'intégration en xi
    
    gamma_dom_in = [0,tau] #[0, integration_bound*sqrt(delta_prime_in+J_beta_in**2)]
    gamma_dom_out = [0,tau] #[0, integration_bound*sqrt(delta_prime_out+ J_beta_out**2)]
    
    def center_in(x):
        return J_beta_in*x/(delta_prime_in*mu_in)
    def center_out(x):
        return J_beta_out*x/(delta_prime_out*mu_out)

    int_value_m_hat_in = dblquad(
        m_star_int_xigamma_Tukey,
        gamma_dom_in[0],
        gamma_dom_in[1],
        lambda x: -var_in+center_in(x),
        lambda x: var_in +center_in(x),
        args=(q, m, V, delta_in, 1, tau, c),epsabs=epsabs, epsrel=epsrel
    )[0]

    int_value_m_hat_out = dblquad(
        m_star_int_xigamma_Tukey,
        gamma_dom_out[0],
        gamma_dom_out[1],
        lambda x: -var_out +center_out(x),
        lambda x: var_out +center_out(x),
        args=(q, m, V, delta_out, beta, tau, c),epsabs=epsabs, epsrel=epsrel
    )[0]
    m_hat = 2*alpha * ((1-percentage)*int_value_m_hat_in+ percentage*int_value_m_hat_out)
    
    int_value_q_hat_in = dblquad(
        q_star_int_xigamma_Tukey,
        gamma_dom_in[0],
        gamma_dom_in[1],
        lambda x: -var_in +center_in(x),
        lambda x: var_in +center_in(x),
        args=(q, m, V, delta_in, 1, tau, c),epsabs=epsabs, epsrel=epsrel
    )[0]

    int_value_q_hat_out = dblquad(
        q_star_int_xigamma_Tukey,
        gamma_dom_out[0],
        gamma_dom_out[1],
        lambda x: -var_out +center_out(x),
        lambda x: var_out  +center_out(x),
        args=(q, m, V, delta_out, beta, tau, c),epsabs=epsabs, epsrel=epsrel
    )[0]
    q_hat = 2*alpha * ((1-percentage)*int_value_q_hat_in+ percentage*int_value_q_hat_out)
    
    int_value_V_hat_in = dblquad(
        V_star_int_xigamma_Tukey,
        gamma_dom_in[0],
        gamma_dom_in[1],
        lambda x: -var_in +center_in(x),
        lambda x: var_in +center_in(x),
        args=(q, m, V, delta_in, 1, tau, c),epsabs=epsabs, epsrel=epsrel
    )[0]

    int_value_V_hat_out = dblquad(
        V_star_int_xigamma_Tukey,
        gamma_dom_out[0],
        gamma_dom_out[1],
        lambda x: -var_out +center_out(x),
        lambda x: var_out +center_out(x),
        args=(q, m, V, delta_out, beta, tau, c),epsabs=epsabs, epsrel=epsrel
    )[0]
    V_hat = -2*alpha * ((1-percentage)*int_value_V_hat_in+ percentage*int_value_V_hat_out)
    
    return m_hat, q_hat, V_hat

def m_star_int_fast(gamma, q, m, V, delta, beta, tau, c,):
    if beta == 0:
        return 0.0
    
    _, _, _,J_beta, _, delta_prime = mu(q, m, V, 0, delta, 0, beta, tau)
    sigma = delta_prime+J_beta**2
    y=0.0
    prox = y- proximal_Tukey_modified_quad(y, y-gamma, V, tau, c)
    f_out = (gamma - prox) / V
    return f_out*gaussian(gamma,0,sigma)*gamma*beta/sigma

def q_star_int_fast(gamma, q, m, V, delta, beta, tau, c):
    _, _, _, J_beta, _, delta_prime = mu(q, m, V, 0, delta, 0, beta, tau)
    sigma = delta_prime + J_beta**2

    y=0.0
    prox = y- proximal_Tukey_modified_quad(y, y-gamma, V, tau, c)
    f_out = (gamma - prox) / V
    
    return (f_out**2) * gaussian(gamma, 0, sigma)

def V_star_int_fast(gamma, q, m, V, delta, beta, tau, c):
    if abs(gamma) > tau : 
        return 0.0
    _, _, _, J_beta, _, delta_prime = mu(q, m, V, 0, delta, 0, beta, tau)
    sigma = delta_prime + J_beta**2

    y=0.0
    prox = y- proximal_Tukey_modified_quad(y, y-gamma, V, tau, c)
    prox_tilde = (prox/tau)**2
    Dprox = (1+V*(1-prox_tilde)*(1-5*prox_tilde))**(-1)

    return (1-Dprox)/V*gaussian(gamma, 0, sigma)


def f_hat_fast(
    m, q, V, alpha,
    delta_in, delta_out, percentage, beta,
    tau, c,
    integration_bound=DEFAULT_N_STD,
    epsabs=1e-12,
    epsrel=1e-8,
):
    """
    Calcule m_hat, q_hat, V_hat en utilisant les intégrandes 1D.
    """
    gamma_min_int, gamma_max_int = 0, tau

    # Contribution "inlier" (beta = 1.0, delta = delta_in)
    args_in_fast = (q, m, V, delta_in, 1.0, tau, c)
    int_m_in,_= quad(m_star_int_fast, gamma_min_int, gamma_max_int, args=args_in_fast, epsabs=epsabs, epsrel=epsrel)
    int_q_in,_ = quad(q_star_int_fast, gamma_min_int, gamma_max_int, args=args_in_fast, epsabs=epsabs, epsrel=epsrel)
    int_V_in,_ = quad(V_star_int_fast, gamma_min_int, gamma_max_int, args=args_in_fast, epsabs=epsabs, epsrel=epsrel)

    # Contribution "outlier" (beta = beta, delta = delta_out)
    args_out_fast = (q, m, V, delta_out, beta, tau, c)
    int_m_out, _ = quad(m_star_int_fast, gamma_min_int, gamma_max_int, args=args_out_fast, epsabs=epsabs, epsrel=epsrel)
    int_q_out, _ = quad(q_star_int_fast, gamma_min_int, gamma_max_int, args=args_out_fast, epsabs=epsabs, epsrel=epsrel)
    int_V_out, _ = quad(V_star_int_fast, gamma_min_int, gamma_max_int, args=args_out_fast, epsabs=epsabs, epsrel=epsrel)

    # Combinaison
    m_hat = 2*alpha * ((1-percentage)*int_m_in + percentage*int_m_out)
    q_hat = 2*alpha * ((1-percentage)*int_q_in + percentage*int_q_out)
    V_hat = 2*alpha * ((1-percentage)*int_V_in + percentage*int_V_out)

    return m_hat, q_hat, V_hat
