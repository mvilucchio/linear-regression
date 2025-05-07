from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad
from linear_regression.aux_functions.moreau_proximals import proximal_Tukey_modified_quad
from linear_regression.aux_functions.misc import gaussian
from linear_regression.aux_functions.loss_functions import DDz_mod_tukey_loss_cubic
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (
    mu)
from traceback import print_exc

DEFAULT_N_STD = 7

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
