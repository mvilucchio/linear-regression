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

# Replicon condition integrand --------------------------

@njit(error_model="numpy", fastmath=False)
def RS_int_mod_Tukey_decorrelated_noise(
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
        * (Dproximal - 1)**2
        / V**2
    )

# -----------------------------------

DEFAULT_INTEGRATION_BOUND = 15

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
    integration_bound=DEFAULT_INTEGRATION_BOUND,
    integration_epsabs=1.49e-8,
    integration_epsrel=1.49e-8,
):
    #y_dom = [-BIG_NUMBER, BIG_NUMBER]
    #xi_dom = [-BIG_NUMBER, BIG_NUMBER]
    y_dom = [-integration_bound, integration_bound]
    xi_dom = [-integration_bound, integration_bound]
    try:
         int_value_m_hat, abserr_m = dblquad(
            m_int_mod_Tukey_decorrelated_noise,
            xi_dom[0], xi_dom[1],
            lambda x: y_dom[0], lambda x: y_dom[1],
            args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),
            epsabs=integration_epsabs, # Tolérance absolue
            epsrel=integration_epsrel # Tolérance relative
         )
    except Exception as e:
         print(f"\nErreur dans dblquad (m_hat): {e}")
         int_value_m_hat = np.nan # Retourner NaN en cas d'échec

    try:
         int_value_q_hat, abserr_q = dblquad(
             q_int_mod_Tukey_decorrelated_noise,
             xi_dom[0], xi_dom[1],
             lambda x: y_dom[0], lambda x: y_dom[1],
             args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),
             epsabs=integration_epsabs,
             epsrel=integration_epsrel
         )
    except Exception as e:
         print(f"\nErreur dans dblquad (q_hat): {e}")
         int_value_q_hat = np.nan

    try:
         int_value_V_hat, abserr_V = dblquad(
             V_int_mod_Tukey_decorrelated_noise,
             xi_dom[0], xi_dom[1],
             lambda x: y_dom[0], lambda x: y_dom[1],
             args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),
             epsabs=integration_epsabs,
             epsrel=integration_epsrel
         )
    except Exception as e:
         print(f"\nErreur dans dblquad (V_hat): {e}")
         int_value_V_hat = np.nan

    # Gérer les NaN potentiels avant de continuer
    if np.isnan(int_value_m_hat) or np.isnan(int_value_q_hat) or np.isnan(int_value_V_hat):
        print("\nAttention : Échec d'au moins une intégration, retour de NaN pour m_hat, q_hat, V_hat")
        return np.nan, np.nan, np.nan

    m_hat = alpha * int_value_m_hat
    q_hat = alpha * int_value_q_hat
    V_hat = -alpha * int_value_V_hat

    # Vérifier si les résultats sont valides avant de retourner
    if not (np.isfinite(m_hat) and np.isfinite(q_hat) and np.isfinite(V_hat)):
        print(f"\nAttention : Valeurs non finies pour m_hat={m_hat}, q_hat={q_hat}, V_hat={V_hat}")
        return np.nan, np.nan, np.nan

    return m_hat, q_hat, V_hat    

    """ int_value_m_hat = dblquad(
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

    return m_hat, q_hat, V_hat """

def RS_alpha_E2_mod_Tukey_decorrelated_noise(
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
    integration_bound=DEFAULT_INTEGRATION_BOUND,
    integration_epsabs=1.49e-8,
    integration_epsrel=1.49e-8,
):
    y_dom = [-integration_bound, integration_bound]
    xi_dom = [-integration_bound, integration_bound]
    
    try:
         int_value_RS, abserr_RS = dblquad(
             RS_int_mod_Tukey_decorrelated_noise,
             xi_dom[0], xi_dom[1],
             lambda x: y_dom[0], lambda x: y_dom[1],
             args=(q, m, V, delta_in, delta_out, percentage, beta, tau, c),
             epsabs=integration_epsabs,
             epsrel=integration_epsrel
         )
    except Exception as e:
         print(f"\nErreur dans dblquad (RS): {e}")
         int_value_RS = np.nan

    # Gérer les NaN potentiels avant de continuer
    if np.isnan(int_value_RS):
        print("\nAttention : Échec d'au moins une intégration, retour de NaN pour RS")
        return np.nan

    RS_alpha_E2 = alpha * int_value_RS

    return RS_alpha_E2
