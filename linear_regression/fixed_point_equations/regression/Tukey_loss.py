from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad, dblquad
from ...aux_functions.moreau_proximals import (proximal_Logistic_loss, Dω_proximal_Logistic_loss, proximal_Tukey_loss_TI, Ddelta_proximal_Tukey_loss_TI)
from ...aux_functions.misc import gaussian
from ...aux_functions.loss_functions import logistic_loss, DDz_logistic_loss, Dr_tukey_loss, DDr_tukey_loss
from ...aux_functions.likelihood_channel_functions import L_cal_multi_decorrelated_noise

BIG_NUMBER = 20


def m_int_Tukey_decorrelated_noise(
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
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return y * gaussian(ξ, 0, 1) * exp(-0.5 * η * ξ**2 / (1 - η)) * (proximal - sqrt(q) * ξ) / V


def q_int_Tukey_decorrelated_noise(
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
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    return (
        gaussian(ξ, 0, 1)
        * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η))))
        * (proximal - sqrt(q) * ξ) ** 2
        / V**2
    )


def V_int_Tukey_decorrelated_noise(
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
    proximal = proximal_Logistic_loss(y, sqrt(q) * ξ, V)
    Dproximal = (1 + V * DDz_logistic_loss(y, proximal)) ** (-1)
    return gaussian(ξ, 0, 1) * (1 + y * erf(sqrt(η) * ξ / sqrt(2 * (1 - η)))) * (Dproximal - 1) / V


# -----------------------------------


def f_hat_Tukey_decorrelated_noise(m, q, V, alpha, delta_in, delta_out, percentage, beta, a):
    domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

    int_value_m_hat = 0.0
    for y_val, domain in domains:
        int_value_m_hat += quad(
            m_int_Tukey_decorrelated_noise, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    m_hat = alpha * int_value_m_hat

    int_value_q_hat = 0.0
    for y_val, domain in domains:
        int_value_q_hat += quad(
            q_int_Tukey_decorrelated_noise, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    q_hat = alpha * int_value_q_hat

    int_value_V_hat = 0.0
    for y_val, domain in domains:
        int_value_V_hat += quad(
            V_int_Tukey_decorrelated_noise, domain[0], domain[1], args=(y_val, q, m, V)
        )[0]
    V_hat = -alpha * int_value_V_hat

    return m_hat, q_hat, V_hat

# Matéo begins

@njit(error_model="numpy", fastmath=False)
def q_int_Tukey_multi_decorrelated_noise_TI(delta, m,q,V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    prox = proximal_Tukey_loss_TI(delta, V, tau)
    densities = L_cal_multi_decorrelated_noise(
        delta,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )
    return (delta-prox)**2 / V**2 * densities

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_multi_decorrelated_noise_TI(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    Dprox = Ddelta_proximal_Tukey_loss_TI(delta, V, tau)
    densities = L_cal_multi_decorrelated_noise(
        delta,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )
    return (1- Dprox) / V * densities

@njit(error_model="numpy", fastmath=False)
def q_int_Tukey_decorrelated_noise_TI_delta(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return q_int_Tukey_multi_decorrelated_noise_TI(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_decorrelated_noise_TI_delta(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return V_int_Tukey_multi_decorrelated_noise_TI(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def q_int_Tukey_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    Dr_Tukey = Dr_tukey_loss(r,tau)
    DDr_Tukey = DDr_tukey_loss(r,tau)

    densities = L_cal_multi_decorrelated_noise(
        r+V*Dr_Tukey,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return Dr_Tukey**2 * (1+V*DDr_Tukey)*densities

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    Dr_Tukey = Dr_tukey_loss(r,tau)
    DDr_Tukey = DDr_tukey_loss(r,tau)

    densities = L_cal_multi_decorrelated_noise(
        r+V*Dr_Tukey,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return DDr_Tukey*densities

@njit(error_model="numpy", fastmath=False)
def q_int_Tukey_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return q_int_Tukey_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return V_int_Tukey_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

def f_hat_Tukey_decorrelated_noise_TI(m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, rho = 1.0, 
                                      q_int_loss_decorrelated_noise_x=q_int_Tukey_decorrelated_noise_TI_r,
                                      V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r,
                                      ):
    
    z_0s= np.array([0.0, 0.0])
    betas = np.array([1.0, beta])
    sigma_sqs = np.array([Delta_in, Delta_out])
    proportions = np.array([1-percentage, percentage])

    qhat_in = 2 * alpha * quad(
        q_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    qhat_out = 2 * alpha * quad(
        q_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    Vhat_in = 2 * alpha * quad(
        V_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    Vhat_out = 2 * alpha * quad(
        V_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    m_hat_in = betas[0]*Vhat_in
    m_hat_out = betas[1]*Vhat_out

    return m_hat_in+m_hat_out, qhat_in+qhat_out, Vhat_in+Vhat_out

@njit(error_model="numpy", fastmath=False)
def E1_RS_l2_reg(reg_param, V_hat):
    return pow( reg_param + V_hat , -2)

@njit(error_model="numpy", fastmath=False)
def E2_RS_int_multi_decorrelated_noise_TI(
    delta, dprox, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0
):
    densities = L_cal_multi_decorrelated_noise(
        delta,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return ((1-dprox)/V)**2 * densities

@njit(error_model="numpy", fastmath=False)
def E2_RS_int_decorrelated_noise_TI(
    delta, dprox, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i =0
):
    return E2_RS_int_multi_decorrelated_noise_TI(
        delta,
        dprox,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        tau,
        rho
    )[i]

@njit(error_model="numpy", fastmath=False)
def E2_RS_Tukey_int_decorrelated_noise_TI(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i = 0):
    dprox = Ddelta_proximal_Tukey_loss_TI(delta, V, tau)

    return E2_RS_int_decorrelated_noise_TI(
        delta,
        dprox,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        tau,
        rho,
        i
    )

@njit(error_model="numpy", fastmath=False)
def E2_RS_Tukey_int_multi_decorrelated_noise_TI_r(
    r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0
):
    Dr_Tukey = Dr_tukey_loss(r, tau)
    DDr_Tukey = DDr_tukey_loss(r, tau)

    densities = L_cal_multi_decorrelated_noise(
        r+ V * Dr_Tukey,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return DDr_Tukey**2 * (1+ V*DDr_Tukey)**(-1) * densities

@njit(error_model="numpy", fastmath=False)
def E2_RS_Tukey_int_decorrelated_noise_TI_r(
    r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i = 0
):
    return E2_RS_Tukey_int_multi_decorrelated_noise_TI_r(
        r,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        tau,
        rho
    )[i]

def RS_Tukey_decorrelated_noise_TI_l2_reg(m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, reg_param, rho = 1.0, 
                                          f_hat_loss_decorrelated_noise=f_hat_Tukey_decorrelated_noise_TI,
                                          E2_RS_loss_int_decorrelated_noise_x=E2_RS_Tukey_int_decorrelated_noise_TI_r,
                                          **integration_args):
    
    z_0s= np.array([0.0, 0.0])
    betas = np.array([1.0, beta])
    sigma_sqs = np.array([Delta_in, Delta_out])
    proportions = np.array([1-percentage, percentage])

    m_hat, q_hat, V_hat = f_hat_loss_decorrelated_noise(
        m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, rho
    )
    E1 = E1_RS_l2_reg(reg_param, V_hat)
    E2_in = quad(
        E2_RS_loss_int_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho,0)
    )[0]
    E2_out = quad(
        E2_RS_loss_int_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1)
    )[0]
    E2 = E2_in + E2_out
    return alpha * E1 * E2
        
# Matéo ends
