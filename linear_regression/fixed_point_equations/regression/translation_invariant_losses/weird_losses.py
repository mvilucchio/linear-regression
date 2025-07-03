# Matéo begins
from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad
from ....aux_functions.loss_functions import Dr_tukey_loss, DDr_tukey_loss
from ....aux_functions.weighted_output_chanels_TI import L_cal_multi_decorrelated_noise
from ...regularisation.Replica_symmetry_E1 import E1_RS_l2_reg

# ----- test with another loss function r -> exp(r^2/(2*tau^2))-----

@njit(error_model="numpy", fastmath=False)
def q_int_exp_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    Dr_exp = exp(r**2/(2*tau**2))*r/tau**2
    DDr_exp = exp(r**2/(2*tau**2))*(1/tau**2+ r**2/(tau**4))

    densities = L_cal_multi_decorrelated_noise(
        r+V*Dr_exp,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return Dr_exp**2 * (1+V*DDr_exp)*densities

@njit(error_model="numpy", fastmath=False)
def V_int_exp_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    Dr_exp = exp(r**2/(2*tau**2))*r/tau**2
    DDr_exp = exp(r**2/(2*tau**2))*(1/tau**2+ r**2/(tau**4))

    densities = L_cal_multi_decorrelated_noise(
        r+V*Dr_exp,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return DDr_exp*densities

@njit(error_model="numpy", fastmath=False)
def q_int_exp_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return q_int_exp_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def V_int_exp_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return V_int_exp_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

def f_hat_exp_decorrelated_noise_TI(m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, rho = 1.0, 
                                      q_int_loss_decorrelated_noise_x=q_int_exp_decorrelated_noise_TI_r,
                                      V_int_loss_decorrelated_noise_x=V_int_exp_decorrelated_noise_TI_r,
                                      integration_bound = 5
                                      ):
    
    z_0s= np.array([0.0, 0.0])
    betas = np.array([1.0, beta])
    sigma_sqs = np.array([Delta_in, Delta_out])
    proportions = np.array([1-percentage, percentage])

    qhat_in = 2 * alpha * quad(
        q_int_loss_decorrelated_noise_x,
        0,
        tau* integration_bound,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    qhat_out = 2 * alpha * quad(
        q_int_loss_decorrelated_noise_x,
        0,
        tau*integration_bound,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    Vhat_in = 2 * alpha * quad(
        V_int_loss_decorrelated_noise_x,
        0,
        tau* integration_bound,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    Vhat_out = 2 * alpha * quad(
        V_int_loss_decorrelated_noise_x,
        0,
        tau*integration_bound,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    m_hat_in = betas[0]*Vhat_in
    m_hat_out = betas[1]*Vhat_out

    return m_hat_in+m_hat_out, qhat_in+qhat_out, Vhat_in+Vhat_out

@njit(error_model="numpy", fastmath=False)
def E2_RS_exp_int_multi_decorrelated_noise_TI_r(
    r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0
):
    Dr_exp = exp(r**2/(2*tau**2))*r/tau**2
    DDr_exp = exp(r**2/(2*tau**2))*(1/tau**2+ r**2/(tau**4))

    densities = L_cal_multi_decorrelated_noise(
        r+ V * Dr_exp,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return DDr_exp**2 * (1+ V*DDr_exp)**(-1) * densities

@njit(error_model="numpy", fastmath=False)
def E2_RS_exp_int_decorrelated_noise_TI_r(
    r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i = 0
):
    return E2_RS_exp_int_multi_decorrelated_noise_TI_r(
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

def RS_exp_decorrelated_noise_TI_l2_reg(m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, reg_param, rho = 1.0, 
                                          f_hat_loss_decorrelated_noise=f_hat_exp_decorrelated_noise_TI,
                                          E2_RS_loss_int_decorrelated_noise_x=E2_RS_exp_int_decorrelated_noise_TI_r,
                                          integration_bound = 5,
                                          **integration_args,
                                          ):
    
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
        tau*integration_bound,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho,0)
    )[0]
    E2_out = quad(
        E2_RS_loss_int_decorrelated_noise_x,
        0,
        tau*integration_bound,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1)
    )[0]
    E2 = E2_in + E2_out
    return alpha * E1 * E2

@njit(error_model="numpy", fastmath=False)
def q_int_Tukey_multi_decorrelated_noise_TI_r_infty(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    Dr_Tukey = Dr_tukey_loss(r,tau)
    DDr_Tukey = DDr_tukey_loss(r,tau)

    densities = L_cal_multi_decorrelated_noise(
        r,
        m,
        q,
        V,
        z_0s,
        betas,
        sigma_sqs,
        proportions,
        rho
    )

    return Dr_Tukey**2 * densities

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_multi_decorrelated_noise_TI_r_infty(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    Dr_Tukey = Dr_tukey_loss(r,tau)
    DDr_Tukey = DDr_tukey_loss(r,tau)

    densities = L_cal_multi_decorrelated_noise(
        r,
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
def q_int_Tukey_decorrelated_noise_TI_r_infty(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return q_int_Tukey_multi_decorrelated_noise_TI_r_infty(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_decorrelated_noise_TI_r_infty(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    return V_int_Tukey_multi_decorrelated_noise_TI_r_infty(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

def f_hat_Tukey_decorrelated_noise_TI_infty(m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, rho = 1.0, 
                                      q_int_loss_decorrelated_noise_x=q_int_Tukey_decorrelated_noise_TI_r_infty,
                                      V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r_infty,
                                      ):
    
    z_0s= np.array([0.0, 0.0])
    betas = np.array([1.0, beta])
    sigma_sqs = np.array([Delta_in, Delta_out])
    proportions = np.array([1-percentage, percentage])

    qhat_in = 2 * quad(
        q_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    qhat_out = 2 * quad(
        q_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    Vhat_in = 2 * quad(
        V_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    Vhat_out = 2 * quad(
        V_int_loss_decorrelated_noise_x,
        0,
        tau,
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    m_hat_in = betas[0]*Vhat_in
    m_hat_out = betas[1]*Vhat_out

    return m_hat_in+m_hat_out, qhat_in+qhat_out, Vhat_in+Vhat_out

@njit(error_model="numpy", fastmath=False)
def f_L2_reg_infty(m_hat: float, q_hat: float, V_hat: float, reg_param: float) -> tuple:
    m = m_hat / V_hat 
    q = m** 2
    V = 1e-10
    return m, q, V

