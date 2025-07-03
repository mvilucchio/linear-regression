# Matéo begins
# This file contains the implementation of the integrands state equations for the Tukey loss function, using translation invariance.
# The equations in the r variable are also implemented, without the need to compute the proximal operator.
# The output channel is a mixture of Gaussians following the inlier-outlier model. As such, m_hat=\beta * V_hat for each individual gaussian.

from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad
from ....aux_functions.moreau_proximals import proximal_Tukey_loss_TI, Ddelta_proximal_Tukey_loss_TI
from ....aux_functions.loss_functions import Dr_tukey_loss, DDr_tukey_loss
from ....aux_functions.weighted_output_chanels_TI import L_cal_multi_decorrelated_noise
from ...regularisation.Replica_symmetry_E1 import E1_RS_l2_reg
from .Replica_symmetry_E2 import E2_RS_int_multi_decorrelated_noise_TI, E2_RS_int_decorrelated_noise_TI
from .f_hat_mixture_of_Gaussian import f_hat_decorrelated_noise_TI, f_hat_multi_decorrelated_noise_TI

@njit(error_model="numpy", fastmath=False)
def q_int_Tukey_multi_decorrelated_noise_TI(delta, m,q,V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    """ Returns the list of integrands (functions of delta) of the q_hat integral for each of the weighted Gaussian components in the mixture."""
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
    """ Returns the list of integrands (functions of delta) of the V_hat integral for each of the weighted Gaussian components in the mixture."""
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
    """ Returns the i-th integrand (function of delta) of the q_hat integral for the i-th weighted Gaussian component in the mixture."""
    return q_int_Tukey_multi_decorrelated_noise_TI(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_decorrelated_noise_TI_delta(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    """ Returns the i-th integrand (function of delta) of the V_hat integral for the i-th weighted Gaussian component in the mixture."""
    return V_int_Tukey_multi_decorrelated_noise_TI(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def q_int_Tukey_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0):
    """ Returns the list of integrands (functions of r) of the q_hat integral for each of the weighted Gaussian components in the mixture."""
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
    """ Returns the list of integrands (functions of r) of the V_hat integral for each of the weighted Gaussian components in the mixture."""
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
    """ Returns the i-th integrand (function of r) of the q_hat integral for the i-th weighted Gaussian component in the mixture."""
    return q_int_Tukey_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def V_int_Tukey_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i=0):
    """ Returns the i-th integrand (function of r) of the V_hat integral for the i-th weighted Gaussian component in the mixture."""
    return V_int_Tukey_multi_decorrelated_noise_TI_r(r, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho)[i]

@njit(error_model="numpy", fastmath=False)
def E2_RS_Tukey_int_decorrelated_noise_TI(delta, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i = 0):
    """ Returns the i-th integrand (function of delta) of the E2 integral for the i-th weighted Gaussian component in the mixture, using the Tukey loss function."""
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
    """ Returns the list of integrands (functions of r) of the E2 integral for each of the weighted Gaussian components in the mixture, using the Tukey loss function."""
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
    """ Returns the i-th integrand (function of r) of the E2 integral for the i-th weighted Gaussian component in the mixture, using the Tukey loss function."""
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
                                          f_hat_loss_decorrelated_noise=f_hat_decorrelated_noise_TI,
                                          m_int_loss_decorrelated_noise_x=None,
                                          q_int_loss_decorrelated_noise_x=q_int_Tukey_decorrelated_noise_TI_r,
                                          V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r,
                                          E2_RS_loss_int_decorrelated_noise_x=E2_RS_Tukey_int_decorrelated_noise_TI_r,
                                          **integration_args):
    """ Returns the RS value for the Tukey loss function with translation invariance, using the inlier-outlier model and l2 regularization.
    Depending on *_int_loss_decorrelated_noise, the integrand computed is a function of delta or r."""
    
    z_0s= np.array([0.0, 0.0])
    betas = np.array([1.0, beta])
    sigma_sqs = np.array([Delta_in, Delta_out])
    proportions = np.array([1-percentage, percentage])

    m_hat, q_hat, V_hat = f_hat_loss_decorrelated_noise(
        m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, rho,
        m_int_loss_decorrelated_noise_x=m_int_loss_decorrelated_noise_x,
        q_int_loss_decorrelated_noise_x=q_int_loss_decorrelated_noise_x,
        V_int_loss_decorrelated_noise_x=V_int_loss_decorrelated_noise_x,
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

def RS_Tukey_multi_decorrelated_noise_TI_l2_reg(m, q, V, alpha, z_0s, betas, sigma_sqs, proportions, tau, reg_param, rho = 1.0,
                                            f_hat_loss_decorrelated_noise=f_hat_multi_decorrelated_noise_TI,
                                            m_int_loss_decorrelated_noise_x=None,
                                            q_int_loss_decorrelated_noise_x=q_int_Tukey_decorrelated_noise_TI_r, 
                                            V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r,
                                            E2_RS_loss_int_decorrelated_noise_x=E2_RS_Tukey_int_decorrelated_noise_TI_r,
                                            **integration_args):
    """ Returns the RS value for the Tukey loss function with translation invariance, using the decorrelated mixture of Gaussians model and l2 regularization.
    Depending on *_int_loss_decorrelated_noise, the integrand computed is a function of delta or r."""

    nb_components = len(proportions)

    if len(betas) != nb_components or len(sigma_sqs) != nb_components or len(z_0s) != nb_components:
        raise ValueError("z_0s, betas, sigma_sqs and proportions must have the same length.")
    if not np.isclose(np.sum(proportions), 1.0) or not all(proportions >= 0):
        raise ValueError("proportions must be non-negative and sum to 1.")
    if not all(sigma_sqs > 0):
        raise ValueError("sigma_sqs must be positive.")
    
    mhat, qhat, Vhat = f_hat_loss_decorrelated_noise(
        m, q, V, alpha, z_0s, betas, sigma_sqs, proportions, tau, rho,
        m_int_loss_decorrelated_noise_x=m_int_loss_decorrelated_noise_x,
        q_int_loss_decorrelated_noise_x=q_int_loss_decorrelated_noise_x,
        V_int_loss_decorrelated_noise_x=V_int_loss_decorrelated_noise_x
    )

    E1 = E1_RS_l2_reg(reg_param, Vhat)

    # Initialize the output variable
    E2 = 0.0
    for component in range(nb_components):
        E2 += quad(
            E2_RS_loss_int_decorrelated_noise_x,
            0,
            tau,
            args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, component)
        )[0]
    return alpha * E1 * E2

# Matéo ends
