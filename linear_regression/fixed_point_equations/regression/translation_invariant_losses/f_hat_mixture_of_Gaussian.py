# Matéo begins
# This file contains the implementation of f_hat for even losses with translation invariance, integrated over 0, tau. 
# This needs to be adapted to fit general translation invariant losses.
from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad

def f_hat_decorrelated_noise_TI(m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, rho = 1.0, 
                                      m_int_loss_decorrelated_noise_x=None,
                                      q_int_loss_decorrelated_noise_x=None,
                                      V_int_loss_decorrelated_noise_x=None,
                                      ):
    """ Returns (m_hat, q_hat, Vhat) for the loss function (None by default) with translation invariance and EVEN parity, using the inlier-outlier model.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI, the integrand computed is a function of delta.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI_r and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r, the integrand computed is a function of r (faster, more robust)."""
    
    z_0s= np.array([0.0, 0.0])
    betas = np.array([1.0, beta])
    sigma_sqs = np.array([Delta_in, Delta_out])
    proportions = np.array([1-percentage, percentage])

    qhat_in = 2 * alpha * quad(
        q_int_loss_decorrelated_noise_x,
        0,                                          # Beware of the 2 because of even parity, and the bounds specific to Tukey loss.
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

def f_hat_multi_decorrelated_noise_TI(m, q, V, alpha, z_0s, betas, sigma_sqs, proportions, tau, rho = 1.0,
                                      m_int_loss_decorrelated_noise_x=None,
                                      q_int_loss_decorrelated_noise_x= None,
                                      V_int_loss_decorrelated_noise_x=None,
                                      ):
    """ Returns (m_hat, q_hat, Vhat) for the loss function (None by default) with translation invariance and EVEN parity, using the decorrelated mixture of Gaussians model.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI, the integrand computed is a function of delta.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI_r and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r, the integrand computed is a function of r (faster, more robust)."""
    
    nb_components = len(proportions)

    if len(betas) != nb_components or len(sigma_sqs) != nb_components or len(z_0s) != nb_components:
        raise ValueError("z_0s, betas, sigma_sqs and proportions must have the same length.")
    
    if not np.isclose(np.sum(proportions), 1.0) or not all(proportions >= 0):
        raise ValueError("proportions must be non-negative and sum to 1.")
    
    if not all(sigma_sqs > 0):
        raise ValueError("sigma_sqs must be positive.")
    
    # Initialize the output variables
    mhat, qhat, Vhat = [0.0]* nb_components, [0.0]* nb_components, [0.0]* nb_components
    
    for component in range(nb_components):
        qhat[component] = 2 * alpha * quad(
            q_int_loss_decorrelated_noise_x,
            0,
            tau,
            args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, component),
        )[0]

        Vhat[component] = Vhat + 2 * alpha * quad(
            V_int_loss_decorrelated_noise_x,
            0,
            tau,
            args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, component),
        )[0]

        mhat[component] = betas[component] * Vhat[component]

    return sum(mhat), sum(qhat), sum(Vhat)
