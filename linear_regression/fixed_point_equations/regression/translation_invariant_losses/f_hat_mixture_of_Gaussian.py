# Matéo begins
# This file contains the implementation of f_hat for a mixture of Gaussian distributions with decorrelated noise, using translation invariance.
from numba import njit
import numpy as np
from math import exp, erf, sqrt, pi
from scipy.integrate import quad

def f_hat_decorrelated_noise_TI(m, q, V, alpha, Delta_in, Delta_out, percentage, beta, tau, rho = 1.0, 
                                      q_int_loss_decorrelated_noise_x=None,
                                      V_int_loss_decorrelated_noise_x=None,
                                      even_loss=False,
                                      qbounds_in= [None,None],
                                      qbounds_out= [None,None],
                                      Vbounds_in= [None,None],
                                      Vbounds_out= [None,None],
                                      ):
    """ Returns (m_hat, q_hat, Vhat) for the loss function (None by default) with translation invariance, using the inlier-outlier model.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI, the integrand computed is a function of delta.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI_r and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r, the integrand computed is a function of r (faster, more robust).
    If even_loss is True, the loss function is assumed to be even, and the lower bound is set to 0.
    If qbounds_in, qbounds_out, Vbounds_in, Vbounds_out are provided, they must be lists of two elements (lower and upper bounds) for the inlier and outlier components.
    Otherwise, they are set to [-tau, tau], or [0, tau] if even_loss is True."""
    
    z_0s= np.array([0.0, 0.0])
    betas = np.array([1.0, beta])
    sigma_sqs = np.array([Delta_in, Delta_out])
    proportions = np.array([1-percentage, percentage])

    even_multiplier = 1.0
    if even_loss:
        even_multiplier = 2.0
        qbounds_in[0] = 0.0
        Vbounds_in[0] = 0.0
        qbounds_out[0] = 0.0
        Vbounds_out[0] = 0.0
        
    if qbounds_in[0] is None:
        qbounds_in[0] = -tau
    if Vbounds_in[0] is None:
        Vbounds_in[0] = -tau
    if qbounds_out[0] is None:
        qbounds_out[0] = -tau
    if Vbounds_out[0] is None:
        Vbounds_out[0] = -tau
    if qbounds_in[1] is None:
        qbounds_in[1] = tau
    if Vbounds_in[1] is None:
        Vbounds_in[1] = tau
    if qbounds_out[1] is None:
        qbounds_out[1] = tau
    if Vbounds_out[1] is None:
        Vbounds_out[1] = tau

    qhat_in = even_multiplier * alpha * quad(
        q_int_loss_decorrelated_noise_x,
        qbounds_in[0],
        qbounds_in[1],
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    qhat_out = even_multiplier * alpha * quad(
        q_int_loss_decorrelated_noise_x,
        qbounds_out[0],
        qbounds_out[1],
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    Vhat_in = even_multiplier * alpha * quad(
        V_int_loss_decorrelated_noise_x,
        Vbounds_in[0],
        Vbounds_in[1],
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 0),
    )[0]

    Vhat_out = even_multiplier * alpha * quad(
        V_int_loss_decorrelated_noise_x,
        Vbounds_out[0],
        Vbounds_out[1],
        args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, 1),
    )[0]

    m_hat_in = betas[0]*Vhat_in
    m_hat_out = betas[1]*Vhat_out

    return m_hat_in+m_hat_out, qhat_in+qhat_out, Vhat_in+Vhat_out

def f_hat_multi_decorrelated_noise_TI(m, q, V, alpha, z_0s, betas, sigma_sqs, proportions, tau, rho = 1.0,
                                      q_int_loss_decorrelated_noise_x= None,
                                      V_int_loss_decorrelated_noise_x=None,
                                        even_loss=False,
                                        qbounds = None,
                                        Vbounds = None,
                                      ):
    """ Returns (m_hat, q_hat, Vhat) for the loss function (None by default) with translation invariance and EVEN parity, using the decorrelated mixture of Gaussians model.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI, the integrand computed is a function of delta.
    If q_int_loss_decorrelated_noise_x= q_int_Tukey_decorrelated_noise_TI_r and V_int_loss_decorrelated_noise_x=V_int_Tukey_decorrelated_noise_TI_r, the integrand computed is a function of r (faster, more robust).
    If even_loss is True, the loss function is assumed to be even, and the lower bound is set to 0.
    If qbounds or Vbounds are provided, they must be lists of lists of two elements (lower and upper bounds) for each component of the Gaussian mixture.
    Otherwise, they are set to [0, tau] for each component."""
    
    nb_components = len(proportions)

    if len(betas) != nb_components or len(sigma_sqs) != nb_components or len(z_0s) != nb_components:
        raise ValueError("z_0s, betas, sigma_sqs and proportions must have the same length.")
    
    if not np.isclose(np.sum(proportions), 1.0) or not all(proportions >= 0):
        raise ValueError("proportions must be non-negative and sum to 1.")
    
    if not all(sigma_sqs > 0):
        raise ValueError("sigma_sqs must be positive.")
    
    if qbounds is not None or Vbounds is not None:
        if not len(qbounds) == nb_components or not len(Vbounds) == nb_components or not all(len(qbounds[i]) == 2 for i in range(nb_components)) or not all(len(Vbounds[i]) == 2 for i in range(nb_components)):
            raise ValueError("qbounds and Vbounds must have the same length as the number of components of the Gaussian mixture. Each element must be a list of two elements (lower and upper bounds).")
    
    if qbounds is None:
        qbounds = [[0, tau]] * nb_components

    if Vbounds is None:
        Vbounds = [[0, tau]] * nb_components

    even_multiplier = 1.0
    if even_loss:
        even_multiplier = 2.0
        for i in range(nb_components):
                qbounds[i][0] = 0.0
                Vbounds[i][0] = 0.0
    
    # Initialize the output variables
    mhat, qhat, Vhat = [0.0]* nb_components, [0.0]* nb_components, [0.0]* nb_components
    
    for component in range(nb_components):
        qhat[component] = even_multiplier * alpha * quad(
            q_int_loss_decorrelated_noise_x,
            qbounds[component][0],
            qbounds[component][1],
            args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, component),
        )[0]

        Vhat[component] = even_multiplier * alpha * quad(
            V_int_loss_decorrelated_noise_x,
            Vbounds[component][0],
            Vbounds[component][1],
            args=(m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho, component),
        )[0]

        mhat[component] = betas[component] * Vhat[component]

    return sum(mhat), sum(qhat), sum(Vhat)
