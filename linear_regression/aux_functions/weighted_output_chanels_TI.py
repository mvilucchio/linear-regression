# ----------------------------------- Matéo begins
# This file contains delta densities (named L_cal) for different output channels with translation invariant losses (TI)

# In the case of a mixture of decorrelated Gaussians, the densities are weighted by the proportions of each Gaussian in the mixture.
# They are returned as a list of densities to split the integration for better performance.
# Each Gaussian is defined by its mean (beta * z + z_0), and variance (sigma_sq). The resulting L_cal is a Gaussian of mean z_0 and variance sigma_delta_sq = sigma_sq + q + beta**2 - 2* m* beta /sqrt(rho).

from math import exp, sqrt, pow, erf, pi, log
from ..aux_functions.misc import gaussian
import numpy as np
from numba import njit

@njit(error_model="numpy", fastmath=False)
def L_cal_single_noise(delta: float, m,q,V,rho =1.0,z_0 = 0.0, beta=0.0, sigma_sq = 1.0):
    """ Calculates the density of the random variable delta for a single noise model."""
    eta = m**2 / (rho*q)
    sigma_delta_sq = sigma_sq + q+ beta**2 - 2* m* beta /sqrt(rho)
    exponent = - (delta - z_0)**2 / (2 * sigma_delta_sq)
    denum = sqrt(2 * pi * sigma_delta_sq)
    return exp(exponent) /denum

@njit(error_model="numpy", fastmath=False)
def L_cal_multi_decorrelated_noise(
    delta: float,
    m: float,
    q: float,
    V: float,
    z_0s : np.ndarray,
    betas : np.ndarray,
    sigma_sqs : np.ndarray,
    proportions : np.ndarray,
    rho : float = 1.0
):
    """ Calculates the list of weighted densities of the random variable delta for multiple decorrelated noise models."""

    eta = m**2 / (rho * q)
    sigma_delta_sqs = sigma_sqs + q + betas**2 - 2 * m * betas / sqrt(rho)
    
    exponent = -((delta - z_0s) ** 2) / (2 * sigma_delta_sqs)
    denum = np.sqrt(2 * pi * sigma_delta_sqs)
    densities = np.exp(exponent) / denum
    
    return proportions* densities

# ----------------------------------- Matéo ends
