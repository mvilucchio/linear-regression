# Matéo begins
# This file contains computations of the E2 integral for translation invariant losses (TI) with decorrelated noise models.

from numba import njit
from ....aux_functions.weighted_output_chanels_TI import L_cal_multi_decorrelated_noise
from ...regularisation.Replica_symmetry_E1 import E1_RS_l2_reg

# This either
@njit(error_model="numpy", fastmath=False)
def E2_RS_int_multi_decorrelated_noise_TI(
    delta, dprox, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0
):
    """ Returns the list of integrands (functions of delta) of the E2 integral for each of the weighted Gaussian components in the mixture."""
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

# This should not be here either
@njit(error_model="numpy", fastmath=False)
def E2_RS_int_decorrelated_noise_TI(
    delta, dprox, m, q, V, z_0s, betas, sigma_sqs, proportions, tau, rho=1.0, i =0
):
    """ Returns the i-th integrand (function of delta) of the E2 integral for the i-th weighted Gaussian component in the mixture."""
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

# Matéo ends
