from numba import njit
from math import sqrt


@njit(error_model="numpy", fastmath=False)
def f_projection_denoising(m_hat: float, q_hat: float, Σ_hat: float, q_fixed: float) -> tuple:
    m = m_hat * sqrt(q_fixed) / sqrt(m_hat**2 + q_hat)
    sigma = sqrt(q_fixed / (q_hat + m_hat**2))
    return m, q_fixed, sigma


# def f_projection_sphere_denoising(m_hat, q_hat, Σ_hat, q_sphere):
#     m = m_hat * sqrt(q_sphere) / sqrt(m_hat**2 + q_hat)
#     sigma = sqrt(q_sphere / (q_hat + m_hat**2))
#     return m, q_sphere, sigma
