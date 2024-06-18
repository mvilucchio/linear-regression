from numba import njit
import numpy as np
from numpy import ndarray


@njit(error_model="numpy", fastmath=False)
def f_L2_reg(m_hat: float, q_hat: float, Σ_hat: float, reg_param: float) -> tuple:
    m = m_hat / (Σ_hat + reg_param)
    q = (m_hat**2 + q_hat) / (Σ_hat + reg_param) ** 2
    sigma = 1.0 / (Σ_hat + reg_param)
    return m, q, sigma


def f_L2_regularisation_adversarial(
    m_hat: float,
    q_hat: float,
    Σ_hat: float,
    P_hat: float,
    reg_param: float,
    Sigmadelta: ndarray,
    Sigmax: ndarray,
    Sigmatheta: ndarray,
) -> tuple:
    d, _ = Sigmax.shape
    H = reg_param + Σ_hat * Sigmax + P_hat * Sigmadelta
    H_inv = np.linalg.pinv(H)
    m = np.trace(m_hat * Sigmax @ Sigmatheta @ Sigmax @ H_inv) / d
    q = (
        np.trace(
            (m_hat**2 * Sigmax @ Sigmatheta @ Sigmax + q_hat * Sigmax) @ Sigmax @ H_inv @ H_inv
        )
        / d
    )
    sigma = np.trace(Sigmax @ H_inv) / d
    P = (
        np.trace(
            (m_hat**2 * Sigmax @ Sigmatheta @ Sigmax + q_hat * Sigmax) @ Sigmadelta @ H_inv @ H_inv
        )
        / d
    )
    return m, q, sigma, P
