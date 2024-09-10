from numba import njit
import numpy as np
from numpy import ndarray, trace
from numpy.linalg import pinv


@njit(error_model="numpy", fastmath=False)
def f_L2_reg(m_hat: float, q_hat: float, V_hat: float, reg_param: float) -> tuple:
    m = m_hat / (V_hat + reg_param)
    q = (m_hat**2 + q_hat) / (V_hat + reg_param) ** 2
    V = 1.0 / (V_hat + reg_param)
    return m, q, V


# @njit(error_model="numpy", fastmath=False)
def f_L2_regularisation_covariate(
    m_hat: float,
    q_hat: float,
    V_hat: float,
    reg_param: float,
    Σx: ndarray,
    Σθ: ndarray,
    Σw: ndarray,
) -> tuple:
    d, _ = Σx.shape
    H_inv = pinv(reg_param * Σw + V_hat * Σx)
    m = m_hat * trace(Σx @ Σθ @ Σx @ H_inv) / d
    q = trace((m_hat**2 * Σx @ Σθ @ Σx + q_hat * Σx) @ Σx @ H_inv @ H_inv) / d
    V = trace(Σx @ H_inv) / d
    return m, q, V


@njit(error_model="numpy", fastmath=False)
def f_L2_regularisation_adversarial(
    m_hat: float,
    q_hat: float,
    V_hat: float,
    P_hat: float,
    reg_param: float,
    Σx: ndarray,
    Σθ: ndarray,
    Σδ: ndarray,
) -> tuple:
    d, _ = Σx.shape
    H_inv = np.linalg.pinv(reg_param + V_hat * Σx + P_hat * Σδ)
    m = np.trace(m_hat * Σx @ Σθ @ Σx @ H_inv) / d
    q = np.trace((m_hat**2 * Σx @ Σθ @ Σx + q_hat * Σx) @ Σx @ H_inv @ H_inv) / d
    V = np.trace(Σx @ H_inv) / d
    P = np.trace((m_hat**2 * Σx @ Σθ @ Σx + q_hat * Σx) @ Σδ @ H_inv @ H_inv) / d
    return m, q, V, P
