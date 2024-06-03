from numba import njit


@njit(error_model="numpy", fastmath=False)
def f_L2_reg(m_hat: float, q_hat: float, Σ_hat: float, reg_param: float) -> tuple:
    m = m_hat / (Σ_hat + reg_param)
    q = (m_hat**2 + q_hat) / (Σ_hat + reg_param) ** 2
    sigma = 1.0 / (Σ_hat + reg_param)
    return m, q, sigma
