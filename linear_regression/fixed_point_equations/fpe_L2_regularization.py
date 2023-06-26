from numba import njit


@njit(error_model="numpy", fastmath=False)
def var_func_L2(m_hat, q_hat, sigma_hat, reg_param):
    m = m_hat / (sigma_hat + reg_param)
    q = (m_hat**2 + q_hat) / (sigma_hat + reg_param) ** 2
    sigma = 1.0 / (sigma_hat + reg_param)
    return m, q, sigma
