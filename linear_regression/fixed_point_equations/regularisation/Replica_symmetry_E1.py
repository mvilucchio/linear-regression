# Matéo begins
# This file contains computations of E1 for different regularization functions.

from numba import njit

@njit(error_model="numpy", fastmath=False)
def E1_RS_l2_reg(reg_param, V_hat):
    return pow( reg_param + V_hat , -2)
# Matéo ends
