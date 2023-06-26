import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
)
from linear_regression.aux_functions.stability_functions import (
    stability_ridge,
    stability_l1_l2,
    stability_huber,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha = 1.000000000000000000e+02
reg_param = -4.0 # -7.960732770734782804e-01
a_hub = 1.0 # 9.648604577287680506e-01

N = 30

initial_point_m = np.linspace(0.0, 1000, N)
initial_point_q = np.linspace(0.0, 1000, N)
initial_point_sigma = np.linspace(0.0, 1000, N)

finals = np.empty((N * N * N, 3))
global_idx = 0
for idx, m in enumerate(initial_point_m):
    for jdx, q in enumerate(initial_point_q):
        for kdx, sigma in enumerate(initial_point_sigma):
            if global_idx % 1000 == 0:
                print(global_idx)

            finals[global_idx, :] = fixed_point_finder(
                var_func_L2,
                var_hat_func_Huber_decorrelated_noise,
                (m, q, sigma),
                {"reg_param": reg_param},
                {
                    "alpha" : alpha,
                    "delta_in": delta_in,
                    "delta_out": delta_out,
                    "percentage": percentage,
                    "beta": beta,
                    "a": a_hub
                },
            )
            global_idx += 1


# plot in 3D the points that are in final
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(finals[:, 0], finals[:, 1], finals[:, 2])
ax.set_xlabel("m")
ax.set_ylabel("q")
plt.show()