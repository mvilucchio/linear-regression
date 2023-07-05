import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_L2_reg,
    f_hat_L1_decorrelated_noise,
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

N = 50

initial_point_m = np.linspace(0.0, 1000, N)
initial_point_q = np.linspace(0.0, 1000, N)
initial_point_sigma = np.linspace(0.0, 1000, N)

finals = np.empty((N * N * N, 3))
global_idx = 0
for idx, m in enumerate(initial_point_m):
    for jdx, q in enumerate(initial_point_q):
        for kdx, sigma in enumerate(initial_point_sigma):
            if global_idx % 100 == 0:
                print(global_idx)
            finals[global_idx, :] = fixed_point_finder(
                f_L2_reg,
                f_hat_L1_decorrelated_noise,
                (m, q, sigma),
                {"reg_param": 1.0},
                {
                    "alpha" : 15,
                    "delta_in": delta_in,
                    "delta_out": delta_out,
                    "percentage": percentage,
                    "beta": beta,
                    "a": 1.0
                },
            )
            global_idx += 1


# plot in 3D the points that are in final
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(finals[:, 0], finals[:, 1], finals[:, 2])
ax.set_xlabel("m")
ax.set_ylabel("q")
ax.set_zlabel(r"\Sigma")
plt.show()