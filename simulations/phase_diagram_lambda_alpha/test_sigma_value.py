import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
)
import numpy as np
from linear_regression.aux_functions.stability_functions import (
    stability_ridge,
    stability_l1_l2,
    stability_huber,
)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
a_hub = 0.2

(alphas, reg_params), (sigmas, qs, ms) = alsw.sweep_alpha_descend_lambda(
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
    0.1,
    10,
    1000,
    -2,
    0.5,
    750,
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": a_hub,
    },
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[list(), list(), list()],
    initial_cond_fpe=(0.6, 0.01, 0.9),
)

# alphas_ls, last_reg_param_stable = alsw.sweep_alpha_minimal_stable_reg_param(
#     var_func_L2,
#     var_hat_func_L2_decorrelated_noise,
#     0.01,
#     10,
#     100,
#     {"reg_param": 3.0},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#     },
#     bounds_reg_param_search=(-100.0, 0.01),
#     points_per_run=10000,
# )

AA, RR = np.meshgrid(alphas, reg_params)

print(AA.shape, np.min(AA), alphas.shape, RR.shape, np.min(RR), reg_params.shape, sigmas.shape)

fig, ax = plt.subplots()
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\lambda$")
ax.set_title("Stability Huber a={:.1f}".format(a_hub))
ax.set_xscale("log")

stab = np.empty_like(sigmas)
for idx in range(sigmas.shape[0]):
    for jdx in range(sigmas.shape[1]):
        stab[idx, jdx] = stability_huber(
            ms[idx, jdx],
            qs[idx, jdx],
            sigmas[idx, jdx],
            alphas[jdx],
            reg_params[idx],
            delta_in,
            delta_out,
            percentage,
            beta, 
            a_hub,
        )

        if stab[idx, jdx] <= 0.0:
            stab[idx, jdx] = np.nan

        if stab[idx, jdx] > 10.0:
            stab[idx, jdx] = np.nan

# print(stab.amin(), stab.amax())

ax.contourf(AA, RR, stab, cmap="jet")
fig.colorbar(ax.contourf(AA, RR, stab, 200, cmap="jet"))

ax.autoscale(False)
ax.plot(alphas, -((1 - np.sqrt(alphas)) ** 2), zorder=1, color="tab:blue", label="condition MP")
# ax.plot(alphas_ls, last_reg_param_stable, zorder=1, color="tab:green", label="last lambda stable fixed point")
ax.grid(which="both")
ax.legend()
plt.show()
