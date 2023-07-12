import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np
from linear_regression.aux_functions.stability_functions import (
    stability_L2_decorrelated_regress,
    stability_L1_decorrelated_regress,
    stability_Huber_decorrelated_regress,
)


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.1, 0.0


# alphas, f_min_vals, (reg_param_opt, hub_param_opt), (sigmas,) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
#     f_L2_reg,
#     f_hat_Huber_decorrelated_noise,
#     0.1,
#     100,
#     250,
#     [1.0, 1.0],
#     {"reg_param": 3.0},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#         "a": 1.0
#     },
#     initial_cond_fpe=(0.6, 0.2, 0.9),
#     funs=[sigma_order_param],
#     funs_args=[{}],
# )

(
    alphas,
    f_min_vals,
    (reg_param_opt, hub_params_opt),
    (sigmas, qs, ms),
) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    0.01,
    100,
    100,
    [3.0, 3.0],
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)

# print("first done")

# alphas_ls, last_reg_param_stable = alsw.sweep_alpha_minimal_stable_reg_param(
#     f_L2_reg,
#     f_hat_L1_decorrelated_noise,
#     0.01,
#     100,
#     50,
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

first_idx = 0
for idx, rp in enumerate(reg_param_opt):
    if rp <= 0.0:
        first_idx = idx
        break


plt.figure(figsize=(10, 10))

plt.subplot(311)
plt.title("Huber regression, Huber loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(delta_in, delta_out, beta))
plt.plot(alphas, f_min_vals)
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.grid()

plt.subplot(312)
plt.plot(alphas, reg_param_opt, label=r"$\lambda_{opt}$")
plt.plot(alphas, hub_params_opt, label=r"$a_{opt}$")
# plt.plot(alphas_ls, last_reg_param_stable, label=r"$\lambda_{stable}$")
# plt.plot(alphas, hub_param_opt, label=r"$\alpha_{opt}$")
# plt.plot(alphas, condition_MP(alphas), label=r"$min (0, 1-\alpha) $")
plt.axvline(alphas[first_idx], color="red")
plt.xscale("log")
plt.ylim([-20, 8])
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()

plt.subplot(313)
# plt.plot(alphas, sigmas, label=r"$\Sigma$")
# plt.plot(alphas, 1 - alphas * (sigmas / (sigmas + 1))**2, label=r"$1 - \alpha \Sigma^2 / (\Sigma + 1)^2$")
plt.plot(alphas, stability_Huber_decorrelated_regress(ms, qs, sigmas, alphas, reg_param_opt, delta_in, delta_out, percentage, beta, hub_params_opt), label=r"Stability")
plt.legend()
plt.axvline(alphas[first_idx], color="red")
plt.xscale("log")
plt.grid()
# plt.ylabel(r"$1 - \alpha \Sigma^2 / (\Sigma + 1)^2$")
plt.ylabel("Stability cond.")
plt.xlabel(r"$\alpha$")

plt.show()

np.savetxt(
    "./simulations/data/TEST_alpha_sweep_Huber.csv",
    np.array([alphas, f_min_vals, reg_param_opt, hub_params_opt]).T,
    delimiter=",",
    header="alpha,f_min,lambda_opt,hub_param_opt",
)
