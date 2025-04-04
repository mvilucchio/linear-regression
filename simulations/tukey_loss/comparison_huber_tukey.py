from linear_regression.data.generation import data_generation, measure_gen_decorrelated
from linear_regression.erm.erm_solvers import find_coefficients_mod_Tukey, find_coefficients_Huber
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regression.Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (
    f_hat_mod_Tukey_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg


def V_order_param(m, q, V):
    return V


def q_order_param(m, q, V):
    return q


def m_order_param(m, q, V):
    return m


alpha_min, alpha_max = 1.0, 10
n_alpha_pts_huber = 100
n_alpha_pts_tukey = 5
delta_in, delta_out, percentage, beta = 1.0, 1.0, 0.1, 0.0
reg_param = 0.1
tau, c = 1.0, 0.01
a = 1.0

(alphas_huber, (ms_huber, qs_huber, Vs_huber)) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts_huber,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": a,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[m_order_param, q_order_param, V_order_param],
    funs_args=[{}, {}, {}],
    update_funs_args=None,
)

# (
#     alphas_huber,
#     f_min_vals_huber,
#     (reg_param_opt, hub_params_opt),
#     (Vs_huber, qs_huber, ms_huber),
# ) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
#     f_L2_reg,
#     f_hat_Huber_decorrelated_noise,
#     alpha_min,
#     alpha_max,
#     n_alpha_pts_huber,
#     [3.0, 3.0],
#     {"reg_param": 3.0},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#         "a": 1.0,
#     },
#     initial_cond_fpe=init_cond,
#     funs=[V_order_param, q_order_param, m_order_param],
#     funs_args=[{}, {}, {}],
#     min_reg_param=1e-10,
# )

init_cond = (0.0797808108383381, 0.2251146635607276, 4.507467545854792)

data_folder = "./data/mod_Tukey_decorrelated_noise/"
file_name = f"mod_Tukey_{tau:.2f}_{c:.2f}_alpha_sweep_{alpha_min:.2f}_{alpha_max:.3f}_{n_alpha_pts_tukey:d}_decorrelated_noise_{delta_in:.2f}_{delta_out:.2f}_{percentage:.2f}_{beta:.2f}_reg_param_{reg_param:.2f}.pkl"

(
    alphas,
    (Vs, qs, ms),
) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_mod_Tukey_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts_tukey,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "tau": tau,
        "c": c,
    },
    initial_cond_fpe=init_cond,
    funs=[V_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    decreasing=False,
)

plt.plot(
    alphas,
    1 - 2 * ms + qs,
    ".-",
    label=r"$\lambda = {:.2f}$".format(reg_param),
)

data = {
    "alphas": alphas,
    "ms": ms,
    "qs": qs,
    "Vs": Vs,
}

with open(os.path.join(data_folder, file_name), "wb") as f:
    pickle.dump(data, f)


with open(os.path.join(data_folder, file_name), "rb") as f:
    data = pickle.load(f)

alphas = data["alphas"]
ms = data["ms"]
qs = data["qs"]
Vs = data["Vs"]


plt.figure(figsize=(7.5, 7.5))
plt.plot(alphas, 1 - 2 * ms + qs, label="mod_Tukey")

plt.plot(alphas_huber, 1 - 2 * ms_huber + qs_huber, ".-", label="Huber")

# reps = 10
# alpha_min = 0.1
# n_alpha_pts = 15
# file_name_erm = f"ERM_mod_Tukey_{tau:.2f}_{c:.2f}_alpha_sweep_{alpha_min:.2f}_{alpha_max:.3f}_{n_alpha_pts:d}_reps_{reps:d}_decorrelated_noise_{delta_in:.2f}_{delta_out:.2f}_{percentage:.2f}_{beta:.2f}.pkl"

# with open(os.path.join(data_folder, file_name_erm), "rb") as f:
#     data = pickle.load(f)

# alphas_erm = data["alphas"]
# ms_erm = data["ms"]
# qs_erm = data["qs"]
# gen_error = data["gen_error"]
# estim_err = data["estim_error"]

# plt.errorbar(
#     alphas_erm,
#     estim_err[:, 0],
#     yerr=estim_err[:, 1],
#     fmt=".",
#     label="ERM",
# )

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$E_{\mathrm{gen}}$")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.xlim(alpha_min, alpha_max)
# plt.ylim(5e-3, 1e0)
plt.legend()

plt.show()
