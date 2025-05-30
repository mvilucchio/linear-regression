from linear_regression.data.generation import data_generation, measure_gen_decorrelated
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from numba import vectorize, float32, float64
from tqdm import tqdm
import pickle
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regression.Tukey_loss import (
    f_hat_Tukey_decorrelated_noise_TI,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.amp.amp_funcs import GAMP_fullyTAP
from linear_regression.aux_functions.moreau_proximals import proximal_Tukey_loss_TI
import linear_regression.aux_functions.prior_regularization_funcs as priors
import linear_regression.aux_functions.likelihood_channel_functions as like
import numpy as np

# Matéo 

def V_order_param(m, q, V):
    return V


def q_order_param(m, q, V):
    return q


def m_order_param(m, q, V):
    return m


# @vectorize(
#     [
#         float32(float32, float32, float32, float32, float32),
#         float64(float64, float64, float64, float64, float64),
#     ]
# )
# def f_out_mod_Tukey(y: float, omega: float, V: float, tau: float, c: float) -> float:
#     return (proximal_Tukey_modified_quad(y, omega, V, tau, c) - omega) / V

alpha_min, alpha_max = 1.0, 300
n_alpha_pts = 25
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.0
reg_param = 2.0
tau = 1.0

data_folder = "./data/Tukey_decorrelated_noise_TI"
file_name = f"test_se_tukey_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}_{reg_param}_{tau}.pkl"


# check if the file exists, if this is the case do not run it
if not os.path.exists(os.path.join(data_folder, file_name)):
    os.makedirs(data_folder, exist_ok=True)
    # init_cond = (0.07304981546047486, 0.0760530419161838, 9.103303146814449)
    init_cond = (1.15e-2, 3.05e-3, 0.5)

    (alphas_se, (ms_se, qs_se, Vs_se)) = alsw.sweep_alpha_fixed_point(
        f_L2_reg,
        f_hat_Tukey_decorrelated_noise_TI,
        alpha_min,
        alpha_max,
        n_alpha_pts,
        {"reg_param": reg_param},
        {
            "Delta_in": delta_in,
            "Delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
            "tau": tau,
        },
        initial_cond_fpe=init_cond,
        funs=[m_order_param, q_order_param, V_order_param],
        funs_args=[{}, {}, {}],
        update_funs_args=None,
        decreasing=True,
    )

    m_hat_se = np.empty_like(alphas_se)
    q_hat_se = np.empty_like(alphas_se)
    V_hat_se = np.empty_like(alphas_se)

    for i in tqdm(range(n_alpha_pts), leave=False, desc="SE"):
        m_hat_se[i], q_hat_se[i], V_hat_se[i] = f_hat_Tukey_decorrelated_noise_TI(
            ms_se[i],
            qs_se[i],
            Vs_se[i],
            alphas_se[i],
            delta_in,
            delta_out,
            percentage,
            beta,
            tau,
        )

    data_se = {
        "alphas": alphas_se,
        "ms": ms_se,
        "qs": qs_se,
        "Vs": Vs_se,
        "m_hat": m_hat_se,
        "q_hat": q_hat_se,
        "V_hat": V_hat_se,
    }

    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data_se, f)

with open(os.path.join(data_folder, file_name), "rb") as f:
    data_se = pickle.load(f)
    alphas_se = data_se["alphas"]
    ms_se = data_se["ms"]
    qs_se = data_se["qs"]
    Vs_se = data_se["Vs"]
    m_hat_se = data_se["m_hat"]
    q_hat_se = data_se["q_hat"]
    V_hat_se = data_se["V_hat"]

# print("Results SE")
# for i in range(n_alpha_pts):
#     print(f"alpha: {alphas_se[i]:.6f}, m: {ms_se[i]:.6f}, q: {qs_se[i]:.6f}, V: {Vs_se[i]:.6f}")

c=0.0
d = 500
reps = 30

data_folder_ERM = "./data/mod_Tukey_decorrelated_noise/"
file_name_ERM = "ERM_mod_Tukey_1.00_0.00e+00_alpha_sweep_1.00_300.000_25_reps_10_d_500_decorrelated_noise_0.10_1.00_0.10_0.00.pkl" #f"ERM_mod_Tukey_{tau:.2f}_{c:.2e}_alpha_sweep_{alpha_min:.2f}_{alpha_max:.3f}_{n_alpha_pts:d}_reps_{reps:d}_d_{d:d}_decorrelated_noise_{delta_in:.2f}_{delta_out:.2f}_{percentage:.2f}_{beta:.2f}.pkl"

data = np.loadtxt(
    join(data_folder_ERM, file_name_ERM),
    delimiter=",",
    skiprows=1,
)

alphas = data[:, 0]
ms_means, ms_stds = data[:, 1], data[:, 2]
qs_means, q_stds = data[:, 3], data[:, 4]
estim_errors_means, estim_errors_stds = data[:, 5], data[:, 6]
gen_errors_means, gen_errors_stds = data[:, 7], data[:, 8]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# ----------- PLOT m -----------
plt.subplot(1, 2, 1)
plt.errorbar(
    alphas,
    ms_means,
    yerr=ms_stds,
    fmt="o",
    label=r"$m$ ERM",
    color="tab:blue",
    capsize=3,
    alpha=0.7,
)
plt.plot(
    alphas_se,
    ms_se,
    linestyle="-",
    linewidth=2,
    label=r"$m$ SE",
    color="tab:orange",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$m$")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend()

# ----------- PLOT q -----------
plt.subplot(1, 2, 2)
plt.errorbar(
    alphas,
    qs_means,
    yerr=q_stds,
    fmt="o",
    label=r"$q$ ERM",
    color="tab:blue",
    capsize=3,
    alpha=0.7,
)
plt.plot(
    alphas_se,
    qs_se,
    linestyle="-",
    linewidth=2,
    label=r"$q$ SE",
    color="tab:orange",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$q$")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
