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
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def V_order_param(m, q, V):
    return V


def q_order_param(m, q, V):
    return q


def m_order_param(m, q, V):
    return m


alpha_min, alpha_max = 2.8, 1_000
n_alpha_pts = 60
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.0

init_cond = (0.9677292548128574, 0.9526508037535942, 0.22209856977678497)

data_folder = "./data/mod_Tukey_decorrelated_noise"
file_name = f"optimal_se_huber_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}.pkl"

(
    alphas,
    f_min_vals,
    (reg_param_opt, hub_params_opt),
    (Vs_se, qs_se, ms_se),
) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    [1.0, 1.0],
    {"reg_param": 1.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=init_cond,
    funs=[V_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-10,
)

data = {
    "alphas": alphas,
    "f_min_vals": f_min_vals,
    "reg_param_opt": reg_param_opt,
    "hub_params_opt": hub_params_opt,
    "Vs": Vs_se,
    "qs": qs_se,
    "ms": ms_se,
}

with open(os.path.join(data_folder, file_name), "wb") as f:
    pickle.dump(data, f)

plt.plot(alphas, f_min_vals)

reps = 10
ds = [1000]
for d in ds:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
    mask = alphas < 100
    alphas = alphas[mask]
    len_alphas = len(alphas)
    ms = np.empty((len_alphas, 2))
    qs = np.empty((len_alphas, 2))
    estim_error = np.empty((len_alphas, 2))
    gen_error = np.empty((len_alphas, 2))

    for i, alpha in enumerate(tqdm(alphas)):
        n = int(alpha * d)

        m_list, q_list, estim_error_list, gen_error_list = [], [], [], []
        rep = 0
        while rep < reps:
            xs, ys, xs_gen, ys_gen, wstar = data_generation(
                measure_gen_decorrelated,
                d,
                n,
                1000,
                (delta_in, delta_out, percentage, beta),
            )

            # try:
            #     w = find_coefficients_Huber(
            #         ys, xs, reg_param, tau, inital_w=wstar + np.random.randn(d) * 1.0
            #     )
            # except ValueError:
            #     pass

            rho = np.sum(wstar**2) / d
            w_init = ms_se[i] * wstar + np.sqrt(qs_se[i] - ms_se[i] ** 2) * np.random.randn(d)
            # w_init = 10.0 * np.random.randn(d)

            try:
                w = find_coefficients_Huber(ys, xs, reg_param_opt[i], hub_params_opt[i])
            except ValueError:
                print("ValueError")
                continue

            m, q = np.dot(w, wstar) / d, np.sum(w**2) / d
            m_list.append(m)
            q_list.append(q)

            estim_error_list.append(np.sum((w - wstar) ** 2) / d)
            gen_error_list.append(np.mean((xs_gen @ w - ys_gen) ** 2))

            rep += 1

        ms[i] = np.mean(m_list), np.std(m_list)
        qs[i] = np.mean(q_list), np.std(q_list)
        estim_error[i] = np.mean(estim_error_list), np.std(estim_error_list)
        gen_error[i] = np.mean(gen_error_list), np.std(gen_error_list)

    # print("Results ERM")
    # for i in range(n_alpha_pts):
    #     if alphas[i] < 100:
    #         print(f"alpha: {alphas[i]:.3f}, m: {ms[i, 0]:.3f}, q: {qs[i, 0]:.3f}")

    plt.errorbar(
        alphas, estim_error[:, 0], yerr=estim_error[:, 1], fmt=".-", label=f"d = {d}", ls=""
    )

    file_name_erm = f"optimal_erm_huber_{alpha_min}_{100}_{len_alphas}_{d}_{delta_in}_{delta_out}_{percentage}_{beta}.pkl"

    erm_data = {
        "alphas": alphas,
        "ms": ms,
        "qs": qs,
        "estim_error": estim_error,
        "gen_error": gen_error,
    }

    with open(os.path.join(data_folder, file_name_erm), "wb") as f:
        pickle.dump(erm_data, f)

plt.title(
    "Huber regression, Huber loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(
        delta_in, delta_out, beta
    )
)

plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.grid()

plt.show()
