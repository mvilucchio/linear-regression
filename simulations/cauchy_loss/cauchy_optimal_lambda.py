from linear_regression.data.generation import data_generation, measure_gen_decorrelated
from linear_regression.erm.erm_solvers import find_coefficients_Cauchy
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regression.cauchy_loss import (
    f_hat_Cauchy_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np


def V_order_param(m, q, V):
    return V


def q_order_param(m, q, V):
    return q


def m_order_param(m, q, V):
    return m


alpha_min, alpha_max = 2.8, 1_000
n_alpha_pts = 60
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.0
reg_param = 0.1
tau = 1.0
c = 0.01

data_folder = "./data/Cauchy_decorrelated_noise"
file_name = f"optimal_se_cauchy_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}_{c}.pkl"


if not os.path.exists(os.path.join(data_folder, file_name)):
    init_cond = (0.9677292548128574, 0.9526508037535942, 0.22209856977678497)

    (
        alphas_se,
        f_min_vals,
        (reg_param_opt, hub_params_opt),
        (Vs_se, qs_se, ms_se),
    ) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
        f_L2_reg,
        f_hat_Cauchy_decorrelated_noise,
        alpha_min,
        alpha_max,
        n_alpha_pts,
        [0.1, 1.0],
        {"reg_param": 3.0},
        {
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
            "tau": 1.0,
        },
        initial_cond_fpe=init_cond,
        funs=[V_order_param, q_order_param, m_order_param],
        funs_args=[{}, {}, {}],
        min_reg_param=1e-5,
    )

    data_se = {
        "alphas": alphas_se,
        "reg_params": reg_param_opt,
        "tau_params": hub_params_opt,
        "ms": ms_se,
        "qs": qs_se,
        "Vs": Vs_se,
    }

    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data_se, f)

with open(os.path.join(data_folder, file_name), "rb") as f:
    data_se = pickle.load(f)
    alphas_se = data_se["alphas"]
    ms_se = data_se["ms"]
    qs_se = data_se["qs"]
    Vs_se = data_se["Vs"]
    reg_param_opt = data_se["reg_params"]
    hub_params_opt = data_se["tau_params"]

print("Results SE")
for i in range(n_alpha_pts):
    print(f"alpha: {alphas_se[i]:.3f}, m: {ms_se[i]:.3f}, q: {qs_se[i]:.3f}, V: {Vs_se[i]:.3f}")

plt.figure()
plt.plot(alphas_se, 1 - 2 * ms_se + qs_se, label="SE")

reps = 5
ds = [300]
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
        for rep in tqdm(range(reps), leave=False):
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

            w = find_coefficients_Cauchy(
                ys, xs, reg_param_opt[i], hub_params_opt[i], initial_w=w_init
            )

            m, q = np.dot(w, wstar) / d, np.sum(w**2) / d
            m_list.append(m)
            q_list.append(q)

            estim_error_list.append(np.sum((w - wstar) ** 2) / d)
            gen_error_list.append(np.mean((xs_gen @ w - ys_gen) ** 2))

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

    file_name_erm = f"optimal_erm_cauchy_{alpha_min}_{100}_{len_alphas}_{d}_{delta_in}_{delta_out}_{percentage}_{beta}.pkl"

    erm_data = {
        "alphas": alphas,
        "ms": ms,
        "qs": qs,
        "estim_error": estim_error,
        "gen_error": gen_error,
    }

    with open(os.path.join(data_folder, file_name_erm), "wb") as f:
        pickle.dump(erm_data, f)

plt.xscale("log")
plt.xlabel(r"$\alpha$")
plt.yscale("log")
plt.legend()

plt.show()
