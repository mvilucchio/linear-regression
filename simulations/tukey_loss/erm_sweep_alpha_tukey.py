from linear_regression.data.generation import data_generation, measure_gen_decorrelated
from linear_regression.erm.erm_solvers import find_coefficients_mod_Tukey, find_coefficients_Huber
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle

d = 300
reps = 10
alpha_min, alpha_max, n_alpha_pts = 0.1, 10, 15
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.1
tau, c = 1.0, 0.1
reg_param = 0.1

data_folder = "./data/mod_Tukey_decorrelated_noise/"
file_name = f"ERM_mod_Tukey_{tau:.2f}_{c:.2f}_alpha_sweep_{alpha_min:.2f}_{alpha_max:.3f}_{n_alpha_pts:d}_reps_{reps:d}_decorrelated_noise_{delta_in:.2f}_{delta_out:.2f}_{percentage:.2f}_{beta:.2f}.pkl"

alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
ms = np.empty((n_alpha_pts, 2))
qs = np.empty((n_alpha_pts, 2))
estim_error = np.empty((n_alpha_pts, 2))
gen_error = np.empty((n_alpha_pts, 2))

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

        w_init = find_coefficients_Huber(
            ys, xs, reg_param, tau, inital_w=wstar + np.random.randn(d) * 1.0
        )

        w = find_coefficients_mod_Tukey(ys, xs, reg_param, tau, c, initial_w=w_init)
        # w = w_init

        m, q = np.dot(w, wstar) / d, np.sum(w**2) / d
        m_list.append(m)
        q_list.append(q)

        estim_error_list.append(np.sum(np.abs(w - wstar)) / d)
        gen_error_list.append(np.mean((xs_gen @ w - ys_gen) ** 2))

    ms[i] = np.mean(m_list), np.std(m_list)
    qs[i] = np.mean(q_list), np.std(q_list)
    estim_error[i] = np.mean(estim_error_list), np.std(estim_error_list)
    gen_error[i] = np.mean(gen_error_list), np.std(gen_error_list)

data = {
    "alphas": alphas,
    "ms": ms,
    "qs": qs,
    "estim_error": estim_error,
    "gen_error": gen_error,
}

with open(os.path.join(data_folder, file_name), "wb") as f:
    pickle.dump(data, f)

plt.figure(figsize=(7.5, 7.5))
# plt.errorbar(alphas, ms[:, 0], yerr=ms[:, 1], fmt=".-", label="m")
# plt.errorbar(alphas, qs[:, 0], yerr=qs[:, 1], fmt=".-", label="q")
plt.errorbar(alphas, estim_error[:, 0], yerr=estim_error[:, 1], fmt=".-", label="estim error")
# plt.errorbar(alphas, gen_error[:, 0], yerr=gen_error[:, 1], fmt=".-", label="gen error")
plt.xscale("log")
plt.xlabel(r"$\alpha$")
plt.yscale("log")
plt.legend()

plt.show()
