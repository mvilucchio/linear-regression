from linear_regression.data.generation import data_generation, measure_gen_decorrelated
from linear_regression.erm.erm_solvers import find_coefficients_mod_Tukey, find_coefficients_Huber
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm
import pickle
import sys
import os

if len(sys.argv) > 1:
    (
        alpha_min,
        alpha_max,
        n_alpha_pts,
        d,
        reps,
        tau,
        c,
        reg_param,
        delta_in,
        delta_out,
        percentage,
        beta,
    ) = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        float(sys.argv[6]),
        float(sys.argv[7]),
        float(sys.argv[8]),
        float(sys.argv[9]),
        float(sys.argv[10]),
        float(sys.argv[11]),
        float(sys.argv[12]),
    )
else:
    d = 500
    reps = 10
    alpha_min, alpha_max, n_alpha_pts = 0.5, 300, 20
    tau, c = 1.0, 0.0
    reg_param = 2.0
    delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.0

data_folder = "./data/mod_Tukey_decorrelated_noise/"
file_name = f"ERM_mod_Tukey_{tau:.2f}_{c:.2e}_alpha_sweep_{alpha_min:.2f}_{alpha_max:.3f}_{n_alpha_pts:d}_reps_{reps:d}_d_{d:d}_decorrelated_noise_{delta_in:.2f}_{delta_out:.2f}_{percentage:.2f}_{beta:.2f}.pkl"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
ms = np.empty((n_alpha_pts, 2))
qs = np.empty((n_alpha_pts, 2))
estim_error = np.empty((n_alpha_pts, 2))
gen_error = np.empty((n_alpha_pts, 2))

for i, alpha in enumerate(tqdm(alphas)):
    n = int(alpha * d)

    m_list, q_list, estim_error_list, gen_error_list = [], [], [], []

    pbar = tqdm(total=reps, leave=False)
    rep = 0
    while rep < reps:
        xs, ys, xs_gen, ys_gen, wstar = data_generation(
            measure_gen_decorrelated,
            d,
            n,
            1000,
            (delta_in, delta_out, percentage, beta),
        )

        try:
            w_init = find_coefficients_Huber(
                ys, xs, reg_param, tau, inital_w=wstar + np.random.randn(d)
            )

            w = find_coefficients_mod_Tukey(ys, xs, reg_param, tau, c, initial_w=w_init)
        except ValueError as e:
            print("Error in finding coefficients:", e)
            #rep += 1
            continue

        m, q = np.dot(w, wstar) / d, np.sum(w**2) / d
        m_list.append(m)
        q_list.append(q)

        estim_error_list.append(np.sum(np.abs(w - wstar) ** 2) / d)
        gen_error_list.append(
            np.mean((ys_gen - xs_gen @ w / np.sqrt(d)) ** 2)
            - np.mean(
                (ys_gen - (1 - percentage + percentage * beta) * xs_gen @ wstar / np.sqrt(d)) ** 2
            )
        )
        rep += 1
        pbar.update(1)

    pbar.close()

    ms[i] = np.mean(m_list), np.std(m_list)
    qs[i] = np.mean(q_list), np.std(q_list)
    estim_error[i] = np.mean(estim_error_list), np.std(estim_error_list)
    gen_error[i] = np.mean(gen_error_list), np.std(gen_error_list)

data = {
    "alpha": alphas,
    "m_mean": ms[:, 0],
    "m_std": ms[:, 1],
    "q_mean": qs[:, 0],
    "q_std": qs[:, 1],
    "estim_err_mean": estim_error[:, 0],
    "estim_err_std": estim_error[:, 1],
    "gen_err_mean": gen_error[:, 0],
    "gen_err_std": gen_error[:, 1],
}

data_array = np.column_stack([data[key] for key in data.keys()])
header = ",".join(data.keys())
np.savetxt(
    join(data_folder, file_name),
    data_array,
    delimiter=",",
    header=header,
    comments="",
)

print("Data saved to", join(data_folder, file_name))

plt.figure(figsize=(7.5, 7.5))
plt.errorbar(alphas, ms[:, 0], yerr=ms[:, 1], fmt=".-", label="m")
plt.errorbar(alphas, qs[:, 0], yerr=qs[:, 1], fmt=".-", label="q")
plt.errorbar(alphas, estim_error[:, 0], yerr=estim_error[:, 1], fmt=".-", label="estim error")
plt.errorbar(alphas, gen_error[:, 0], yerr=gen_error[:, 1], fmt=".-", label="gen error")
plt.xscale("log")
plt.xlabel(r"$\alpha$")
plt.yscale("log")
plt.legend()

plt.show()
