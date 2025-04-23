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
    reps = 30
    alpha_min, alpha_max, n_alpha_pts = 1.0, 300, 25
    tau, c = 1.0, 0.001
    reg_param = 2.0

delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.1

data_folder = "./data/mod_Tukey_decorrelated_noise/"
file_name = f"ERM_mod_Tukey_{tau:.2f}_{c:.2e}_alpha_sweep_{alpha_min:.2f}_{alpha_max:.3f}_{n_alpha_pts:d}_reps_{reps:d}_d_{d:d}_decorrelated_noise_{delta_in:.2f}_{delta_out:.2f}_{percentage:.2f}_{beta:.2f}.pkl"

data = np.loadtxt(
    join(data_folder, file_name),
    delimiter=",",
    skiprows=1,
)

alphas = data[:, 0]
ms_means, ms_stds = data[:, 1], data[:, 2]
qs_means, q_stds = data[:, 3], data[:, 4]
estim_errors_means, estim_errors_stds = data[:, 5], data[:, 6]
gen_errors_means, gen_errors_stds = data[:, 7], data[:, 8]

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.errorbar(
    alphas,
    ms_means,
    yerr=ms_stds,
    label=r"$\hat{m}$",
    fmt="o-",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$m$")
plt.xscale("log")
plt.yscale("log")
plt.grid()

plt.subplot(2, 2, 2)
plt.errorbar(
    alphas,
    qs_means,
    yerr=q_stds,
    label=r"$\hat{q}$",
    fmt="o-",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$q$")
plt.xscale("log")
plt.yscale("log")
plt.grid()

plt.subplot(2, 2, 3)
plt.errorbar(
    alphas,
    estim_errors_means,
    yerr=estim_errors_stds,
    label=r"$\hat{m}$",
    fmt="o-",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"Estim error")
plt.xscale("log")
plt.yscale("log")
plt.grid()

plt.subplot(2, 2, 4)
plt.errorbar(
    alphas,
    gen_errors_means,
    yerr=gen_errors_stds,
    label=r"$\hat{m}$",
    fmt="o-",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"Generalization error")
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()

plt.show()
