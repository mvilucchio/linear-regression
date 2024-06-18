import matplotlib.pyplot as plt
import numpy as np
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L1,
    find_coefficients_Logistic_approx_L1,
)
from linear_regression.data.generation import data_generation, measure_gen_no_noise_clasif
from linear_regression.erm.metrics import (
    estimation_error_data,
    generalisation_error_classification,
    adversarial_error_data,
)
import pickle
from tqdm.auto import tqdm
import os

import warnings

warnings.filterwarnings("error")

alpha_min, alpha_max, n_alpha_pts = 0.1, 2, 8
eps_t = 0.1
eps_g = 0.1
reg_params = [1e-2]
pstar = 1.0

d = 700
reps = 10
n_gen = 1000

data_folder = "./data"
file_name = f"ERM_data_Linf_reg_order_{{:d}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_{n_alpha_pts:d}_dim_{d:d}_reps_{reps:d}_reg_param_{{:.1e}}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl"

plt.figure()

for reg_param in reg_params:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

    gen_error_mean = np.empty_like(alphas)
    gen_error_std = np.empty_like(alphas)

    adversarial_errors_mean = np.empty_like(alphas)
    adversarial_errors_std = np.empty_like(alphas)

    for j, alpha in enumerate(alphas):
        n = int(alpha * d)

        tmp_gen_errors = []
        tmp_adversarial_errors = []

        iter = 0
        pbar = tqdm(total=reps)
        while iter < reps:
            xs_train, ys_train, xs_gen, ys_gen, wstar = data_generation(
                measure_gen_no_noise_clasif, d, n, n_gen, tuple()
            )

            try:
                w = find_coefficients_Logistic_adv_Linf_L1(ys_train, xs_train, reg_param, eps_t)
            except UserWarning as e:
                continue
            except ValueError as e:
                continue

            tmp_gen_errors.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))
            tmp_adversarial_errors.append(
                adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_g, pstar)
            )

            del w
            del xs_gen
            del ys_gen
            del xs_train
            del ys_train
            del wstar

            iter += 1
            pbar.update(1)

        pbar.close()

        gen_error_mean[j] = np.mean(tmp_gen_errors)
        gen_error_std[j] = np.std(tmp_gen_errors)

        adversarial_errors_mean[j] = np.mean(tmp_adversarial_errors)
        adversarial_errors_std[j] = np.std(tmp_adversarial_errors)

    plt.errorbar(
        alphas,
        gen_error_mean,
        yerr=gen_error_std,
        marker="x",
        label=f"Generalisation error, reg_param = {reg_param:.1e} 1",
    )

    plt.errorbar(
        alphas,
        adversarial_errors_mean,
        yerr=adversarial_errors_std,
        marker="x",
        label=f"Adversarial error, reg_param = {reg_param:.1e} 1",
    )

for reg_param in reg_params:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

    gen_error_mean = np.empty_like(alphas)
    gen_error_std = np.empty_like(alphas)

    adversarial_errors_mean = np.empty_like(alphas)
    adversarial_errors_std = np.empty_like(alphas)

    for j, alpha in enumerate(alphas):
        n = int(alpha * d)

        tmp_gen_errors = []
        tmp_adversarial_errors = []

        iter = 0
        pbar = tqdm(total=reps)
        while iter < reps:
            xs_train, ys_train, xs_gen, ys_gen, wstar = data_generation(
                measure_gen_no_noise_clasif, d, n, n_gen, tuple()
            )

            try:
                w = find_coefficients_Logistic_approx_L1(
                    ys_train, xs_train, reg_param, eps_t, pstar, wstar
                )
            except UserWarning as e:
                print(e)
                continue
            except ValueError as e:
                print(e)
                continue

            tmp_gen_errors.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))
            tmp_adversarial_errors.append(
                adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_g, pstar)
            )

            del w
            del xs_gen
            del ys_gen
            del xs_train
            del ys_train
            del wstar

            iter += 1
            pbar.update(1)

        pbar.close()

        gen_error_mean[j] = np.mean(tmp_gen_errors)
        gen_error_std[j] = np.std(tmp_gen_errors)

        adversarial_errors_mean[j] = np.mean(tmp_adversarial_errors)
        adversarial_errors_std[j] = np.std(tmp_adversarial_errors)

    plt.errorbar(
        alphas,
        gen_error_mean,
        yerr=gen_error_std,
        marker="o",
        label=f"Generalisation error, reg_param = {reg_param:.1e} 2",
    )

    plt.errorbar(
        alphas,
        adversarial_errors_mean,
        yerr=adversarial_errors_std,
        marker="o",
        label=f"Adversarial error, reg_param = {reg_param:.1e} 2",
    )

plt.xlabel("alpha")
plt.ylabel("Error")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.title(f"Comparison different L1 norms approximations, $\\varepsilon$ = {eps_t:.1f} d = {d:d}")

plt.show()
