import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
    data_generation_hastie,
)
from linear_regression.erm.metrics import (
    percentage_flipped_labels_estim,
    percentage_error_from_true,
)
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
)
from scipy.integrate import quad
from tqdm.auto import tqdm
import os
import sys
from itertools import product
import warnings
import pickle

warnings.filterwarnings("error")

if len(sys.argv) > 1:
    eps_min, eps_max, n_epss, alpha, gamma, reg_param, eps_training = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
        float(sys.argv[7]),
    )
else:
    eps_min, eps_max, n_epss, alpha, gamma, reg_param, eps_training = (
        0.1,
        10.0,
        15,
        1.0,
        1.0,
        1e-2,
        0.0,
    )

# DO NOT CHANGE, NOT IMPLEMENTED FOR OTHERS
pstar_t = 1.0

dimensions = [int(2**a) for a in range(8, 11)]
reps = 10

epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

data_folder = "./data/hastie_model_training"
file_name = f"ERM_misclass_Hastie_Linf_d_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}.pkl"

for d in tqdm(dimensions, desc="dim", leave=False):
    p = int(d / gamma)
    n = int(d * alpha)

    # works for p = "inf"
    # epss_rescaled = epss * (p ** (-1 / 2))
    epss_rescaled = epss / np.sqrt(d)

    vals = np.empty((reps, len(epss)))
    estim_vals_m = np.empty((reps,))
    estim_vals_q = np.empty((reps,))
    estim_vals_q_latent = np.empty((reps,))
    estim_vals_q_feature = np.empty((reps,))
    estim_vals_rho = np.empty((reps,))
    estim_vals_P = np.empty((reps,))

    j = 0
    # for j in tqdm(range(reps), desc="reps", leave=False):
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
            measure_gen_no_noise_clasif,
            d=d,
            n=max(n, 1),
            n_gen=1000,
            measure_fun_args={},
            gamma=gamma,
            noi=True,
        )

        assert xs.shape == (n, p)
        assert ys.shape == (n,)
        assert zs.shape == (n, d)
        assert F.shape == (p, d)

        try:
            if eps_training == 0.0:
                w = find_coefficients_Logistic(ys, xs, reg_param)
            else:
                w = find_coefficients_Logistic_adv(
                    ys, xs, 0.5 * reg_param, eps_training, 2.0, pstar_t, F @ wstar
                )
        except ValueError as e:
            print("Error in finding coefficients:", e)
            continue

        estim_vals_rho[j] = np.sum(wstar**2) / d
        estim_vals_m[j] = np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma))
        estim_vals_q[j] = np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p
        estim_vals_q_latent[j] = np.dot(F.T @ w, F.T @ w) / d
        estim_vals_q_feature[j] = np.dot(w, w) / p
        estim_vals_P[j] = np.mean(np.abs(w))

        yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)

        yhat_gen = np.sign(xs_gen @ w)

        i = 0
        while i < len(epss_rescaled):
            eps_i = epss_rescaled[i]
            try:
                adv_perturbation = find_adversarial_perturbation_linear_rf(
                    ys_gen, zs_gen, w, F.T, wstar, eps_i, "inf"
                )
            except (ValueError, UserWarning) as e:
                print("Error in finding adversarial perturbation:", e)
                vals[j, i] = np.nan
                i += 1
                continue
            flipped = np.mean(
                ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )

            vals[j, i] = flipped
            i += 1

        j += 1

    mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
    mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
    mean_q_latent, std_q_latent = np.mean(estim_vals_q_latent), np.std(estim_vals_q_latent)
    mean_q_feature, std_q_feature = np.mean(estim_vals_q_feature), np.std(estim_vals_q_feature)
    mean_P, std_P = np.mean(estim_vals_P), np.std(estim_vals_P)
    mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)
    mean_misclass, std_misclass = np.mean(vals, axis=0), np.std(vals, axis=0)

    data = {
        "eps": epss,
        "vals": vals,
        "mean_m": mean_m,
        "std_m": std_m,
        "mean_q": mean_q,
        "std_q": std_q,
        "mean_q_latent": mean_q_latent,
        "std_q_latent": std_q_latent,
        "mean_q_feature": mean_q_feature,
        "std_q_feature": std_q_feature,
        "mean_P": mean_P,
        "std_P": std_P,
        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "mean_misclass": mean_misclass,
        "std_misclass": std_misclass,
    }

    data_file = os.path.join(data_folder, file_name.format(d))

    # save with pickle
    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    plt.errorbar(
        epss, mean_misclass, yerr=std_misclass, linestyle="", marker=".", label=f"$d = {d}$"
    )

if gamma <= 1:
    AA = epss * np.sqrt(mean_q_latent - mean_m**2 / gamma) * np.sqrt(2 / np.pi) * np.sqrt(gamma)
else:
    AA = epss * np.sqrt(mean_q_feature - mean_m**2 / gamma) / np.sqrt(gamma) * np.sqrt(2 / np.pi)

out_theory = np.empty((len(epss),))
for i, eps in enumerate(epss):
    int_val_1 = quad(
        lambda x: np.exp(-(x**2) / (2 * mean_q))
        / np.sqrt(2 * np.pi * mean_q)
        * erfc(
            mean_m
            / np.sqrt(gamma)
            * x
            / np.sqrt(2 * mean_q * (mean_q * mean_rho - mean_m**2 / gamma))
        )
        * np.heaviside(-AA[i] - x, 0),
        -np.inf,
        np.inf,
    )[0]
    int_val_2 = quad(
        lambda x: np.exp(-(x**2) / (2 * mean_q))
        / np.sqrt(2 * np.pi * mean_q)
        * (
            1
            + erf(
                mean_m
                / np.sqrt(gamma)
                * x
                / np.sqrt(2 * mean_q * (mean_q * mean_rho - mean_m**2 / gamma))
            )
        )
        * np.heaviside(x - AA[i], 0),
        -np.inf,
        np.inf,
    )[0]
    out_theory[i] = 1 - 0.5 * (int_val_1 + int_val_2)

plt.plot(epss, out_theory, label="theoretical", linestyle="--")

plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\mathbb{P}(\hat{y} \neq y)$")
plt.xscale("log")
plt.grid(which="both")
plt.legend()
plt.show()
