import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
    data_generation_hastie,
)
from linear_regression.aux_functions.percentage_flipped import percentage_misclassified_hastie_model
from linear_regression.erm.metrics import (
    percentage_flipped_labels_estim,
    percentage_error_from_true,
    adversarial_error_data,
)
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
    find_adversarial_error_rf,
)
from scipy.integrate import quad
from tqdm.auto import tqdm
import os
import sys
from cvxpy.error import SolverError
import pickle

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
        1.5,
        0.5,
        1e-2,
        0.0,
    )

# DO NOT CHANGE
pstar = 1.0
reg = 2.0

dimensions = [int(2**a) for a in range(9, 10)]
reps = 10

epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

data_folder = "./data/hastie_model_training"
file_name = f"ERM_sweep_eps_Hastie_Linf_d_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}_regparam_{reg_param:.1e}.csv"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

for d in tqdm(dimensions, desc="dim", leave=False):
    p = int(d / gamma)
    n = int(d * alpha)

    print(f"p: {p}, d: {d}, n: {n}")

    # works for p = "inf"
    # epss_rescaled = epss * (p ** (-1 / 2))
    epss_rescaled = epss / np.sqrt(d)

    vals_misclass = np.empty((reps, len(epss)))
    vals_flipped = np.empty((reps, len(epss)))
    vals_adverr = np.empty((reps, len(epss)))

    estim_vals_m = np.empty((reps,))
    estim_vals_q = np.empty((reps,))
    estim_vals_q_latent = np.empty((reps,))
    estim_vals_q_feature = np.empty((reps,))
    estim_vals_rho = np.empty((reps,))
    estim_vals_P = np.empty((reps,))

    j = 0
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
            w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_training)
        except (ValueError, UserWarning, SolverError) as e:
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
            except (ValueError, UserWarning, SolverError) as e:
                print("Error in finding adversarial perturbation:", e)
                break

            flipped = np.mean(
                ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )

            vals_flipped[j, i] = flipped

            try:
                adv_perturbation = find_adversarial_perturbation_linear_rf(
                    yhat_gen, zs_gen, w, F.T, wstar, eps_i, "inf"
                )
            except (ValueError, UserWarning, SolverError) as e:
                print("Error in finding adversarial perturbation:", e)
                break

            misclass = np.mean(
                yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )

            vals_misclass[j, i] = misclass

            try:
                adv_perturbation = find_adversarial_error_rf(
                    ys_gen, zs_gen, w, F.T, wstar, eps_i, "inf"
                )
            except (ValueError, UserWarning, SolverError) as e:
                print("Error in finding adversarial perturbation:", e)
                break

            adv_err = np.mean(
                ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )

            vals_adverr[j, i] = adv_err
            i += 1
        else:
            j += 1

    mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
    mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
    mean_q_latent, std_q_latent = np.mean(estim_vals_q_latent), np.std(estim_vals_q_latent)
    mean_q_feature, std_q_feature = np.mean(estim_vals_q_feature), np.std(estim_vals_q_feature)
    mean_P, std_P = np.mean(estim_vals_P), np.std(estim_vals_P)
    mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)
    mean_misclass, std_misclass = np.mean(vals_misclass, axis=0), np.std(vals_misclass, axis=0)
    mean_flipped, std_flipped = np.mean(vals_flipped, axis=0), np.std(vals_flipped, axis=0)
    mean_adverr, std_adverr = np.mean(vals_adverr, axis=0), np.std(vals_adverr, axis=0)

    data = {
        "eps": epss,
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
        "mean_flipped": mean_flipped,
        "std_flipped": std_flipped,
        "mean_adverr": mean_adverr,
        "std_adverr": std_adverr,
    }

    data_file = os.path.join(data_folder, file_name.format(d))

    with open(data_file, "wb") as f:
        pickle.dump(data, f)
