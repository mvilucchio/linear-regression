import numpy as np
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
        0.7,
        1.2,
        1e-2,
        0.0,
    )

# DO NOT CHANGE, NOT IMPLEMENTED FOR OTHERS
pstar_t = 1.0

dimensions = [int(2**a) for a in range(8, 10)]
reps = 10

epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

data_folder = "./data/hastie_model_training"
file_name = f"ERM_flipped_Hastie_Linf_d_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}.pkl"

for d in tqdm(dimensions, desc="dim", leave=False):
    p = int(d / gamma)
    n = int(p * alpha)

    # works for p = "inf"
    epss_rescaled = epss * (p ** (-1 / 2))

    vals = np.empty((reps, len(epss)))
    estim_vals_m = np.empty((reps,))
    estim_vals_q = np.empty((reps,))
    estim_vals_rho = np.empty((reps,))
    estim_vals_P = np.empty((reps,))

    for j in tqdm(range(reps), desc="reps", leave=False):
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F = data_generation_hastie(
            measure_gen_no_noise_clasif,
            d=d,
            n=max(n, 1),
            n_gen=1000,
            measure_fun_args={},
            gamma=gamma,
        )

        assert xs.shape == (n, p)
        assert ys.shape == (n,)
        assert zs.shape == (n, d)
        assert F.shape == (p, d)

        print("all checks passed")

        if eps_training == 0.0:
            w = find_coefficients_Logistic(ys, xs, reg_param)
        else:
            w = find_coefficients_Logistic_adv(
                ys, xs, 0.5 * reg_param, eps_training, 2.0, pstar_t, F @ wstar
            )
        # w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_training)

        estim_vals_rho[j] = np.sum(wstar**2) / d
        estim_vals_m[j] = np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma))
        estim_vals_q[j] = np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p
        estim_vals_P[j] = np.mean(np.abs(w))

        yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)

        yhat_gen = np.sign(xs_gen @ w)

        for i, eps_i in enumerate(tqdm(epss_rescaled, desc="eps", leave=False)):
            adv_perturbation = find_adversarial_perturbation_linear_rf(
                yhat_gen, zs_gen, w, F.T, wstar, eps_i, "inf"
            )

            flipped = np.mean(np.sign(yhat_gen) != np.sign((zs_gen + adv_perturbation) @ F.T @ w))

            vals[j, i] = flipped

    mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
    mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
    mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)
    mean_flipped, std_flipped = np.mean(vals, axis=0), np.std(vals, axis=0)

    data = {
        "eps": epss,
        "vals": vals,
        "mean_m": mean_m,
        "std_m": std_m,
        "mean_q": mean_q,
        "std_q": std_q,
        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "mean_flipped": mean_flipped,
        "std_flipped": std_flipped,
    }

    data_file = os.path.join(data_folder, file_name.format(d))

    # save with pickle
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
