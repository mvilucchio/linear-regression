import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import percentage_error_from_true
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_direct_space,
)
from tqdm.auto import tqdm
import pickle
import os
import sys

alpha, eps_training = float(sys.argv[1]), float(sys.argv[2])

reg_param = 1e-3
pstar_t = 1.0
ps = [2, 3, "inf"]
dimensions = [int(2**a) for a in range(10, 12)]

epss = np.logspace(-1.5, 1.5, 15)
reps = 10

data_folder = "./data/direct_space"
file_name = f"ERM_direct_space_perc_misclass_n_features_{{:d}}_alpha_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

for p in tqdm(ps, desc="p", leave=False):
    for d in tqdm(dimensions, desc="n", leave=False):
        if p == "inf":
            epss_rescaled = epss * (d ** (-1 / 2))
        else:
            epss_rescaled = epss * (d ** (-1 / 2 + 1 / p))

        vals = np.empty((reps, len(epss)))
        estim_vals_m = np.empty((reps,))
        estim_vals_q = np.empty((reps,))
        estim_vals_rho = np.empty((reps,))

        for j in tqdm(range(reps), desc="rps", leave=False):
            xs, ys, xs_gen, ys_gen, wstar = data_generation(
                measure_gen_no_noise_clasif,
                n_features=d,
                n_samples=max(int(d * alpha), 1),
                n_generalization=1000,
                measure_fun_args={},
            )

            if eps_training == 0.0:
                w = find_coefficients_Logistic(ys, xs, reg_param)
            else:
                w = find_coefficients_Logistic_adv(
                    ys, xs, reg_param, eps_training, 2.0, pstar_t, wstar
                )

            estim_vals_rho[j] = np.sum(wstar**2) / d
            estim_vals_m[j] = np.sum(wstar * w) / d
            estim_vals_q[j] = np.sum(w**2) / d

            for i, eps_i in enumerate(tqdm(epss_rescaled, desc="eps", leave=False)):
                adv_perturbation = find_adversarial_perturbation_direct_space(
                    ys_gen, xs_gen, w, wstar, eps_i, p
                )

                flipped = percentage_error_from_true(
                    ys_gen,
                    xs_gen,
                    w,
                    wstar,
                    xs_gen + adv_perturbation,
                )

                vals[j, i] = flipped

        mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
        mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
        mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)

        data = {
            "epss": epss,
            "vals": vals,
            "estim_vals_m": estim_vals_m,
            "estim_vals_q": estim_vals_q,
            "estim_vals_rho": estim_vals_rho,
            "mean_m": mean_m,
            "std_m": std_m,
            "mean_q": mean_q,
            "std_q": std_q,
            "mean_rho": mean_rho,
            "std_rho": std_rho,
        }

        with open(
            os.path.join(
                data_folder,
                file_name.format(d, alpha, p, reg_param, eps_training, pstar_t),
            ),
            "wb",
        ) as f:
            pickle.dump(data, f)

        print(f"done for d = {d} and p = {p}")
