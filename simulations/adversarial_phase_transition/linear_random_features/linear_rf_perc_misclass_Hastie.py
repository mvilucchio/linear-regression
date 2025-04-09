import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
    data_generation_hastie,
)
from linear_regression.erm.metrics import (
    percentage_error_from_true,
)
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
)
from tqdm.auto import tqdm
import os
import pickle
import sys

gamma, alpha, eps_training = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

pstar_t = 1.0
reg_param = 1e-3
ps = ["inf"]
dimensions = [int(2**a) for a in range(10, 12)]
epss = np.logspace(-1.5, 1.5, 15)
reps = 10

data_folder = "./data/linear_random_features"
file_name = f"ERM_Hastie_linear_rf_perc_misclass_n_features_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

for p in tqdm(ps, desc="p", leave=False):
    for d in tqdm(dimensions, desc="dim", leave=False):
        # n_features is p in the paper
        n_features = int(d / gamma)
        n_samples = int(d * alpha)

        if p == "inf":
            epss_rescaled = epss * (d ** (-1 / 2))
        else:
            epss_rescaled = epss * (d ** (-1 / 2 + 1 / p))

        vals = np.empty((reps, len(epss)))
        estim_vals_m = np.empty((reps,))
        estim_vals_q = np.empty((reps,))
        estim_vals_rho = np.empty((reps,))

        for j in tqdm(range(reps), desc="reps", leave=False):
            xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F = data_generation_hastie(
                measure_gen_no_noise_clasif,
                d=d,
                n=max(n_samples, 1),
                n_gen=1000,
                measure_fun_args={},
                gamma=gamma,
            )

            assert F.shape == (d, n_features)
            assert xs.shape == (n_samples, n_features)

            if eps_training == 0.0:
                w = find_coefficients_Logistic(ys, xs, reg_param)
            else:
                w = find_coefficients_Logistic_adv(
                    ys,
                    xs,
                    0.5 * reg_param,
                    eps_training,
                    2.0,
                    pstar_t,
                    F.T @ wstar / np.sqrt(d),
                )

            estim_vals_rho[j] = np.sum(wstar**2) / d
            estim_vals_m[j] = np.sum(np.dot(wstar, F @ w)) / d
            estim_vals_q[j] = np.sum((F @ w) ** 2) / d + np.sum(w**2) / d

            yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)

            yhat_gen = np.sign(xs_gen @ w)

            for i, eps_i in enumerate(tqdm(epss_rescaled, desc="eps", leave=False)):
                adv_perturbation = find_adversarial_perturbation_linear_rf(
                    ys_gen, xs_gen, w, F, wstar, eps_i, p
                )

                flipped = percentage_error_from_true(
                    ys_gen,
                    xs_gen,
                    w,
                    wstar,
                    xs_gen + adv_perturbation,
                    hidden_model=True,
                    projection_matrix=F,
                )

                vals[j, i] = flipped

        mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
        mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
        mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)

        print(
            f"Estimated m = {mean_m:.3f} ± {std_m:.3f} q = {mean_q:.3f} ± {std_q:.3f} rho = {mean_rho:.3f} ± {std_rho:.3f}"
        )

        data = {
            "epss": epss,
            "vals": vals,
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
                file_name.format(d, alpha, gamma, p, reg_param, eps_training, pstar_t),
            ),
            "wb",
        ) as f:
            pickle.dump(data, f)
            print(f"Saved data for d = {d} p = {p}")

        print(f"Saved data for d = {d} features = {n_features} p = {p}")
