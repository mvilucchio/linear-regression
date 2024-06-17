import matplotlib.pyplot as plt
import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import percentage_flipped_labels_estim
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.aux_functions.percentage_flipped import percentage_flipped_direct_space_FGM
import pickle
import os
from mpi4py import MPI
from itertools import product

alphas = [1.5, 2.0]
reg_params = [1.0, 0.1, 0.01]
pstar_t = 1.0
epsilon_t = 0.0

ps = [2, 3, 5, 10]
dimensions = [512, 1024, 2048, 4096]

epss = np.logspace(-2, 2, 15)
eps_dense = np.logspace(-2, 2, 100)
reps = 10

data_folder = "./data"
file_name = f"ERM_direct_space_adv_transition_n_features_{{:d}}_alpha_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

params = list(product(alphas, reg_params, ps, dimensions))

assert len(params) <= size

alpha, reg_param, p, n_features = params[rank]

if p == "inf":
    # epss_rescaled = epss / (n_features ** (1 / 2))
    epss_rescaled = (
        epss
        / np.sqrt(n_features)
        * (
            np.sqrt(2 * np.log(n_features))
            + np.log(2) / np.sqrt(2 * np.log(n_features))
            - 0.5
            * (np.log(np.log(n_features)) + np.log(4 * np.pi))
            / (np.sqrt(2 * np.log(n_features)))
            + (1 / (np.sqrt(2 * np.log(n_features)))) * 0.5772156649
        )
    )
else:
    epss_rescaled = epss / (n_features ** (1 / 2 - 1 / p))

print(f"Generating data ... n_features = {n_features:d} n_samples = {int(n_features * alpha):d}")

vals = np.empty((reps, len(epss)))
estim_vals_m = np.empty((reps,))
estim_vals_q = np.empty((reps,))
estim_vals_rho = np.empty((reps,))

for j in range(reps):
    xs, ys, xs_gen, ys_gen, wstar = data_generation(
        measure_gen_no_noise_clasif,
        n_features=n_features,
        n_samples=max(int(n_features * alpha), 1),
        n_generalization=1000,
        measure_fun_args={},
    )

    if epsilon_t == 0.0:
        w = find_coefficients_Logistic(ys, xs, reg_param)
    else:
        w = find_coefficients_Logistic_adv(ys, xs, reg_param, epsilon_t, pstar_t)

    estim_vals_rho[j] = np.sum(wstar**2) / n_features
    estim_vals_m[j] = np.sum(wstar * w) / n_features
    estim_vals_q[j] = np.sum(w**2) / n_features

    yhat = np.repeat(np.sign(xs_gen @ w).reshape(-1, 1), n_features, axis=1)

    direction_adv = w - ((wstar @ w) / np.sum(wstar**2)) * wstar

    if p == "inf":
        direction_adv_norm = direction_adv / np.max(np.abs(direction_adv))
    else:
        direction_adv_norm = direction_adv / np.linalg.norm(direction_adv, ord=p)

    adv_perturbation = -(yhat * direction_adv_norm[None, :])

    for i, eps_i in enumerate(epss_rescaled):
        flipped = percentage_flipped_labels_estim(
            yhat,
            xs_gen,
            w,
            wstar,
            xs_gen + eps_i * adv_perturbation,
        )

        vals[j, i] = flipped

data_dict = {
    "epss": epss,
    "error_mean": np.mean(vals, axis=0),
    "error_std": np.std(vals, axis=0),
    "estim_vals_m": estim_vals_m,
    "estim_vals_q": estim_vals_q,
    "estim_vals_rho": estim_vals_rho,
}

with open(
    os.path.join(
        data_folder,
        file_name.format(n_features, alpha, p, reg_param, epsilon_t, pstar_t),
    ),
    "wb",
) as file:
    pickle.dump(
        data_dict,
        file,
    )
