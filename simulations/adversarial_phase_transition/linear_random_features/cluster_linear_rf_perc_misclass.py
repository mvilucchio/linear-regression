import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import (
    percentage_flipped_labels_estim,
    percentage_error_from_true,
)
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.erm.erm_solvers import find_adversarial_perturbation_RandomFeatures_space
from tqdm.auto import tqdm
import os
import pickle
from mpi4py import MPI
from itertools import product
import warnings

warnings.filterwarnings("error")

alphas = [0.25, 0.5, 1.0, 1.5, 2.0]
gammas = [0.25, 0.5, 1.0, 1.5, 2.0]
reg_params = [0.01, 1, 1.0]
ps = ["inf"]

eps_training = 0.0
pstar_t = 1.0
dimensions = [int(2**a) for a in range(9, 11)]
reps = 10

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

epss = np.logspace(-1, 1, 15)

params_list = list(product(alphas, gammas, reg_params, ps))

assert len(params_list) <= size

alpha, gamma, reg_param, p = params_list[rank]

data_folder = "./data"
file_name = f"ERM_linear_features_adv_transition_true_label_n_features_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_p_{p}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}_pstar_t_{pstar_t}.pkl"

for n_hidden_features in tqdm(dimensions, desc="dim", leave=False):
    n_features = int(n_hidden_features / gamma)
    n_samples = int(n_features * alpha)

    if p == "inf":
        epss_rescaled = epss * (n_features ** (-1 / 2))
    else:
        epss_rescaled = epss * (n_features ** (-1 / 2 + 1 / p))

    vals = np.empty((reps, len(epss)))
    estim_vals_m = np.empty((reps,))
    estim_vals_q = np.empty((reps,))
    estim_vals_rho = np.empty((reps,))

    F = np.random.normal(0.0, 1.0, (n_hidden_features, n_features))

    assert F.shape == (n_hidden_features, n_features)

    for j in tqdm(range(reps), desc="reps", leave=False):
        cs, ys, cs_gen, ys_gen, wstar = data_generation(
            measure_gen_no_noise_clasif,
            n_features=n_hidden_features,
            n_samples=max(n_samples, 1),
            n_generalization=1000,
            measure_fun_args={},
        )

        assert cs.shape == (n_samples, n_hidden_features)

        xs = cs @ F / np.sqrt(n_hidden_features)

        assert xs.shape == (n_samples, n_features)

        if eps_training == 0.0:
            w = find_coefficients_Logistic(ys, xs, reg_param)
        else:
            w = find_coefficients_Logistic_adv(ys, xs, reg_param, eps_training, 2.0, pstar_t, wstar)

        estim_vals_rho[j] = np.sum(wstar**2) / n_hidden_features
        estim_vals_m[j] = np.sum(np.dot(wstar, F @ w)) / n_hidden_features
        estim_vals_q[j] = np.sum((F @ w) ** 2) / n_hidden_features

        yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), n_hidden_features, axis=1)

        xs_gen = cs_gen @ F / np.sqrt(n_hidden_features)

        yhat_gen = np.sign(xs_gen @ w)

        for i, eps_i in enumerate(tqdm(epss_rescaled, desc="eps", leave=False)):
            adv_perturbation = find_adversarial_perturbation_RandomFeatures_space(
                yhat_gen, cs_gen, w, F, wstar, eps_i, p
            )

            flipped = percentage_error_from_true(
                yhat_gen,
                cs_gen,
                w,
                wstar,
                cs_gen + adv_perturbation,
                hidden_model=True,
                projection_matrix=F,
            )

            vals[j, i] = flipped

    mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
    mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
    mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)

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

    with open(os.path.join(data_folder, file_name.format(n_hidden_features)), "wb") as f:
        pickle.dump(data, f)
