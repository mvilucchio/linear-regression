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
from linear_regression.erm.erm_solvers import find_adversarial_perturbation_direct_space
from tqdm.auto import tqdm
import pickle
import os
from mpi4py import MPI
from itertools import product

alphas = [1.5]
reg_param = 1.0
eps_training = 0.0
pstar_training = 1.0
ps = [2, 3, 5, "inf"]
dimensions = [int(2**a) for a in range(9, 10)]

epss = np.logspace(-2, 2, 15)
eps_dense = np.logspace(-2, 2, 100)
reps = 25

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

tuples_params = list(product(alphas, ps, dimensions))

assert len(tuples_params) == size

alpha, p, n_features = tuples_params[rank]

data_folder = "./data"
file_name = f"ERM_direct_space_adv_transition_n_features_{{:d}}_alpha_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

if p == "inf":
    epss_rescaled = epss / (n_features ** (1 / 2))
else:
    epss_rescaled = epss / (n_features ** (1 / 2 - 1 / p))

vals = np.empty((reps, len(epss)))
estim_vals_m = np.empty((reps,))
estim_vals_q = np.empty((reps,))
estim_vals_rho = np.empty((reps,))

for j in tqdm(range(reps), desc="rps", leave=False):
    xs, ys, xs_gen, ys_gen, wstar = data_generation(
        measure_gen_no_noise_clasif,
        n_features=n_features,
        n_samples=max(int(n_features * alpha), 1),
        n_generalization=1000,
        measure_fun_args={},
    )

    if eps_training == 0:
        w = find_coefficients_Logistic(ys, xs, reg_param)
    else:
        # the 2.0 is the L2 regularisation
        w = find_coefficients_Logistic_adv(
            ys, xs, reg_param, eps_training, 2.0, pstar_training, wstar
        )

    estim_vals_rho[j] = np.sum(wstar**2) / n_features
    estim_vals_m[j] = np.sum(wstar * w) / n_features
    estim_vals_q[j] = np.sum(w**2) / n_features

    yhat = np.sign(xs_gen @ w)

    for i, eps_i in enumerate(tqdm(epss_rescaled, desc="eps", leave=False)):
        adv_perturbation = find_adversarial_perturbation_direct_space(
            yhat, xs_gen, w, wstar, eps_i, p
        )

        flipped = percentage_flipped_labels_estim(
            yhat,
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
        data_folder, file_name.format(n_features, alpha, p, reg_param, eps_training, pstar_training)
    ),
    "wb",
) as f:
    pickle.dump(data, f)
