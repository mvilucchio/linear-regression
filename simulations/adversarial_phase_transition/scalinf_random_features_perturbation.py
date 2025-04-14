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
from linear_regression.erm.erm_solvers import find_adversarial_perturbation_linear_rf
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_linear_features,
)
from tqdm.auto import tqdm
import os
import pickle

dimensions = [2**a for a in range(4, 13)]
min_dimension = min(dimensions)
max_dimension = max(dimensions)
ps = [2, 3, 5]
reg_param = 1.0
alpha = 1.0
gamma = 0.5

epss = np.logspace(-1, 2, 10)
dim_dense = np.logspace(np.log10(min_dimension), np.log10(max_dimension), 100)
reps = 5

eps_i = 1.0
run_experiment = False

for p in tqdm(ps, desc="p"):
    if run_experiment:
        value_of_minima = np.empty_like(dimensions)
        for i, n_hidden_features in enumerate(tqdm(dimensions, desc="dim")):
            n_features = int(n_hidden_features / gamma)
            n_samples = int(n_features * alpha)

            vals = np.empty((reps,))
            estim_vals_m = np.empty((reps,))
            estim_vals_q = np.empty((reps,))
            estim_vals_rho = np.empty((reps,))

            F = np.random.normal(0.0, 1.0, (n_hidden_features, n_features))

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

                w = find_coefficients_Logistic(ys, xs, reg_param)

                estim_vals_rho[j] = np.sum(wstar**2) / n_hidden_features
                estim_vals_m[j] = np.sum(np.dot(wstar, F @ w)) / n_hidden_features
                estim_vals_q[j] = np.sum((F @ w) ** 2) / n_hidden_features

                yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), n_hidden_features, axis=1)

                xs_gen = cs_gen @ F / np.sqrt(n_hidden_features)
                yhat = np.sign(xs_gen @ w)

                adv_perturbation = find_adversarial_perturbation_linear_rf(
                    yhat, cs_gen, w, F, wstar, eps_i, p
                )

                tmp = adv_perturbation

                vals[j] = np.dot(tmp, F @ w)

            value_of_minima[i] = np.mean(vals)

        with open(f"minima_p_{p}.pkl", "wb") as f:
            pickle.dump(value_of_minima, f)

    with open(f"minima_p_{p}.pkl", "rb") as f:
        value_of_minima = pickle.load(f)
        print(value_of_minima.shape)

    plt.plot(dimensions, value_of_minima, "x", label=f"p = {p}")

    # perform the log log linear fit of value_of_minima and find the exponent
    x_vals = np.log(dimensions[2:])
    y_vals = np.log(value_of_minima[2:])
    A = np.vstack([x_vals, np.ones(len(x_vals))]).T
    m, c = np.linalg.lstsq(A, y_vals, rcond=None)[0]
    print(f"p = {p}, intercept = {c}, slope = {m}")

    plt.plot(
        dim_dense, np.exp(c) * dim_dense ** (3 / 2 - 1 / p), "--", label=f"p = {p} (theoretical)"
    )
    plt.plot(dim_dense, np.exp(c) * dim_dense**m, ":", label=f"p = {p} (fitted)")
    print(f"diff exponents = {3 / 2 - 1 / p - m}")

plt.xlabel("n_features")
plt.ylabel("value of minima")
plt.xscale("log", base=10)
plt.yscale("log", base=10)
plt.grid(which="major")
plt.legend()

plt.show()
