import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import percentage_flipped_labels_NLRF
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_non_linear_rf,
)
import jax.numpy as jnp
from tqdm.auto import tqdm
import os
import sys
import pickle
from numba import vectorize


@vectorize(["float64(float64)"], nopython=True)
def non_linearity(x):
    return np.tanh(x)


# gamma, alpha, eps_training = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

gamma, alpha, eps_training = 1.0, 1.0, 0.0

pstar_t = 1.0
reg_param = 1e-3
ps = [2.0, np.inf]
dimensions = [int(2**a) for a in range(7, 8)] + [int(2**a) for a in range(9, 10)]
epss = np.logspace(-1.5, 1.5, 10)
reps = 5

data_folder = "./data/non_linear_random_features"
file_name = f"ERM_non_linear_rf_perc_misclass_n_features_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

for p in tqdm(ps, desc="p", leave=False):
    for d in tqdm(dimensions, desc="dim", leave=False):
        n_features = int(d / gamma)
        n_samples = int(d * alpha)

        if p == "inf":
            epss_rescaled = epss  # * (d ** (-1 / 2))
        else:
            epss_rescaled = epss

        vals = np.empty((reps, len(epss)))
        estim_vals_m = np.empty((reps,))
        estim_vals_q = np.empty((reps,))
        estim_vals_rho = np.empty((reps,))

        F = np.random.normal(0.0, 1.0, (d, n_features))

        assert F.shape == (d, n_features)

        for j in tqdm(range(reps), desc="reps", leave=False):
            cs, ys, cs_gen, ys_gen, wstar = data_generation(
                measure_gen_no_noise_clasif,
                n_features=d,
                n_samples=max(n_samples, 1),
                n_generalization=1000,
                measure_fun_args={},
            )

            assert cs.shape == (n_samples, d)

            xs = non_linearity(cs @ F / np.sqrt(d))

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
                    F.T @ wstar / np.sqrt(d),  # this is just an intial guess
                )

            estim_vals_rho[j] = np.sum(wstar**2) / d
            estim_vals_m[j] = np.sum(np.dot(wstar, F @ w)) / d
            estim_vals_q[j] = np.sum((F @ w) ** 2) / d

            # yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), n_hidden_features, axis=1)
            yhat = np.sign(xs @ w)

            xs_gen = non_linearity(cs_gen @ F / np.sqrt(d))

            # convert all the arrays to jnp.arrrays
            cs_gen = jnp.array(cs_gen)
            ys_gen = jnp.array(ys_gen)
            w = jnp.array(w)
            F = jnp.array(F)
            wstar = jnp.array(wstar)
            adv_perturbation = jnp.zeros_like(cs)

            for i, eps_i in enumerate(tqdm(epss_rescaled, desc="eps", leave=False)):
                adv_perturbation = find_adversarial_perturbation_non_linear_rf(
                    ys_gen,
                    cs_gen,
                    w,
                    F,
                    wstar,
                    eps_i,
                    p,
                    step_size=1e-6,
                    abs_tol=1e-6,
                    step_block=50,
                    adv_pert=adv_perturbation,
                )

                flipped = percentage_flipped_labels_NLRF(
                    ys_gen, cs_gen, w, wstar, cs_gen + adv_perturbation, F, non_linearity
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

        with open(
            os.path.join(
                data_folder,
                file_name.format(d, alpha, gamma, p, reg_param, eps_training, pstar_t),
            ),
            "wb",
        ) as f:
            pickle.dump(data, f)

        # print(
        #     f"Estimated m = {mean_m:.3f} ± {std_m:.3f} q = {mean_q:.3f} ± {std_q:.3f} rho = {mean_rho:.3f} ± {std_rho:.3f}"
        # )
