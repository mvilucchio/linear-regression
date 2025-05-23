import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import percentage_different_labels_estim
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_non_linear_rf,
)
from numba import njit
from tqdm.auto import tqdm
import os
import sys
from time import time
import pickle


# @njit(error_model="numpy", fastmath=True)
# def non_linearity(x):
#     return np.tanh(x)


# @njit(error_model="numpy", fastmath=True)
# def D_non_linearity(x):
#     return 1 - np.tanh(x) ** 2


@njit(error_model="numpy", fastmath=True, parallel=True)
def non_linearity(x):
    return np.divide(1, np.add(1, np.exp(-x)))


@njit(error_model="numpy", fastmath=True, parallel=True)
def D_non_linearity(x):
    return non_linearity(x) * (1 - non_linearity(x))


non_linearity_name = "logistic"


gamma, alpha, eps_training = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

# gamma, alpha, eps_training = 1.0, 1.0, 0.0

pstar_t = 1.0
reg_param = 1e-3
ps = [np.float32("inf")]
dimensions = [int(2**a) for a in range(7, 8)]
epss = np.logspace(-1, 1, 7, dtype=np.float32)
reps = 3

data_folder = "./data/non_linear_random_features"
file_name = f"ERM_non_linear_rf_perc_misclass_n_features_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}_{non_linearity_name}.pkl"

# for p in tqdm(ps, desc="p", leave=False):
for p in ps:
    print(f"p = {p}")
    # for d in tqdm(dimensions, desc="dim", leave=False):
    for d in dimensions:
        print(f"Dimension {d}")
        n_features = int(d / gamma)
        n_samples = int(d * alpha)

        if p == "inf":
            epss_rescaled = epss * (d ** (-1 / 2))
        else:
            epss_rescaled = epss * (d ** (-1 / 2 + 1 / p))

        epss_rescaled = epss_rescaled.astype(np.float32)

        vals = np.empty((reps, len(epss)))
        estim_vals_m = np.empty((reps,))
        estim_vals_q = np.empty((reps,))
        estim_vals_rho = np.empty((reps,))

        F = np.random.normal(0.0, 1.0, (d, n_features)).astype(np.float32)

        assert F.shape == (d, n_features)

        # for j in tqdm(range(reps), desc="reps", leave=False):
        for j in range(reps):
            print(f"Repetition {j + 1}/{reps}")
            cs, ys, cs_gen, ys_gen, wstar = data_generation(
                measure_gen_no_noise_clasif,
                n_features=d,
                n_samples=max(n_samples, 1),
                n_generalization=500,
                measure_fun_args={},
            )

            assert cs.shape == (n_samples, d)

            xs = non_linearity(cs @ F / np.sqrt(d))

            xs = xs.astype(np.float32)

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

            w = w.astype(np.float32)

            estim_vals_rho[j] = np.sum(wstar**2) / d
            estim_vals_m[j] = np.sum(np.dot(wstar, F @ w)) / (d * np.sqrt(n_features))
            estim_vals_q[j] = np.sum((F @ w) ** 2) / (d * n_features)

            yhat = np.sign(xs @ w)

            yhat = yhat.astype(np.float32)

            print("Computing vals matrix")

            xs_gen = non_linearity(cs_gen @ F / np.sqrt(d))
            xs_gen = xs_gen.astype(np.float32)

            yhat_gen = np.sign(xs_gen @ w)
            yhat_gen = yhat_gen.astype(np.float32)

            ys_gen = ys_gen.astype(np.float32)

            adv_pert = np.zeros_like(cs_gen)
            # for j, eps in enumerate(tqdm(epss, desc="eps", leave=False)):
            for i, eps in enumerate(epss_rescaled):
                print("starting computation for i = ", i, " eps = ", eps)
                eps = np.float32(eps)
                p = np.float32(p)
                start_time = time()
                adv_pert = find_adversarial_perturbation_non_linear_rf(
                    ys_gen,
                    cs_gen,
                    w,
                    F,
                    wstar,
                    eps,
                    p,
                    non_linearity,
                    D_non_linearity,
                    step_size=np.float32(1.0),
                    abs_tol=1e-4,
                    step_block=50,
                    max_iterations=100,
                    adv_pert=(adv_pert + np.random.normal(0, 1, adv_pert.shape)).astype(np.float32),
                    test_iters=20,
                )
                print(f"Time taken: {time() - start_time:.3f}")

                print(
                    f"Max norm adversarial perturbation: {np.max(np.linalg.norm(adv_pert, ord=p, axis=1)):.3f}\n"
                    + f"Mean norm adversarial perturbation: {np.mean(np.linalg.norm(adv_pert, ord=p, axis=1)):.3f}\n"
                    + f"Max orthogonalities: {np.max(np.abs(np.dot(adv_pert, wstar)))} eps {eps}\n"
                )

                start_time = time()
                vals[j, i] = percentage_different_labels_estim(
                    ys_gen, w, cs_gen + adv_pert, "non_linear_rf", F, non_linearity
                )
                # vals[j, i] = percentage_flipped_labels_NLRF(
                #     yhat_gen, cs_gen, w, wstar, cs_gen + adv_pert, F, d
                # )
                print(f"Time taken: {time() - start_time:.3f}")

        print("Finished computation")

        mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
        mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
        mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)

        print("Saving data")

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

        print(f"Data saved for d = {d} p = {p}")
