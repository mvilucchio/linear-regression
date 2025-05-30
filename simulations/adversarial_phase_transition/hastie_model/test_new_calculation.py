import numpy as np
import matplotlib.pyplot as plt
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
    find_adversarial_perturbation_linear_rf_new,
)
from scipy.special import erf
from tqdm.auto import tqdm
import sys
import warnings

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
        0.3,
        0.2,
        1e-2,
        0.2,
    )

# DO NOT CHANGE, NOT IMPLEMENTED FOR OTHERS
pstar_t = 1.0

dimensions = [int(2**a) for a in range(9, 10)]
reps = 5

epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

data_folder = "./data/hastie_model_training"
file_name = f"ERM_flipped_Hastie_Linf_d_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}.pkl"

for d in tqdm(dimensions, desc="dim", leave=False):
    p = int(d / gamma)
    n = int(d * alpha)

    print(f"p: {p}, d: {d}, n: {n}")

    # works for p = "inf"
    # epss_rescaled = epss * (d ** (-1 / 2))
    epss_rescaled = epss / np.sqrt(d)

    vals = np.empty((reps, len(epss)))
    estim_vals_m = np.empty((reps,))
    estim_vals_q = np.empty((reps,))
    estim_vals_q_latent = np.empty((reps,))
    estim_vals_q_feature = np.empty((reps,))
    estim_vals_rho = np.empty((reps,))
    estim_vals_P = np.empty((reps,))

    j = 0
    # for j in tqdm(range(reps), desc="reps", leave=False):
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
            # if eps_training == 0.0:
            #     w = find_coefficients_Logistic(ys, xs, reg_param)
            # else:
            #     w = find_coefficients_Logistic_adv(
            #         ys, xs, 0.5 * reg_param, eps_training, 2.0, pstar_t, F @ wstar
            #     )
            w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_training)
            print("j", j)
        except (ValueError, UserWarning) as e:
            print(f"Error in finding coefficients {j}:", e)
            continue

        estim_vals_rho[j] = np.sum(wstar**2) / d
        estim_vals_m[j] = np.dot(wstar, F.T @ w) / (d)
        estim_vals_q[j] = np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p
        estim_vals_q_latent[j] = np.dot(F.T @ w, F.T @ w) / d
        estim_vals_q_feature[j] = np.dot(w, w) / p
        estim_vals_P[j] = np.mean(np.abs(w))

        yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)

        yhat_gen = np.sign(np.dot(xs_gen, w))

        i = 0
        while i < len(epss_rescaled):
            eps_i = epss_rescaled[i]
            try:
                adv_perturbation = find_adversarial_perturbation_linear_rf(
                    yhat_gen, zs_gen, w, F.T, wstar, eps_i, "inf"
                )
            except (ValueError, UserWarning) as e:
                print("Error in finding adversarial perturbation:", e)
                vals[j, i] = np.nan
                # i += 1
                continue

            flipped = np.mean(
                yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )

            vals[j, i] = flipped
            i += 1

        j += 1

    mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
    mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
    mean_q_latent, std_q_latent = np.mean(estim_vals_q_latent), np.std(estim_vals_q_latent)
    mean_q_feature, std_q_feature = np.mean(estim_vals_q_feature), np.std(estim_vals_q_feature)
    mean_P, std_P = np.mean(estim_vals_P), np.std(estim_vals_P)
    mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)
    mean_flipped, std_flipped = np.mean(vals, axis=0), np.std(vals, axis=0)

    print(
        f"m = {mean_m:.4f} ± {std_m:.4f}, q = {mean_q:.4f} ± {std_q:.4f},\n"
        f"q_latent = {mean_q_latent:.4f} ± {std_q_latent:.4f}, q_feature = {mean_q_feature:.4f} ± {std_q_feature:.4f},\n"
        f"rho = {mean_rho:.4f} ± {std_rho:.4f}, P = {mean_P:.4f} ± {std_P:.4f},\n"
    )

    plt.errorbar(epss, mean_flipped, yerr=std_flipped, linestyle="", marker=".", label=f"$d = {d}$")

if gamma < 1:
    plt.plot(
        epss,
        erf(
            epss
            * np.sqrt(mean_q_latent - mean_m**2)
            * np.sqrt(1 / np.pi)
            / np.sqrt(mean_q)
            * np.sqrt(gamma)
        ),
        label="theoretical gamma < 1",
        linestyle="--",
    )
else:
    plt.plot(
        epss,
        erf(
            epss
            * np.sqrt(mean_q_feature - mean_m**2)
            / np.sqrt(gamma)
            * np.sqrt(1 / np.pi)
            / np.sqrt(mean_q)
        ),
        label="theoretical gamma > 1",
        linestyle="--",
    )

plt.title(f"gamma = {gamma:.2f}")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\mathbb{P}(\hat{y} \neq y)$")
plt.xscale("log")
plt.yscale("log")
plt.grid(which="both")
plt.legend()
plt.show()
