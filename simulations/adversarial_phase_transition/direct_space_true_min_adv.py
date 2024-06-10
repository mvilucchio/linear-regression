import matplotlib.pyplot as plt
import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import percentage_flipped_labels
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.erm.erm_solvers import find_adversarial_perturbation_direct_space
from linear_regression.aux_functions.percentage_flipped import percentage_flipped_direct_space
from tqdm.auto import tqdm
import pickle
import os


alpha = 1.5
reg_param = 1.0
eps_training = 0.0
pstar_t = 1.0
ps = [2, 3, "inf"]
dimensions = [int(2**a) for a in range(10, 11)]

epss = np.logspace(-2, 2, 15)
eps_dense = np.logspace(-2, 2, 100)
reps = 10
run_experiment = True

colors = [f"C{i}" for i in range(len(dimensions))]
linestyles = ["-", "--", "-.", ":"]
markers = [".", "x", "1", "2", "+", "3", "4"]

assert len(linestyles) >= len(ps)
assert len(markers) >= len(ps)

data_folder = "./data"
file_name = f"ERM_direct_space_adv_transition_n_features_{{:d}}_alpha_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

for p, ls, mrk in zip(tqdm(ps, desc="p", leave=False), linestyles, markers):
    for n_features, c in zip(tqdm(dimensions, desc="n", leave=False), colors):
        if run_experiment:
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

                if eps_training == 0.0:
                    w = find_coefficients_Logistic(ys, xs, reg_param)
                else:
                    w = find_coefficients_Logistic_adv(
                        ys, xs, reg_param, eps_training, 2.0, pstar_t, wstar
                    )

                estim_vals_rho[j] = np.sum(wstar**2) / n_features
                estim_vals_m[j] = np.sum(wstar * w) / n_features
                estim_vals_q[j] = np.sum(w**2) / n_features

                yhat = np.sign(xs_gen @ w)

                for i, eps_i in enumerate(tqdm(epss_rescaled, desc="eps", leave=False)):
                    adv_perturbation = find_adversarial_perturbation_direct_space(
                        yhat, xs_gen, w, wstar, eps_i, p
                    )

                    flipped = percentage_flipped_labels(
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
                    data_folder,
                    file_name.format(n_features, alpha, p, reg_param, eps_training, pstar_t),
                ),
                "wb",
            ) as f:
                pickle.dump(data, f)

        with open(
            os.path.join(
                data_folder,
                file_name.format(n_features, alpha, p, reg_param, eps_training, pstar_t),
            ),
            "rb",
        ) as f:
            data = pickle.load(f)
            epss = data["epss"]
            vals = data["vals"]
            estim_vals_m = data["estim_vals_m"]
            estim_vals_q = data["estim_vals_q"]
            estim_vals_rho = data["estim_vals_rho"]
            mean_m = data["mean_m"]
            std_m = data["std_m"]
            mean_q = data["mean_q"]
            std_q = data["std_q"]
            mean_rho = data["mean_rho"]

        plt.errorbar(
            epss,
            np.mean(vals, axis=0),
            yerr=np.std(vals, axis=0),
            marker=mrk,
            linestyle="None",
            color=c,
        )

    out = np.empty_like(eps_dense)

    for i, eps_i in enumerate(eps_dense):
        out[i] = percentage_flipped_direct_space(mean_m, mean_q, mean_rho, eps_i, p)

    plt.plot(eps_dense, out, linestyle=ls, color="black", linewidth=0.5)

plt.title(f"Direct space $\\alpha$ = {alpha:.1f} $\\lambda$ = {reg_param:.1e}")
plt.xscale("log")
plt.xlabel(r"$\epsilon (\sqrt[p]{d} / \sqrt{d})$")
plt.ylabel("Percentage of flipped labels")
plt.grid()

handles = []
labels = []
for p, ls, mrk in zip(ps, linestyles, markers):
    handle = plt.Line2D([], [], linestyle=ls, linewidth=0.5, marker=mrk, color="black")
    handles.append(handle)
    labels.append(f"p = {p}")

for dim, c in zip(dimensions, colors):
    handle_dim = plt.Line2D([], [], linestyle="None", marker="o", color=c)
    handles.append(handle_dim)
    labels.append(f"d = {dim:d}")

plt.legend(handles, labels)

plt.show()
