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
from linear_regression.erm.erm_solvers import find_adversarial_perturbation_RandomFeatures_space
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_linear_features_space_true_min,
)
from tqdm.auto import tqdm
import os
import pickle


alpha = 0.5
gamma = 1.5

eps_training = 0.0
pstar_t = 1.0
reg_param = 1.0
ps = [2, 3, "inf"]
dimensions = [int(2**a) for a in range(9, 12)]
epss = np.logspace(-1, 1, 15)
eps_dense = np.logspace(-1, 1, 100)
reps = 10

run_experiment = False

colors = [f"C{i}" for i in range(len(dimensions))]
linestyles = ["-", "--", "-.", ":"]
markers = [".", "x", "1", "2", "+", "3", "4"]

assert len(linestyles) >= len(ps)
assert len(markers) >= len(ps)

data_folder = "./data"
img_folder = "./imgs"
file_name = f"ERM_direct_space_adv_transition_n_features_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"
img_name = f"random_linear_features_alpha_{alpha:.2f}_gamma_{gamma:.2f}_ps_{*ps,}.png"

for p, ls, mrk in zip(tqdm(ps, desc="p"), linestyles, markers):
    for n_hidden_features, c in zip(tqdm(dimensions, desc="dim", leave=False), colors):
        if run_experiment:
            n_features = int(n_hidden_features / gamma)
            n_samples = int(n_features * alpha)

            if p == "inf":
                epss_rescaled = epss * (n_features ** (-1 / 2))
            else:
                epss_rescaled = epss * (n_features ** (-1 / 2 + 1 / p))

            # print(
            #     f"n_features = {n_features:d} n_samples = {n_samples:d} n_hidden_features = {n_hidden_features:d}\n"
            # )

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
                    w = find_coefficients_Logistic_adv(
                        ys, xs, reg_param, eps_training, 2.0, pstar_t, wstar
                    )

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

                    flipped = percentage_flipped_labels_estim(
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

            # print(
            #     f"Estimated m = {mean_m:.3f} ± {std_m:.3f} q = {mean_q:.3f} ± {std_q:.3f} rho = {mean_rho:.3f} ± {std_rho:.3f}"
            # )

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
                    file_name.format(
                        n_hidden_features, alpha, gamma, p, reg_param, eps_training, pstar_t
                    ),
                ),
                "wb",
            ) as f:
                pickle.dump(data, f)

        with open(
            os.path.join(
                data_folder,
                file_name.format(
                    n_hidden_features, alpha, gamma, p, reg_param, eps_training, pstar_t
                ),
            ),
            "rb",
        ) as f:
            data = pickle.load(f)
            epss = data["epss"]
            vals = data["vals"]
            mean_m = data["mean_m"]
            std_m = data["std_m"]
            mean_q = data["mean_q"]
            std_q = data["std_q"]
            mean_rho = data["mean_rho"]
            std_rho = data["std_rho"]

        plt.errorbar(
            epss,
            np.mean(vals, axis=0),
            yerr=np.std(vals, axis=0),
            linestyle="",
            color=c,
            marker=mrk,
            # label=f"p = {p}",
        )

    out = np.empty_like(eps_dense)

    for i, eps_i in enumerate(eps_dense):
        out[i] = percentage_flipped_linear_features_space_true_min(
            mean_m, mean_q, mean_rho, eps_i, p, gamma
        )

    plt.plot(eps_dense, out, linestyle=ls, color="black", linewidth=0.5)

plt.title(
    f"Random Features $\\alpha$ = {alpha:.1f} $\\gamma$ = {gamma:.1f} $\\lambda$ = {reg_param:.1e}"
)
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

plt.savefig(os.path.join(img_folder, img_name), format="png")

plt.show()
