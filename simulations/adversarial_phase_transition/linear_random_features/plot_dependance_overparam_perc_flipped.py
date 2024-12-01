from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_linear_features,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import pickle

alphas = [0.5, 1.0, 2.0]
gammas = [0.5, 1.0, 1.5]
reps = 10
eps_training = 0.0
pstar_t = 1.0
p = "inf"
reg_param = 1e-3

dimensions = [int(2**a) for a in range(10, 12)]

data_folder = "./data/linear_random_features"
file_name = f"ERM_linear_rf_perc_flipped_n_features_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

markers = [".", "x", "1", "2", "+", "3", "4"]

eps_dense = np.logspace(-1.5, 1.5, 100)
out = np.empty_like(eps_dense)

plt.figure(figsize=(5 * len(alphas) - 2, 5))

for k, alpha in enumerate(alphas):
    plt.subplot(1, len(alphas), k + 1)
    for idx, gamma in enumerate(gammas):

        for i, (dimension, mks) in enumerate(zip(dimensions, markers)):
            with open(
                os.path.join(
                    data_folder,
                    file_name.format(dimension, alpha, gamma, p, reg_param, eps_training, pstar_t),
                ),
                "rb",
            ) as f:
                data = pickle.load(f)

                epss_g = data["epss"]
                vals_g = data["vals"]
                mean_m = data["mean_m"]
                std_m = data["std_m"]
                mean_q = data["mean_q"]
                std_q = data["std_q"]
                mean_rho = data["mean_rho"]
                std_rho = data["std_rho"]

            plt.errorbar(
                epss_g,
                np.mean(vals_g, axis=0),
                yerr=np.std(vals_g, axis=0),
                linestyle="",
                color=f"C{idx}",
                marker=mks,
                label=f"$\\gamma = $ {gamma:.1f}",
            )

        for j, eps_i in enumerate(eps_dense):
            out[j] = percentage_flipped_linear_features(mean_m, mean_q, mean_rho, eps_i, p)

        plt.plot(eps_dense, out, color=f"C{idx}")

    custom_lines = [
        Line2D([0], [0], color="C0", lw=4),
        Line2D([0], [0], color="C1", lw=4),
        Line2D([0], [0], color="C2", lw=4),
    ]

    plt.legend(custom_lines, [f"$\\gamma = $ {gamma:.1f}" for gamma in gammas])

    plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    plt.xlabel(r"$\epsilon_g (\sqrt[p]{d} / \sqrt{d})$")
    plt.ylabel("Percentage of flipped labels")
    plt.grid(which="both")
    plt.title(f"$\\alpha$ = {alpha:.1f} $\\epsilon_t = {eps_training:.2f}$")

plt.suptitle(f"Linear Random Features p = {p} $\\lambda$ = {reg_param:.1e}")
plt.tight_layout()

plt.show()
