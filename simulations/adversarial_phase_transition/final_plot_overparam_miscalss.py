from linear_regression.aux_functions.percentage_flipped import (
    percentage_misclassified_direct_space,
    percentage_misclassified_linear_features,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import pickle

reps = 10
eps_training = 0.0
pstar_t = 1.0
p = "inf"
reg_param = 1e-3

data_folder_ds = "./data/direct_space"
data_folder_rf = "./data/linear_random_features"
file_name_ds = f"ERM_direct_space_perc_misclass_n_features_{{:d}}_alpha_{{:.1f}}_reps_{reps:d}_p_{p}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}_pstar_t_{pstar_t}.pkl"
file_name_rf = f"ERM_linear_rf_perc_misclass_n_features_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

img_folder = "./imgs/adversarial_phase_transition"
file_name_img = f"final_plot_overparam_misclass_reps_{reps:d}_p_{p}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}_pstar_t_{pstar_t}.pdf"

dimensions = [int(2**a) for a in range(10, 12)]

gammas = [0.5, 1.0, 1.5]
alphas = [0.5, 1.0, 2.0]
markers = [".", "x", "1", "2", "+", "3", "4"]

eps_dense = np.logspace(-1.5, 1, 100)
out = np.empty_like(eps_dense)

min_val = 1.0

plt.style.use("./plotting/latex_ready.mplstyle")
columnwidth = 469.75502
fig_width_pt = columnwidth
inches_per_pt = 1.0 / 72.27
figwidth = fig_width_pt * inches_per_pt
figheight = figwidth * ((5.0**0.5 - 1.0) / 2.0) * 0.5

fig, axs = plt.subplots(figsize=(figwidth, figheight), ncols=len(alphas))
for k, alpha in enumerate(alphas):
    ax = axs[k]
    for idx, gamma in enumerate(gammas):
        for i, (dimension, mks) in enumerate(zip(dimensions, markers)):
            with open(
                os.path.join(
                    data_folder_rf,
                    file_name_rf.format(
                        dimension, alpha, gamma, p, reg_param, eps_training, pstar_t
                    ),
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

            ax.errorbar(
                epss_g,
                np.mean(vals_g, axis=0),
                yerr=np.std(vals_g, axis=0),
                linestyle="",
                color=f"C{idx}",
                marker=mks,
                markersize=2,
                label=f"$\\gamma = $ {gamma:.1f}",
            )

        for j, eps_i in enumerate(eps_dense):
            out[j] = percentage_misclassified_linear_features(mean_m, mean_q, mean_rho, eps_i, p)

        ax.plot(eps_dense, out, color=f"C{idx}")

        min_val = min(min_val, np.min(out))

    for i, (dimension, mks) in enumerate(zip(dimensions, markers)):
        with open(os.path.join(data_folder_ds, file_name_ds.format(dimension, alpha)), "rb") as f:
            data = pickle.load(f)

            epss_g = data["epss"]
            vals_g = data["vals"]
            mean_m = data["mean_m"]
            std_m = data["std_m"]
            mean_q = data["mean_q"]
            std_q = data["std_q"]
            mean_rho = data["mean_rho"]
            std_rho = data["std_rho"]

        ax.errorbar(
            epss_g,
            np.mean(vals_g, axis=0),
            yerr=np.std(vals_g, axis=0),
            linestyle="",
            color="k",
            marker=mks,
            markersize=2,
        )

        for j, eps_i in enumerate(eps_dense):
            out[j] = percentage_misclassified_direct_space(mean_m, mean_q, mean_rho, eps_i, p)

    ax.plot(eps_dense, out, color="k")

    min_val = min(min_val, np.min(out))

    custom_lines = [
        Line2D([0], [0], color="C0", lw=2),
        Line2D([0], [0], color="C1", lw=2),
        Line2D([0], [0], color="C2", lw=2),
        Line2D([0], [0], color="k", lw=2),
    ]

    if k == 0:
        ax.legend(
            custom_lines,
            [f"$\\gamma = $ {gamma:.1f}" for gamma in gammas] + ["No Features"],
            handlelength=1,
        )

    ax.set_xscale("log")
    # ax.set_xlabel(r"$\varepsilon \, d^{\frac{1}{p} - \frac{1}{2}}$")
    ax.set_xlabel(r"$\varepsilon_g$")
    if k == 0:
        ax.set_ylabel(r"$E_{\mathrm{rob}}^{\mathrm{cns}}$", labelpad=0.0)
    ax.grid(True)
    if k != 0:
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

    ax.set_title(f"$\\alpha$ = {alpha:.1f}")
    ax.set_xlim(10 ** (-1.5), 10 ** (1.2))

for k in range(len(alphas)):
    axs[k].set_ylim(min_val, 1)

fig.set_constrained_layout(True)
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

plt.savefig(
    os.path.join(img_folder, file_name_img),
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)

plt.show()
