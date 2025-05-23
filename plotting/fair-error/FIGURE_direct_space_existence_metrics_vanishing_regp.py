import matplotlib.pyplot as plt
import os
import numpy as np
from linear_regression.aux_functions.percentage_flipped import (
    percentage_misclassified_hastie_model,
    percentage_flipped_hastie_model,
)
from linear_regression.aux_functions.misc import (
    classification_adversarial_error,
    misclassification_error_direct_space,
    flipped_error_direct_space,
    boundary_error_direct_space,
)


IMG_DIRECTORY = "./imgs"


def save_plot(fig, name, formats=["pdf"], date=True):
    for f in formats:
        fig.savefig(
            os.path.join(IMG_DIRECTORY, "{}".format(name) + "." + f),
            format=f,
        )


def set_size(width, fraction=1, subplots=(1, 1)):
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * (golden_ratio) * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


width = 458.63788

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.4)

# Define parameters
data_folder = "./data/direct_space_model_training"
reps = 10
reg_param = 1e-3
eps_min, eps_max, n_epss = 0.1, 10, 15
d = 500
reg = 2.0

alpha_list = [1.0]
different_pstars = [1.0, 2.0]

file_name_erm = f"ERM_sweep_eps_direct_model_d_{d:d}_alpha_{{:.1f}}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{{:.1f}}_reg_{reg:.1f}_regparam_{reg_param:.1e}.csv"

fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(tuple_size[0], 1.618 * tuple_size[1]),
    gridspec_kw={"hspace": 0, "wspace": 0.0},
)


eps_dense = np.logspace(-1.2, 1.2, 15)
out = np.empty_like(eps_dense)

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
linestyles = ["-", "--", ":", "-."]  # Different linestyles for different alphas

# the rows are for the different metrics in order : adv err, flipped, misclassified
# the columns are for different alphas
for i, pstar in enumerate(different_pstars):

    if pstar == 2.0:
        adv_geometry = 2.0
    if pstar == 1.0:
        adv_geometry = "inf"
    for j, alpha in enumerate(alpha_list):
        # Use different linestyle for each alpha
        current_linestyle = linestyles[j % len(linestyles)]
        marker_style = [".", ".", ".", "."][j % 4]  # Different markers for different alphas

        file_path = os.path.join(data_folder, file_name_erm.format(alpha, pstar))

        try:
            # Load CSV file instead of pickle
            with open(file_path, "r") as f:
                header = f.readline().strip().split(",")

            data_array = np.loadtxt(file_path, delimiter=",", skiprows=1)

            data = {header[i]: data_array[:, i] for i in range(len(header))}

            print(f"Loaded data from {file_path}")

            epss_g = data["eps"]
            mean_adverr = data["mean_adverr"]
            std_adverr = data["std_adverr"]
            mean_flipped = data["mean_flipped"]
            std_flipped = data["std_flipped"]
            mean_misclassified = data["mean_misclass"]
            std_misclassified = data["std_misclass"]
            mean_bound = data["mean_bound"]
            std_bound = data["std_bound"]

            # These values are now the same for all epsilon values (repeated in each row)
            # So we just take the first element
            mean_m = data["mean_m"][0]
            mean_q = data["mean_q"][0]
            mean_P = data["mean_P"][0]

            print(f"Mean m: {mean_m}, Mean q: {mean_q}, Mean P: {mean_P}")

            # Fix axis indexing to use [row, column] format and use different linestyle
            axs[0].errorbar(
                epss_g,
                mean_adverr,
                yerr=std_adverr,
                linestyle="",
                marker=marker_style,
                color=colors[i],
                alpha=0.7,
                markersize=2,
            )
            axs[1].errorbar(
                epss_g,
                mean_bound,
                yerr=std_bound,
                linestyle="",
                marker=marker_style,
                color=colors[i],
                alpha=0.7,
                markersize=2,
            )
            axs[2].errorbar(
                epss_g,
                mean_misclassified,
                yerr=std_misclassified,
                linestyle="",
                marker=marker_style,
                color=colors[i],
                alpha=0.7,
                markersize=2,
            )

            print(f"Calculating adversarial error for pstar={pstar}, alpha={alpha}")
            for k, eps_i in enumerate(eps_dense):
                out[k] = classification_adversarial_error(mean_m, mean_q, mean_P, eps_i, pstar)
            axs[0].plot(eps_dense, out, color=colors[i], linestyle=current_linestyle)

            print(f"Calculating bound error for pstar={pstar}, alpha={alpha}")
            for k, eps_i in enumerate(eps_dense):
                out[k] = boundary_error_direct_space(mean_m, mean_q, mean_P, eps_i, pstar)
            axs[1].plot(eps_dense, out, color=colors[i], linestyle=current_linestyle)

            print(f"Calculating flipped error for pstar={pstar}, alpha={alpha}")
            for k, eps_i in enumerate(eps_dense):
                out[k] = misclassification_error_direct_space(mean_m, mean_q, mean_P, eps_i, pstar)
            axs[2].plot(eps_dense, out, color=colors[i], linestyle=current_linestyle)

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

print("done")
for i in range(3):
    axs[i].set_ylabel(
        [
            r"$E_{\mathrm{rob}}$",
            r"$E_{\mathrm{bnd}}^{\mathrm{cns}}$",
            r"$E_{\mathrm{flip}}^{\mathrm{cns}}$",
        ][i]
    )
    axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)

axs[0].set_ylim(0.25, 1.05)
axs[1].set_ylim(-0.05, 0.8)
axs[2].set_ylim(0.25, 1.05)

axs[2].set_xlabel(r"$\varepsilon_g$")
axs[2].set_xscale("log")
axs[2].set_xlim(eps_min, eps_max)

plt.tight_layout()

# Update the legend to include both pstar and alpha values
legend_handles = []
for i, pstar in enumerate(different_pstars):
    if pstar == 1.0:
        pstar_label = r"$L_{\infty}$ attack"
    elif pstar == 2.0:
        pstar_label = r"$L_{2}$ attack"

    for j, alpha in enumerate(alpha_list):
        current_linestyle = linestyles[j % len(linestyles)]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=colors[i],
                lw=2,
                linestyle=current_linestyle,
                label=f"{pstar_label}",
            )
        )

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(different_pstars),
    frameon=False,
)

# You may need to adjust the top margin to accommodate the larger legend
plt.subplots_adjust(top=0.82)
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(left=0.2)
plt.subplots_adjust(right=0.95)

save_plot(
    fig,
    "direct_sapce_model_eps_sweep_vanishing_regp_alpha_{:.1f}".format(alpha),
    formats=["pdf", "png"],
)

plt.show()
