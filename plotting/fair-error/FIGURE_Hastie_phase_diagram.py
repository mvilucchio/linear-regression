import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

# Import necessary functions for theory curves
from linear_regression.aux_functions.percentage_flipped import (
    percentage_misclassified_hastie_model,
    percentage_flipped_hastie_model,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.hastie_model_pstar_attacks import (
    f_hastie_L2_reg_Linf_attack,
    q_latent_hastie_L2_reg_Linf_attack,
    q_features_hastie_L2_reg_Linf_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm_hastie import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
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

tuple_size = set_size(width, fraction=0.50)


# Define parameters
data_folder = "./data/hastie_model_optimal_values_pd"
alpha_min, alpha_max, n_alphas = 0.65, 1.2, 50
gamma_min, gamma_max, n_gammas = 0.7, 1.2, 50
# alpha_min, alpha_max, n_alphas = 0.05, 1.5, 40
# gamma_min, gamma_max, n_gammas = 0.7, 1.5, 40

metric_name = "bound"
# File name templates
file_name_both = "PD_optimal_both_{}_alphas_{:.2f}_{:.2f}_{:d}_gammas_{:.2f}_{:.2f}_{:d}_2.csv"
file_name_just_regp = "PD_optimal_reg_p_{}_alphas_{:.2f}_{:.2f}_{:d}_gammas_{:.2f}_{:.2f}_{:d}.csv"

file_name = file_name_both

fig, axs = plt.subplots(1, 3, sharex=True, figsize=(3.5 * tuple_size[0], 2.5 * 0.7 * tuple_size[0]))

try:
    file_name = file_name.format(
        metric_name, alpha_min, alpha_max, n_alphas, gamma_min, gamma_max, n_gammas
    )
    data = np.loadtxt(os.path.join(data_folder, file_name), delimiter=",", skiprows=1)

    alphas = data[:, 0]
    gammas = data[:, 1]
    flipped_percentages_adv = data[:, 2]
    optimal_reg_param = data[:, 3]
    optimal_eps_training = data[:, 4]

    # reshape the data into n_alphas x n_gammas
    flipped_percentages_adv = flipped_percentages_adv.reshape(n_alphas, n_gammas)
    optimal_reg_param = optimal_reg_param.reshape(n_alphas, n_gammas)
    optimal_eps_training = optimal_eps_training.reshape(n_alphas, n_gammas)
    ALPHAS = alphas.reshape(n_alphas, n_gammas)
    GAMMAS = gammas.reshape(n_alphas, n_gammas)

except (FileNotFoundError, IOError):
    print(f"Error loading file {file_name}. ")

# Different colormaps for each plot
cmap1 = "viridis"
cmap2 = "plasma"
cmap3 = "inferno"

# First plot with viridis colormap
contour0 = axs[0].contourf(
    ALPHAS,
    GAMMAS,
    flipped_percentages_adv.T,
    cmap=cmap1,
)
axs[0].set_title(r"$E_{\mathrm{flip}}^{\mathrm{true}}$")
axs[0].set_ylabel(r"$\gamma$")
axs[0].set_xlabel(r"$\alpha$")
# axs[0].set_xscale("log")
cbar0 = fig.colorbar(contour0, ax=axs[0], pad=0.01)

# Second plot with plasma colormap
contour1 = axs[1].contourf(
    ALPHAS,
    GAMMAS,
    optimal_reg_param.T,
    cmap=cmap2,
)
axs[1].set_title(r"$\lambda_{\mathrm{opt}}$")
axs[1].set_ylabel(r"$\gamma$")
axs[1].set_xlabel(r"$\alpha$")
# axs[1].set_xscale("log")
cbar1 = fig.colorbar(contour1, ax=axs[1], pad=0.01)

# Third plot with inferno colormap
contour2 = axs[2].contourf(
    ALPHAS,
    GAMMAS,
    optimal_eps_training.T,
    cmap=cmap1,
    levels=np.linspace(0, 0.5, 20),
)
axs[2].set_title(r"$\epsilon_{\mathrm{opt}}$")
axs[2].set_ylabel(r"$\gamma$")
axs[2].set_xlabel(r"$\alpha$")
# axs[2].set_xscale("log")
cbar2 = fig.colorbar(contour2, ax=axs[2], pad=0.01)


# Add some spacing between subplots to accommodate the colorbars
plt.tight_layout()

plt.show()

# Create a new figure for line plots
fig2, axs2 = plt.subplots(1, 3, figsize=(2.5 * tuple_size[0], 1.2 * tuple_size[0]))

# Choose middle alpha value
middle_alpha_idx = n_alphas // 2
fixed_alpha = ALPHAS[middle_alpha_idx, 0]

# Extract data for the central column (fixed alpha)
gammas_line = GAMMAS[:, 0]  # All gamma values
flip_at_fixed_alpha = flipped_percentages_adv[middle_alpha_idx, :]
reg_param_at_fixed_alpha = optimal_reg_param[middle_alpha_idx, :]
eps_at_fixed_alpha = optimal_eps_training[middle_alpha_idx, :]

# Plot 1: Flipped percentages
axs2[0].plot(gammas_line, flip_at_fixed_alpha, "o-", color="blue", linewidth=2)
axs2[0].set_title(r"$E_{\mathrm{flip}}^{\mathrm{true}}$ at $\alpha=" + f"{fixed_alpha:.2f}$")
axs2[0].set_xlabel(r"$\gamma$")
axs2[0].set_ylabel(r"$E_{\mathrm{flip}}^{\mathrm{true}}$")
axs2[0].grid(True, alpha=0.3)

# Plot 2: Optimal regularization parameter
axs2[1].plot(gammas_line, reg_param_at_fixed_alpha, "o-", color="purple", linewidth=2)
axs2[1].set_title(r"$\lambda_{\mathrm{opt}}$ at $\alpha=" + f"{fixed_alpha:.2f}$")
axs2[1].set_xlabel(r"$\gamma$")
axs2[1].set_ylabel(r"$\lambda_{\mathrm{opt}}$")
axs2[1].grid(True, alpha=0.3)

# Plot 3: Optimal epsilon training
axs2[2].plot(gammas_line, eps_at_fixed_alpha, "o-", color="red", linewidth=2)
axs2[2].set_title(r"$\epsilon_{\mathrm{opt}}$ at $\alpha=" + f"{fixed_alpha:.2f}$")
axs2[2].set_xlabel(r"$\gamma$")
axs2[2].set_ylabel(r"$\epsilon_{\mathrm{opt}}$")
axs2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# # Save the plots if needed
save_plot(fig, "hastie_phase_diagram_contour", formats=["pdf", "png"])
save_plot(fig2, "hastie_phase_diagram_line_plots", formats=["pdf", "png"])
