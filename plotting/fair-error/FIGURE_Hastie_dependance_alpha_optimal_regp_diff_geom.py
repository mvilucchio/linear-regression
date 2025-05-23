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


def compute_theory_overlaps(reg_param, eps_train, alpha, gamma, init_cond):
    f_kwargs = {"reg_param": reg_param, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": eps_train}

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_hastie_L2_reg_Linf_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-6,
    )

    m_hat, q_hat, V_hat, P_hat = f_hat_Logistic_no_noise_Linf_adv_classif(
        m_se, q_se, V_se, P_se, eps_train, alpha, gamma
    )

    q_latent_se = q_latent_hastie_L2_reg_Linf_attack(m_hat, q_hat, V_hat, P_hat, reg_param, gamma)
    q_features_se = q_features_hastie_L2_reg_Linf_attack(
        m_hat, q_hat, V_hat, P_hat, reg_param, gamma
    )

    return m_se, q_se, q_latent_se, q_features_se, V_se, P_se


width = 458.63788

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

# Define parameters SE
alpha_min, alpha_max, n_alphas_se, gammas = 0.2, 2.5, 50, [0.5, 1.0, 1.5]

pstar_t = 1.0
reg_t = 2.0

attack_pstar_list = [1.0, 2.0]
attack_labels = [r"$L_{\infty}$ attack", "$L_2$ attack"]

data_folder = "./data/hastie_model_training_optimal_diff_geom"

file_name_sweep_alpha_misclass = f"SE_optimal_regp_misclass_gamma_{{:.2f}}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_pstar_{pstar_t:.1f}_{{:.1f}}_reg_{reg_t:.1f}.csv"
# file_name_sweep_alpha_flipped = f"SE_optimal_regp_flipped_gamma_{{:.2f}}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_pstar_{pstar:.1f}_{{:.1f}}_reg_{reg:.1f}.csv"
file_name_sweep_alpha_bound = f"SE_optimal_regp_bound_gamma_{{:.2f}}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_pstar_{pstar_t:.1f}_{{:.1f}}_reg_{reg_t:.1f}.csv"
file_name_sweep_alpha_adverr = f"SE_optimal_regp_adverr_gamma_{{:.2f}}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_pstar_{pstar_t:.1f}_{{:.1f}}_reg_{reg_t:.1f}.csv"

# Define parameters ERM
d = 500
reps = 10
alpha_min_erm, alpha_max_erm, n_alphas_erm = max(0.5, alpha_min), min(5.0, alpha_max), 10
delta = 0.0

file_name_sweep_alpha_erm_misclass = f"ERM_optimal_regp_misclass_gamma_{{:.2f}}_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar_t:.1f}_{{:.1f}}_reg_{reg_t:.1f}.csv"
# file_name_sweep_alpha_erm_flipped = f"ERM_optimal_regp_flipped_gamma_{{:.2f}}_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar:.1f}_{{:.1f}}_reg_{reg:.1f}.csv"
file_name_sweep_alpha_erm_bound = f"ERM_optimal_regp_bound_gamma_{{:.2f}}_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar_t:.1f}_{{:.1f}}_reg_{reg_t:.1f}.csv"
file_name_sweep_alpha_erm_adverr = f"ERM_optimal_regp_adverr_gamma_{{:.2f}}_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar_t:.1f}_{{:.1f}}_reg_{reg_t:.1f}.csv"

# ---------------------------------------------------------------------------- #
#                                 alpha sweeps                                 #
# ---------------------------------------------------------------------------- #

print("Plotting alpha sweeps...")
fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=(tuple_size[0], tuple_size[0]), gridspec_kw={"hspace": 0}
)

# First adjust the top margin to make space for the legends
fig.subplots_adjust(left=0.20)
fig.subplots_adjust(bottom=0.12)
fig.subplots_adjust(top=0.9)  # Reduced from 0.92 to make room for legends
fig.subplots_adjust(right=0.96)

line_styles = ["-", "--"]  # solid for pstar=1, dashed for pstar=2

for i, gamma in enumerate(gammas):
    for k, pstar_g in enumerate(attack_pstar_list):
        # State evolution
        file_path = os.path.join(data_folder, file_name_sweep_alpha_misclass.format(gamma, pstar_g))

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)

            alphas_se = data[:, 0]
            misclas_fair = data[:, 11]
            reg_param = data[:, -1]

            axs[2].plot(alphas_se, misclas_fair, color=f"C{i}", linestyle=line_styles[k])

        except (FileNotFoundError, IOError):
            print(f"SE data file not found: {file_path}. Skipping...")

        file_path = os.path.join(data_folder, file_name_sweep_alpha_bound.format(gamma, pstar_g))

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)

            alphas_se = data[:, 0]
            flipped_fair = data[:, 12]
            reg_param = data[:, -1]

            axs[1].plot(alphas_se, flipped_fair, color=f"C{i}", linestyle=line_styles[k])

        except (FileNotFoundError, IOError):
            print(f"SE data file not found: {file_path}. Skipping...")

        file_path = os.path.join(data_folder, file_name_sweep_alpha_adverr.format(gamma, pstar_g))

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)

            alphas_se = data[:, 0]
            adv_err = data[:, 8]
            reg_param = data[:, -1]

            label = f"$\\gamma = $ {gamma:.1f}" if k == 0 else None
            axs[0].plot(alphas_se, adv_err, color=f"C{i}", linestyle=line_styles[k], label=label)

        except (FileNotFoundError, IOError):
            print(f"SE data file not found: {file_path}. Skipping...")

        # ERM
        file_path = os.path.join(
            data_folder, file_name_sweep_alpha_erm_misclass.format(gamma, pstar_g)
        )

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)

            gammas = data[:, 0]
            misclas_fairs_mean, misclas_fairs_std = data[:, 17], data[:, 18]

            marker = "o" if k == 0 else "s"
            axs[2].errorbar(
                gammas,
                misclas_fairs_mean,
                yerr=misclas_fairs_std,
                color=f"C{i}",
                fmt=marker,
            )

        except (FileNotFoundError, IOError):
            print(f"File {file_path} does not exist. Skipping...")

        file_path = os.path.join(
            data_folder, file_name_sweep_alpha_erm_bound.format(gamma, pstar_g)
        )

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)

            gammas = data[:, 0]
            flipped_fairs_mean, flipped_fairs_std = data[:, 15], data[:, 16]

            marker = "o" if k == 0 else "s"
            axs[1].errorbar(
                gammas,
                flipped_fairs_mean,
                yerr=flipped_fairs_std,
                color=f"C{i}",
                fmt=marker,
            )

        except (FileNotFoundError, IOError):
            print(f"File {file_path} does not exist. Skipping...")

        file_path = os.path.join(
            data_folder, file_name_sweep_alpha_erm_adverr.format(gamma, pstar_g)
        )

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)

            gammas = data[:, 0]
            adv_errors_mean, adv_errors_std = data[:, 13], data[:, 14]

            marker = "o" if k == 0 else "s"
            axs[0].errorbar(
                gammas,
                adv_errors_mean,
                yerr=adv_errors_std,
                color=f"C{i}",
                fmt=marker,
            )

        except (FileNotFoundError, IOError):
            print(f"File {file_path} does not exist. Skipping...")

# Create custom legend
from matplotlib.lines import Line2D

# Create legend handles for gamma values
gamma_handles = [Line2D([0], [0], color=f"C{i}", lw=2) for i, gamma in enumerate(gammas)]
gamma_labels = [f"$\\gamma = {gamma:.1f}$" for gamma in gammas]

# Create legend handles for pstar values
pstar_handles = [Line2D([0], [0], color="black", lw=2, linestyle=ls) for ls in line_styles]
pstar_labels = [f"$p^* = {pstar}$" for pstar in attack_pstar_list]

# Create a legend for gammas on the first row (moved down slightly)
gamma_legend = axs[0].legend(
    handles=gamma_handles,
    labels=gamma_labels,
    bbox_to_anchor=(0.5, 1.05),  # Adjusted from 1.15
    loc="lower center",
    ncols=len(gammas),
    labelspacing=0.0,
    handlelength=1.5,
    handletextpad=0.5,
    columnspacing=0.8,
    fontsize=8,
    frameon=False,
)

# Add the legend to the figure
axs[0].add_artist(gamma_legend)

# Create a second legend for pstar values on the second row
axs[0].legend(
    handles=pstar_handles,
    labels=pstar_labels,
    bbox_to_anchor=(0.5, 0.9),  # Adjusted from 1.05
    loc="lower center",
    ncols=len(attack_pstar_list),
    labelspacing=0.0,
    handlelength=1.5,
    handletextpad=0.5,
    columnspacing=0.8,
    fontsize=8,
    frameon=False,
)

axs[2].set_xlabel(f"$\\alpha = n / d$")
axs[2].set_xlim([0.3, 2.5])
axs[0].set_ylabel(r"$E_{\mathrm{rob}}$")
axs[1].set_ylabel(r"$E_{\mathrm{bnd}}$")
axs[2].set_ylabel(r"$E_{\mathrm{rob}}^{\mathrm{true}}$")
for ax in axs:
    ax.grid(which="both", alpha=0.7)

save_plot(fig, "hastie_model_alpha_sweep_optimal_regp", formats=["pdf", "png"])

plt.show()
