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

# Define parameters SE
alpha_min, alpha_max, n_alphas_se = 0.2, 2.5, 30
pstar_reg_pairs = [(1.0, 1.0)]

data_folder = "./data/direct_space_model_training_optimal_new"

file_name_sweep_alpha_misclass_regp = f"SE_optimal_regp_misclass_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_sweep_alpha_bound_regp = f"SE_optimal_regp_bound_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_sweep_alpha_adverr_regp = f"SE_optimal_regp_adverr_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"

alpha_min_opt, alpha_max_opt, n_alphas_se_opt = 0.05, 2.5, 60

file_name_sweep_alpha_misclass_robtrain = f"SE_optimal_misclass_direct_alphas_{alpha_min_opt:.1f}_{alpha_max_opt:.1f}_{n_alphas_se_opt:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_sweep_alpha_adverr_robtrain = f"SE_optimal_adverr_direct_alphas_{alpha_min_opt:.1f}_{alpha_max_opt:.1f}_{n_alphas_se_opt:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_sweep_alpha_bound_robtrain = f"SE_optimal_bound_direct_alphas_{alpha_min_opt:.1f}_{alpha_max_opt:.1f}_{n_alphas_se_opt:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"


# Define parameters ERM
d = 500
reps = 10
alpha_min_erm, alpha_max_erm, n_alphas_erm = max(0.5, alpha_min), min(5.0, alpha_max), 10
delta = 0.0

file_name_misclass_regp = f"ERM_optimal_regp_misclass_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_adverr_regp = f"ERM_optimal_regp_adverr_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_bound_regp = f"ERM_optimal_regp_bound_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"

file_name_misclass_robtrain = f"ERM_optimal_misclass_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_adverr_robtrain = f"ERM_optimal_adverr_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"
file_name_bound_robtrain = f"ERM_optimal_bound_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{{:.1f}}_reg_{{:.1f}}.csv"

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=(tuple_size[0], tuple_size[0]), gridspec_kw={"hspace": 0}
)
fig.subplots_adjust(left=0.20)
fig.subplots_adjust(bottom=0.12)
fig.subplots_adjust(top=0.92)
fig.subplots_adjust(right=0.96)


for pstar, reg_param in pstar_reg_pairs:
    print(f"pstar: {pstar}, reg_param: {reg_param}")
    # ---------------------------------------------------------------------------- #
    #                                State evolution                               #
    # ---------------------------------------------------------------------------- #

    # ------------------------------ optimal lambda ------------------------------ #
    file_path = os.path.join(
        data_folder, file_name_sweep_alpha_misclass_regp.format(pstar, reg_param)
    )

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_se = data[:, 0]
        misclas_fair = data[:, 11]
        loaded_reg_param = data[:, -1]

        axs[2].plot(alphas_se, misclas_fair, color="k")
        # axs[2].plot(alphas_se, loaded_reg_param, "--", color="k")

    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    file_path = os.path.join(data_folder, file_name_sweep_alpha_bound_regp.format(pstar, reg_param))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_se = data[:, 0]
        bound_err = data[:, 12]
        loaded_reg_param = data[:, -1]

        axs[1].plot(alphas_se, bound_err, color="k")
        # axs[1].plot(alphas_se, loaded_reg_param, "--", color="k")

    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    file_path = os.path.join(
        data_folder, file_name_sweep_alpha_adverr_regp.format(pstar, reg_param)
    )

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_se = data[:, 0]
        adv_err = data[:, 8]
        loaded_reg_param = data[:, -1]

        axs[0].plot(alphas_se, adv_err, color="k")
        # axs[0].plot(alphas_se, loaded_reg_param, "--", color="k")

    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    # ------------------------- optimal lambda and eps_t ------------------------- #

    # file_name_sweep_alpha_misclass_robtrain = f"SE_optimal_misclass_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_50_pstar_{{:.1f}}_reg_{{:.1f}}.csv"

    file_path = os.path.join(
        data_folder, file_name_sweep_alpha_misclass_robtrain.format(pstar, reg_param)
    )

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        alphas_se = data[:, 0]
        misclas_fair = data[:, 11]
        loaded_reg_param = data[:, -2]
        loaded_eps_t = data[:, -1]

        loaded_m = data[:, 1]
        loaded_q = data[:, 2]

        axs[2].plot(alphas_se, misclas_fair, "--", color="b")
        # axs[2].plot(alphas_se, loaded_reg_param, "--", color="k")
        # axs[2].plot(alphas_se, loaded_eps_t, "--", color="r")
    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    # file_name_sweep_alpha_bound_robtrain = f"SE_optimal_bound_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_50_pstar_{{:.1f}}_reg_{{:.1f}}.csv"

    file_path = os.path.join(
        data_folder, file_name_sweep_alpha_bound_robtrain.format(pstar, reg_param)
    )
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        alphas_se = data[:, 0]
        bound_err = data[:, 12]
        loaded_reg_param = data[:, -2]
        loaded_eps_t = data[:, -1]

        loaded_m = data[:, 1]
        loaded_q = data[:, 2]

        axs[1].plot(alphas_se, bound_err, "--", color="b")
        # axs[1].plot(alphas_se, np.sqrt(loaded_q - loaded_m**2), "--", color="g")
        # axs[1].plot(alphas_se, loaded_reg_param, "--", color="k")
        # axs[1].plot(alphas_se, loaded_eps_t, "--", color="r")
    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    # file_name_sweep_alpha_adverr_robtrain = f"SE_optimal_adverr_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_50_pstar_{{:.1f}}_reg_{{:.1f}}.csv"

    file_path = os.path.join(
        data_folder, file_name_sweep_alpha_adverr_robtrain.format(pstar, reg_param)
    )
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        alphas_se = data[:, 0]
        adv_err = data[:, 8]
        loaded_reg_param = data[:, -2]
        loaded_eps_t = data[:, -1]
        axs[0].plot(alphas_se, adv_err, "--", color="b")
        # axs[0].plot(alphas_se, loaded_reg_param, "--", color="k")
        # axs[0].plot(alphas_se, loaded_eps_t, "--", color="r")
    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    # ---------------------------------------------------------------------------- #
    #                                      ERM                                     #
    # ---------------------------------------------------------------------------- #
    # ------------------------------ optimal lambda ------------------------------ #
    file_path = os.path.join(data_folder, file_name_misclass_regp.format(pstar, reg_param))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_erm = data[:, 0]
        misclas_fairs_mean, misclas_fairs_std = data[:, 13], data[:, 14]

        axs[2].errorbar(
            alphas_erm,
            misclas_fairs_mean,
            yerr=misclas_fairs_std,
            color="k",
            fmt=".",
            linestyle="",
        )

    except (FileNotFoundError, IOError):
        print(f"ERM data file not found: {file_path}. Skipping...")

    file_path = os.path.join(data_folder, file_name_bound_regp.format(pstar, reg_param))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_erm = data[:, 0]
        bound_err_mean, bound_err_std = data[:, 15], data[:, 16]

        axs[1].errorbar(
            alphas_erm,
            bound_err_mean,
            yerr=bound_err_std,
            color="k",
            fmt=".",
            linestyle="",
        )

    except (FileNotFoundError, IOError):
        print(f"ERM data file not found: {file_path}. Skipping...")

    file_path = os.path.join(data_folder, file_name_adverr_regp.format(pstar, reg_param))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_erm = data[:, 0]
        adv_errors_mean, adv_errors_std = data[:, 9], data[:, 10]

        axs[0].errorbar(
            alphas_erm,
            adv_errors_mean,
            yerr=adv_errors_std,
            color="k",
            fmt=".",
            linestyle="",
        )

    except (FileNotFoundError, IOError):
        print(f"ERM data file not found: {file_path}. Skipping...")

    # ------------------------- optimal lambda and eps_t ------------------------- #


axs[2].set_xlabel(f"$\\alpha = n / d$")
axs[2].set_xlim([0.3, 2.5])
axs[0].set_ylabel(r"$E_{\mathrm{rob}}$")
axs[1].set_ylabel(r"$E_{\mathrm{bnd}}^{\mathrm{prop}}$")
axs[2].set_ylabel(r"$E_{\mathrm{rob}}^{\mathrm{prop}}$")
for ax in axs:
    ax.grid(which="both", alpha=0.7)
    ax.set_xscale("log")
axs[0].set_title(r"$\mathrm{Direct\ Space\ Model}\ \mathrm{optimal} \lambda$")

save_plot(fig, "direct_space_model_alpha_sweep_effect_adv_train", formats=["pdf", "png"])

plt.show()
