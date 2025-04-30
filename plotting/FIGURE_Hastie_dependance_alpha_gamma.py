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
alpha_min, alpha_max, n_alphas_se, gammas = 0.1, 5.0, 50, [0.1, 1.0, 2.0]
gamma_min, gamma_max, n_gammas_se, alphas = 0.1, 3.0, 50, [0.1, 1.0, 2.0]

eps_t = 0.1
reg_param = 1e-2

# File name templates
pstar = 1.0
data_folder = "./data/hastie_model_training"
file_name_sweep_alpha = f"SE_training_gamma_{{gamma:.2f}}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_se:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"
file_name_sweep_gamma = f"SE_training_alpha_{{alpha:.2f}}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas_se:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"

# Define parameters ERM
# dimension = 1024
# param_pairs = [(2.0, 0.5, 2**10)]  # (alpha, gamma, d)
# param_pairs = [(0.01, 3.0, 2**10)]  # (alpha, gamma, d)
d = 500
reps = 10
alpha_min_erm, alpha_max_erm, n_alphas_erm = 0.5, 3.0, 10
gamma_min_erm, gamma_max_erm, n_gammas_erm = 0.5, 3.0, 10
delta = 0.0
file_name_sweep_gamma_erm = f"ERM_training_alpha_{{alpha:.2f}}_gammas_{gamma_min_erm:.1f}_{gamma_max_erm:.1f}_{n_gammas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"
file_name_sweep_alpha_erm = f"ERM_training_gamma_{{gamma:.2f}}_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"


marker_size = 3

# first plot sweep in alpha
# the axes above is the one for the flipped one
# the one below is the one for the misclass one
fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=(tuple_size[0], tuple_size[0]), gridspec_kw={"hspace": 0}
)
fig.subplots_adjust(left=0.20)
fig.subplots_adjust(bottom=0.12)
fig.subplots_adjust(top=0.92)
fig.subplots_adjust(right=0.96)

for i, gamma in enumerate(gammas):

    file_path = os.path.join(data_folder, file_name_sweep_alpha.format(gamma=gamma))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_se = data[:, 0]
        m = data[:, 1]
        q = data[:, 2]
        q_latent = data[:, 3]
        q_features = data[:, 4]
        V = data[:, 5]
        P = data[:, 6]
        gen_err = data[:, 9]
        adv_err = data[:, 8]
        flipped_fair = data[:, 10]
        misclas_fair = data[:, 11]

        axs[0].plot(alphas_se, adv_err, color=f"C{i}", label=f"$\\gamma = $ {gamma:.1f}")
        axs[1].plot(alphas_se, flipped_fair, color=f"C{i}")
        axs[2].plot(alphas_se, misclas_fair, color=f"C{i}")

    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    file_path = os.path.join(data_folder, file_name_sweep_alpha_erm.format(gamma=gamma))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        gammas = data[:, 0]
        m_mean, m_std = data[:, 1], data[:, 2]
        q_mean, q_std = data[:, 3], data[:, 4]
        q_latent_mean, q_latent_std = data[:, 5], data[:, 6]
        q_feature_mean, q_feature_std = data[:, 7], data[:, 8]
        P_mean, P_std = data[:, 9], data[:, 10]
        gen_errors_mean, gen_errors_std = data[:, 11], data[:, 12]
        adv_errors_mean, adv_errors_std = data[:, 13], data[:, 14]
        flipped_fairs_mean, flipped_fairs_std = data[:, 15], data[:, 16]
        misclas_fairs_mean, misclas_fairs_std = data[:, 17], data[:, 18]

        axs[0].errorbar(
            gammas,
            adv_errors_mean,
            yerr=adv_errors_std,
            color=f"C{i}",
            fmt=".",
        )

        axs[1].errorbar(
            gammas,
            flipped_fairs_mean,
            yerr=flipped_fairs_std,
            color=f"C{i}",
            fmt=".",
        )

        axs[2].errorbar(
            gammas,
            misclas_fairs_mean,
            yerr=misclas_fairs_std,
            color=f"C{i}",
            fmt=".",
        )

    except (FileNotFoundError, IOError):
        print(f"File {file_path} does not exist. Skipping...")


axs[2].set_xlabel(f"$\\alpha = n / d$")
axs[2].set_xlim([0.3, 2.5])
axs[0].set_ylabel(r"$E_{\mathrm{adv}}$")
axs[1].set_ylabel(r"$E_{\mathrm{flip}}$")
axs[2].set_ylabel(r"$E_{\mathrm{flip}}^{\mathrm{true}}$")

for ax in axs:
    ax.grid(which="both", alpha=0.7)

axs[0].legend(bbox_to_anchor=(-0.15, 1.0), loc="lower left", ncols=3, labelspacing=0.3)

save_plot(fig, "hastie_model_alpha_sweep", formats=["pdf"])

plt.show()

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=(tuple_size[0], tuple_size[0]), gridspec_kw={"hspace": 0}
)
fig.subplots_adjust(left=0.20)
fig.subplots_adjust(bottom=0.12)
fig.subplots_adjust(top=0.92)
fig.subplots_adjust(right=0.96)

for i, alpha in enumerate(alphas):

    file_path = os.path.join(data_folder, file_name_sweep_gamma.format(alpha=alpha))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_se = data[:, 0]
        m = data[:, 1]
        q = data[:, 2]
        q_latent = data[:, 3]
        q_features = data[:, 4]
        V = data[:, 5]
        P = data[:, 6]
        gen_err = data[:, 9]
        adv_err = data[:, 8]
        flipped_fair = data[:, 10]
        misclas_fair = data[:, 11]

        axs[0].plot(alphas_se, adv_err, color=f"C{i}", label=f"$\\alpha = $ {alpha:.1f}")
        axs[1].plot(alphas_se, flipped_fair, color=f"C{i}")
        axs[2].plot(alphas_se, misclas_fair, color=f"C{i}")

    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}. Skipping...")

    file_path = os.path.join(data_folder, file_name_sweep_gamma_erm.format(alpha=alpha))

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        gammas = data[:, 0]
        m_mean, m_std = data[:, 1], data[:, 2]
        q_mean, q_std = data[:, 3], data[:, 4]
        q_latent_mean, q_latent_std = data[:, 5], data[:, 6]
        q_feature_mean, q_feature_std = data[:, 7], data[:, 8]
        P_mean, P_std = data[:, 9], data[:, 10]
        gen_errors_mean, gen_errors_std = data[:, 11], data[:, 12]
        adv_errors_mean, adv_errors_std = data[:, 13], data[:, 14]
        flipped_fairs_mean, flipped_fairs_std = data[:, 15], data[:, 16]
        misclas_fairs_mean, misclas_fairs_std = data[:, 17], data[:, 18]

        axs[0].errorbar(
            gammas,
            adv_errors_mean,
            yerr=adv_errors_std,
            color=f"C{i}",
            fmt=".",
        )

        axs[1].errorbar(
            gammas,
            flipped_fairs_mean,
            yerr=flipped_fairs_std,
            color=f"C{i}",
            fmt=".",
        )

        axs[2].errorbar(
            gammas,
            misclas_fairs_mean,
            yerr=misclas_fairs_std,
            color=f"C{i}",
            fmt=".",
        )

    except (FileNotFoundError, IOError):
        print(f"File {file_path} does not exist. Skipping...")

axs[2].set_xlabel(f"$\\gamma = d / p$")
# axs[2].set_xlim([gamma_min, gamma_max])
axs[2].set_xlim([0.4, gamma_max])
axs[0].set_ylabel(r"$E_{\mathrm{adv}}$")
axs[1].set_ylabel(r"$E_{\mathrm{flip}}$")
axs[2].set_ylabel(r"$E_{\mathrm{flip}}^{\mathrm{true}}$")

for ax in axs:
    ax.grid(which="both", alpha=0.7)

axs[0].legend(bbox_to_anchor=(-0.15, 1.0), ncols=3, loc="lower left", labelspacing=0.3)

# Save the second plot as PDF
save_plot(fig, "hastie_model_gamma_sweep", formats=["pdf"])

plt.show()
