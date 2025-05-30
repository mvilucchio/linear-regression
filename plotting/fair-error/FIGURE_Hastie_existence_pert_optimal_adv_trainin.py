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

# Define parameters
data_folder = "./data/hastie_model_training_optimal_cluster"
reps = 10
reg_param = 1e-2
eps_min, eps_max, n_epss = 0.1, 10, 15

param_pairs = [(1.0, 0.5, 2**10)]  # (alpha, gamma, d)
# param_pairs = []  # (alpha, gamma, d)
# param_pairs = [(3.0, 0.5, 2**10)]  # (alpha, gamma, d)

# File name templates
file_name_flipped_no_adv = f"ERM_flipped_optimal_noadvtrain_Hastie_Linf_d_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_epss_{{:.1f}}_{{:.1f}}_{{:d}}.csv"
file_name_flipped_adv = f"ERM_flipped_optimal_advtrain_Hastie_Linf_d_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_epss_{{:.1f}}_{{:.1f}}_{{:d}}.csv"
file_name_misclass_no_adv = file_name_flipped_no_adv.replace("flipped", "misclass")
file_name_misclass_adv = file_name_flipped_adv.replace("flipped", "misclass")

# dimension = 1024

marker_size = 3
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=(tuple_size[0], tuple_size[1]), gridspec_kw={"hspace": 0}
)
# fig, axs = plt.subplots(
#     2, 1, sharex=True, figsize=(tuple_size[0], tuple_size[1]), gridspec_kw={"hspace": 0}
# )

eps_dense = np.logspace(-1.2, 1.2, 50)
out = np.empty_like(eps_dense)

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

for i, (alpha, gamma, dimension) in enumerate(param_pairs):
    color = colors[i % len(colors)]
    marker = ["o", "s", "^", "D", "v", ">", "<", "p"][i % 8]

    file_path_flipped_no_adv = os.path.join(
        data_folder,
        file_name_flipped_no_adv.format(dimension, alpha, gamma, eps_min, eps_max, n_epss),
    )

    try:
        with open(file_path_flipped_no_adv, "rb") as f:
            data = pickle.load(f)

            epss_g = data["eps"]
            mean_flipped = data["mean_flipped"]
            std_flipped = data["std_flipped"]

            opt_reg_param = data["optimal_reg_param"]

            if "mean_m" in data and "mean_q" in data and "mean_P" in data:
                mean_m = data["mean_m"]
                mean_q = data["mean_q"]
                mean_P = data["mean_P"]

            axs[0].errorbar(
                epss_g,
                mean_flipped,
                yerr=std_flipped,
                linestyle="",
                color=color,
                marker=marker,
                markersize=marker_size,
                label=f"ERM $\\alpha$={alpha}, $\\gamma$={gamma}",
            )

        m_se, q_se, q_latent_se, q_features_se, V_se, P_se = compute_theory_overlaps(
            opt_reg_param, 0.0, alpha, gamma, (mean_m, mean_q, 1.0, mean_P)
        )

        for j, eps_i in enumerate(eps_dense):
            out[j] = percentage_flipped_hastie_model(
                m_se, q_se, q_latent_se, q_features_se, 1, eps_i, gamma, "inf"
            )

        axs[0].plot(
            eps_dense,
            out,
            color=color,
            linestyle="--",
            label=f"Theory $\\alpha$={alpha}, $\\gamma$={gamma}",
        )
    except FileNotFoundError:
        print(f"Error: File not found - {file_path_flipped_no_adv}")

    file_path_flipped_adv = os.path.join(
        data_folder, file_name_flipped_adv.format(dimension, alpha, gamma, eps_min, eps_max, n_epss)
    )

    try:
        with open(file_path_flipped_adv, "rb") as f:
            data = pickle.load(f)

            epss_g = data["eps"]
            mean_flipped = data["mean_flipped"]
            std_flipped = data["std_flipped"]

            opt_reg_param = data["optimal_reg_param"]
            opt_eps_training = data["optimal_eps_training"]

            if "mean_m" in data and "mean_q" in data and "mean_P" in data:
                mean_m = data["mean_m"]
                mean_q = data["mean_q"]
                mean_P = data["mean_P"]

            axs[0].errorbar(
                epss_g,
                mean_flipped,
                yerr=std_flipped,
                linestyle="",
                color=color,
                marker=marker,
                markersize=marker_size,
                label=f"ERM $\\alpha$={alpha}, $\\gamma$={gamma}",
            )

        m_se, q_se, q_latent_se, q_features_se, V_se, P_se = compute_theory_overlaps(
            opt_reg_param, opt_eps_training, alpha, gamma, (mean_m, mean_q, 1.0, mean_P)
        )

        for j, eps_i in enumerate(eps_dense):
            out[j] = percentage_flipped_hastie_model(
                m_se, q_se, q_latent_se, q_features_se, 1, eps_i, gamma, "inf"
            )

        axs[0].plot(
            eps_dense,
            out,
            color=color,
            linestyle="-",
        )
    except FileNotFoundError:
        print(f"Error: File not found - {file_path_flipped_adv}")

    file_path_misclass_no_adv = os.path.join(
        data_folder,
        file_name_misclass_no_adv.format(dimension, alpha, gamma, eps_min, eps_max, n_epss),
    )

    try:
        print(f"Loading {file_path_misclass_no_adv}")
        with open(file_path_misclass_no_adv, "rb") as f:
            data = pickle.load(f)

            epss_g = data["eps"]
            mean_misclass = data["mean_misclass"]
            std_misclass = data["std_misclass"]

            opt_reg_param = data["optimal_reg_param"]

            if "mean_m" in data and "mean_q" in data and "mean_P" in data:
                mean_m = data["mean_m"]
                mean_q = data["mean_q"]
                mean_P = data["mean_P"]

            axs.errorbar(
                epss_g,
                mean_misclass,
                yerr=std_misclass,
                linestyle="",
                color=color,
                marker=marker,
                markersize=marker_size,
                label=f"ERM $\\alpha$={alpha}, $\\gamma$={gamma}",
            )

        m_se, q_se, q_latent_se, q_features_se, V_se, P_se = compute_theory_overlaps(
            opt_reg_param, 0.0, alpha, gamma, (mean_m, mean_q, 1.0, mean_P)
        )

        for j, eps_i in enumerate(eps_dense):
            out[j] = percentage_misclassified_hastie_model(
                m_se, q_se, q_latent_se, q_features_se, 1, eps_i, gamma, "inf"
            )

        axs.plot(
            eps_dense,
            out,
            color=color,
            linestyle="--",
        )
    except FileNotFoundError:
        print(f"Error: File not found - {file_path_misclass_no_adv}")

    file_path_misclass_adv = os.path.join(
        data_folder,
        file_name_misclass_adv.format(dimension, alpha, gamma, eps_min, eps_max, n_epss),
    )

    try:
        print(f"Loading {file_path_misclass_adv}")
        with open(file_path_misclass_adv, "rb") as f:
            data = pickle.load(f)

            epss_g = data["eps"]
            mean_misclass = data["mean_misclass"]
            std_misclass = data["std_misclass"]

            opt_reg_param = data["optimal_reg_param"]
            opt_eps_training = data["optimal_eps_training"]

            if "mean_m" in data and "mean_q" in data and "mean_P" in data:
                mean_m = data["mean_m"]
                mean_q = data["mean_q"]
                mean_P = data["mean_P"]

            axs.errorbar(
                epss_g,
                mean_misclass,
                yerr=std_misclass,
                linestyle="",
                color=color,
                marker=marker,
                markersize=marker_size,
                label=f"ERM $\\alpha$={alpha}, $\\gamma$={gamma}",
            )
        m_se, q_se, q_latent_se, q_features_se, V_se, P_se = compute_theory_overlaps(
            opt_reg_param, opt_eps_training, alpha, gamma, (mean_m, mean_q, 1.0, mean_P)
        )

        for j, eps_i in enumerate(eps_dense):
            out[j] = percentage_misclassified_hastie_model(
                m_se, q_se, q_latent_se, q_features_se, 1, eps_i, gamma, "inf"
            )

        axs.plot(
            eps_dense,
            out,
            color=color,
            linestyle="-",
        )
    except FileNotFoundError:
        print(f"Error: File not found - {file_path_misclass_adv}")

# Set subplot properties
for i, ax in enumerate([axs]):
    ax.set_xscale("log")
    if i == 0:
        ax.set_ylim(0, 1)
    ax.set_xlim(0.1, 10)
    ax.grid(which="both", alpha=0.3)

    # Set y-labels
    if i == 1:
        ax.set_ylabel(r"$E_{\mathrm{flip}}$")
    else:
        ax.set_ylabel(r"$E_{\mathrm{flip}}^{\mathrm{true}}$")

axs.set_xlabel(r"$\varepsilon_g$")

plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.show()
