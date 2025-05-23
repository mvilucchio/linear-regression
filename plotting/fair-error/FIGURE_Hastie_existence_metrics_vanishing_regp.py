import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
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
from linear_regression.aux_functions.misc import classification_adversarial_error_latent
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
data_folder = "./data/hastie_model_training"
reps = 10
reg_param = 1e-3
eps_min, eps_max, n_epss = 0.1, 10, 15

different_alphas = [0.5, 1.0, 1.5]
different_gammas = [0.1, 0.5, 1.0, 1.5, 2.0]

dim = 512

pstar = 1.0
reg = 2.0

# File name templates
file_name_erm = f"ERM_sweep_eps_Hastie_Linf_d_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_epss_{{:.1f}}_{{:.1f}}_{{:d}}_pstar_{pstar:.1f}_reg_{reg:.1f}_regparam_{reg_param:.1e}.csv"

fig, axs = plt.subplots(
    3,
    len(different_alphas),
    sharex=True,
    sharey=False,  # Changed from True to False
    figsize=(3 * tuple_size[0], 3 * tuple_size[1]),
    gridspec_kw={"hspace": 0, "wspace": 0.0},
)

# Manually share y-axis within each row
for row in range(3):
    for col in range(1, len(different_alphas)):
        axs[row, col].sharey(axs[row, 0])

eps_dense = np.logspace(-1.2, 1.2, 50)
out = np.empty_like(eps_dense)

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

# the rows are for the different metrics in order : adv err, flipped, misclassified
# the columns are for different alphas
for i, alpha in enumerate(different_alphas):
    axs[0, i].set_title(f"$\\alpha$={alpha}", fontsize=12)

    for j, gamma in enumerate(different_gammas):

        file_path = os.path.join(
            data_folder, file_name_erm.format(dim, alpha, gamma, eps_min, eps_max, n_epss)
        )

        try:
            # Load CSV file instead of pickle
            with open(file_path, "r") as f:
                header = f.readline().strip().split(",")

            # Load data with column names from header
            data_array = np.loadtxt(file_path, delimiter=",", skiprows=1)

            # Create dictionary mapping column names to data
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
            mean_q_latent = data["mean_q_latent"][0]
            mean_q_features = data["mean_q_features"][0]

            # Fix axis indexing to use [row, column] format
            axs[0, i].errorbar(epss_g, mean_adverr, yerr=std_adverr, linestyle="", color=colors[j])
            axs[1, i].errorbar(epss_g, mean_bound, yerr=std_bound, linestyle="", color=colors[j])
            axs[2, i].errorbar(
                epss_g, mean_misclassified, yerr=std_misclassified, linestyle="", color=colors[j]
            )

            for k, eps_i in enumerate(eps_dense):
                out[k] = classification_adversarial_error_latent(
                    mean_m, mean_q, mean_q_features, mean_q_latent, 1.0, mean_P, eps_i, gamma, pstar
                )
            axs[0, i].plot(eps_dense, out, color=colors[j], linestyle="--")

            for k, eps_i in enumerate(eps_dense):
                out[k] = percentage_flipped_hastie_model(
                    mean_m, mean_q, mean_q_latent, mean_q_features, 1.0, eps_i, gamma, "inf"
                )
            axs[1, i].plot(eps_dense, out, color=colors[j], linestyle="--")

            for k, eps_i in enumerate(eps_dense):
                out[k] = percentage_misclassified_hastie_model(
                    mean_m, mean_q, mean_q_latent, mean_q_features, 1.0, eps_i, gamma, "inf"
                )
            axs[2, i].plot(eps_dense, out, color=colors[j], linestyle="--")

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue


for i in range(3):
    axs[i, 0].set_ylabel(
        [r"$E_{\mathrm{adv}}$", r"$E_{\mathrm{flip}}$", r"$E_{\mathrm{flip}}^{\mathrm{true}}$"][i]
    )

    for j in range(len(different_alphas)):
        axs[i, j].grid(True, which="both", linestyle="--", linewidth=0.5)

# Remove yticks and labels for second and third columns
for i in range(3):
    for j in range(1, len(different_alphas)):
        axs[i, j].tick_params(axis="y", which="both", left=False, labelleft=False)

for j in range(len(different_alphas)):
    axs[2, j].set_xlabel(r"$\varepsilon$")
    axs[2, j].set_xscale("log")
    axs[2, j].set_xlim(eps_min, eps_max)


plt.tight_layout()

legend_handles = []
for j, gamma in enumerate(different_gammas):
    legend_handles.append(
        plt.Line2D([0], [0], color=colors[j], lw=2, label=f"$\\gamma={gamma:.1f}$")
    )

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=len(different_gammas),
    frameon=False,
)

plt.subplots_adjust(top=0.88)

save_plot(fig, "hastie_model_eps_sweep_optimal_regp", formats=["pdf", "png"])

plt.show()
