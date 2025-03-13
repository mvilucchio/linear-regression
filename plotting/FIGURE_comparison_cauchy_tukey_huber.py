import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle


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


save = True
width = 433.62

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)


fig, axs = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(tuple_size[0], 0.75 * tuple_size[0]),
    gridspec_kw={"hspace": 0, "wspace": 0},
)
fig.subplots_adjust(left=0.20)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.98)
fig.subplots_adjust(right=0.96)

alpha_min, alpha_max = 2.8, 1_000
n_alpha_pts = 60
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.0
c = 0.01
d = 300

alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
mask = alphas < 100
alphas = alphas[mask]
len_alphas = len(alphas)

data_folder = "./data/mod_Tukey_decorrelated_noise"
tukey_se = f"optimal_se_tukey_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}_{c}.pkl"
huber_se = f"optimal_se_huber_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}.pkl"

tukey_erm = f"optimal_erm_tukey_{alpha_min}_{100}_{len_alphas}_{d}_{delta_in}_{delta_out}_{percentage}_{beta}_{c}.pkl"
huber_erm = f"optimal_erm_huber_{alpha_min}_{100}_{len_alphas}_{1000}_{delta_in}_{delta_out}_{percentage}_{beta}.pkl"

data_folder_cauchy = "./data/Cauchy_decorrelated_noise"
cauchy_se = f"optimal_se_cauchy_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}_{c}.pkl"
cauchy_erm = f"optimal_erm_cauchy_{alpha_min}_{100}_{len_alphas}_{d}_{delta_in}_{delta_out}_{percentage}_{beta}.pkl"


L2_se = f"optimal_se_L2_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}_{c}.pkl"
L2_erm = f"optimal_erm_L2_{alpha_min}_{100}_{len_alphas}_{d}_{delta_in}_{delta_out}_{percentage}_{beta}.pkl"

with open(os.path.join(data_folder, tukey_se), "rb") as f:
    data_tukey = pickle.load(f)

plt.plot(
    data_tukey["alphas"], 1 - 2 * data_tukey["ms"] + data_tukey["qs"], label="Tukey", color="C0"
)

with open(os.path.join(data_folder, huber_se), "rb") as f:
    data_huber = pickle.load(f)

plt.plot(data_huber["alphas"], data_huber["f_min_vals"], label="Huber", color="C1")

with open(os.path.join(data_folder, tukey_erm), "rb") as f:
    data_tukey_erm = pickle.load(f)

plt.errorbar(
    data_tukey_erm["alphas"],
    data_tukey_erm["estim_error"][:, 0],
    yerr=data_tukey_erm["estim_error"][:, 1],
    ls="None",
    color="C0",
)

with open(os.path.join(data_folder, huber_erm), "rb") as f:
    data_huber_erm = pickle.load(f)

plt.errorbar(
    data_huber_erm["alphas"],
    data_huber_erm["estim_error"][:, 0],
    yerr=data_huber_erm["estim_error"][:, 1],
    ls="None",
    color="C1",
)

with open(os.path.join(data_folder_cauchy, cauchy_se), "rb") as f:
    data_cauchy = pickle.load(f)

plt.plot(
    data_cauchy["alphas"], 1 - 2 * data_cauchy["ms"] + data_cauchy["qs"], label="Cauchy", color="C2"
)

with open(os.path.join(data_folder_cauchy, cauchy_erm), "rb") as f:
    data_cauchy_erm = pickle.load(f)

plt.errorbar(
    data_cauchy_erm["alphas"],
    data_cauchy_erm["estim_error"][:, 0],
    yerr=data_cauchy_erm["estim_error"][:, 1],
    ls="None",
    color="C2",
)

with open(os.path.join(data_folder_cauchy, L2_se), "rb") as f:
    data_L2 = pickle.load(f)

plt.plot(data_L2["alphas"], 1 - 2 * data_L2["ms"] + data_L2["qs"], label="Ridge", color="C3")

with open(os.path.join(data_folder_cauchy, L2_erm), "rb") as f:
    data_L2_erm = pickle.load(f)

plt.errorbar(
    data_L2_erm["alphas"],
    data_L2_erm["estim_error"][:, 0],
    yerr=data_L2_erm["estim_error"][:, 1],
    ls="None",
    color="C3",
)

# axs.set_title(r"$\alpha = {:.0f}$ $a = {:.0f}$ $\Delta_{{\mathrm{{IN}}}} = {:.1f}$ $\Delta_{{\mathrm{{OUT}}}} = {:.1f}$ $\beta = {:.1f}$ $\epsilon = {:.1f}$".format(alpha, a, delta_in, delta_out, beta, percentage), fontsize=9)
axs.set_xscale("log")
axs.set_yscale("log")
# axs.set_ylabel(r"$E_{\mathrm{train}}$", labelpad=2.0)
axs.set_ylabel(r"$\frac{1}{d} \norm{\hat{\mathbf{w}} - \mathbf{w}_\star }_2^2$", labelpad=2.0)
axs.set_xlabel(r"$\alpha$", labelpad=0.0)
axs.grid(which="both", axis="both", alpha=0.5)
# axs.set_xlim(1e-1, 1e2)
# axs.set_ylim(0.5, 2.5)
axs.legend()

# create a legend with the specific order
# handles, labels = axs.get_legend_handles_labels()
# order = [3, 0, 2, 1]
# axs.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

if save:
    save_plot(
        fig,
        "comparison_Tukey_Huber",
    )

plt.show()
