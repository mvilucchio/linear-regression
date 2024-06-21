import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


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
width = 1.25 * 458.63788

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

multiplier = 0.8
second_multiplier = 0.6

fig, axs = plt.subplots(
    nrows=1,
    ncols=1,
    sharex=True,
    figsize=(multiplier * tuple_size[0], 0.75 * multiplier * tuple_size[0]),
    gridspec_kw={"hspace": 0},
)
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)

# data loading
dat = np.loadtxt(
    "./reviewer_data/review_alpha_sweep_repetitions_a_100.0.csv", skiprows=1, delimiter=","
)
alphas = dat[:, 1]
Eestim_mean = dat[:, 2]
Eestim_std = dat[:, 3]

axs.errorbar(
    alphas,
    Eestim_mean,
    yerr=Eestim_std,
    marker=".",
    # markersize=0.5,
    # linewidth=0.5,
    # linestyle="--",
    label=r"$\ell_2$",
)


dat = np.loadtxt(
    "./reviewer_data/review_alpha_sweep_repetitions_a_0.001.csv", skiprows=1, delimiter=","
)
alphas = dat[:, 1]
Eestim_mean = dat[:, 2]
Eestim_std = dat[:, 3]

axs.errorbar(
    alphas,
    Eestim_mean,
    yerr=Eestim_std,
    marker=".",
    # markersize=0.5,
    # linewidth=0.5,
    # linestyle="--",
    label=r"$\ell_1$",
)

dat = np.loadtxt(
    "./reviewer_data/review_alpha_sweep_repetitions_a_1.0.csv", skiprows=1, delimiter=","
)
alphas = dat[:, 1]
Eestim_mean = dat[:, 2]
Eestim_std = dat[:, 3]

axs.errorbar(
    alphas,
    Eestim_mean,
    yerr=Eestim_std,
    marker=".",
    # markersize=0.5,
    # linewidth=0.5,
    # linestyle="--",
    label=r"Huber",
)


axs.set_xscale("log")
axs.set_ylabel(r"$E_{\mathrm{estim}}$", labelpad=2.0)
axs.set_xlabel(r"$\alpha$", labelpad=2.0)
axs.grid(which="both", axis="both", alpha=0.5)
axs.legend()

if save:
    save_plot(
        fig,
        "appendix_real_estimation_error_alpha_sweep",
    )

plt.show()
