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

dat = np.loadtxt("./reviewer_data/review_delta_sweep_repetitions.csv", skiprows=1, delimiter=",")
alphas = dat[:, 1]
Eestim_mean = dat[:, 6]
Eestim_std = dat[:, 7]

# axs.errorbar(
#     alphas,
#     Eestim_mean,
#     yerr=Eestim_std,
#     marker=".",
#     # markersize=0.5,
#     # linewidth=0.5,
#     # linestyle="--",
#     label=r"Huber"
# )

dat2 = np.loadtxt(
    "./reviewer_data/review_delta_sweep_repetitions_short.csv", skiprows=1, delimiter=","
)
alphas2 = dat2[:, 1]
Eestim_mean2 = dat2[:, 6]
Eestim_std2 = dat2[:, 7]

# concatenate teh arrays
alphas3 = np.concatenate((alphas, alphas2))
Eestim_mean3 = np.concatenate((Eestim_mean, Eestim_mean2))
Eestim_std3 = np.concatenate((Eestim_std, Eestim_std2))

# axs.errorbar(
#     alphas3,
#     Eestim_mean3,
#     yerr=Eestim_std3,
#     marker=".",
#     # markersize=0.5,
#     # linewidth=0.5,
#     # linestyle="--",
#     label=r"Huber"
# )

dat3 = np.loadtxt(
    "./reviewer_data/review_delta_sweep_repetitions_short_3.csv", skiprows=1, delimiter=","
)
alphas3 = dat3[:, 1]
Eestim_mean3 = dat3[:, 6]
Eestim_std3 = dat3[:, 7]

axs.errorbar(
    alphas3,
    Eestim_mean3,
    yerr=Eestim_std3,
    marker=".",
    # markersize=0.5,
    # linewidth=0.5,
    # linestyle="--",
    label=r"Huber",
)

axs.set_xscale("log")
axs.set_ylabel(r"$a_{\mathrm{opt}}$", labelpad=2.0)
axs.set_xlabel(r"$\Delta_{\mathrm{OUT}}$", labelpad=2.0)
axs.grid(which="both", axis="both", alpha=0.5)
# axs.legend()

if save:
    save_plot(
        fig,
        "appendix_real_a_delta_sweep",
    )

plt.show()
