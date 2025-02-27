import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.signal import find_peaks
import datetime

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
width = 1.0 * 458.63788

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8
N = 1000

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

fname_se = "./simulations/data/q_sweep_exponential_loss_probit_delta_{:.4f}_alpha_{:.2f}.npz"
deltas = [0.0, 0.01, 0.1, 1.0]  # , 0.01, 0.1, 1.0
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
alpha = 2.0

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

for delta_noise, c in zip(deltas, color_list):
    # load the SE data
    data_se = np.load(fname_se.format(delta_noise, alpha))

    # plt.plot(
    #     data_se["qs"],
    #     data_se["stab_values"],
    #     color=c,
    #     linestyle="--",
    # )

    axs.plot(
        data_se["qs"],
        data_se["training_error"],
        color=c,
        label=r"$\Delta = {:.2f}$".format(delta_noise),
    )

    # first_neg_idx = 0
    # for i in range(len(data_se["stab_values"])):
    #     if data_se["stab_values"][i] < 0.0:
    #         first_neg_idx = i
    #         break

    # second_neg_idx = len(data_se["stab_values"]) - 1
    # for i in range(len(data_se["stab_values"]) - 1, 0, -1):
    #     if data_se["stab_values"][i] < 0.0:
    #         second_neg_idx = i
    #         break

    # stab_idx_1 = np.arange(len(data_se["stab_values"])) < first_neg_idx
    # stab_idx_2 = np.arange(len(data_se["stab_values"])) > second_neg_idx
    # not_stab_idx = (np.arange(len(data_se["stab_values"])) >= first_neg_idx - 1) * (np.arange(len(data_se["stab_values"])) <= second_neg_idx + 1)

    # axs.plot(
    #     data_se["qs"][stab_idx_1],
    #     data_se["training_error"][stab_idx_1],
    #     # linestyle="--",
    #     color=c,
    #     label=r"$\Delta = {:.2f}$".format(delta_noise),
    # )
    # axs.plot(
    #     data_se["qs"][stab_idx_2],
    #     data_se["training_error"][stab_idx_2],
    #     # linestyle="--",
    #     color=c,
    # )
    # axs.plot(
    #     data_se["qs"][not_stab_idx],
    #     data_se["training_error"][not_stab_idx],
    #     linestyle="--",
    #     color=c,
    # )

# axs.set_ylim(0.0, 1.0)
# axs.set_yscale("log")
axs.set_xscale("log")
axs.set_ylabel(r"Training Error", labelpad=1.0)
axs.set_xlabel(r"$q_{\mathrm{fix}}$", labelpad=0.0)
axs.grid(which="both", axis="both", alpha=0.5)
axs.legend()

# axs[1].set_ylabel("Iters. AMP")
# # axs[1].set_ylim(0.0, 1000.0)
# axs[1].set_yscale("log")
# axs[1].grid(which="both", axis="both", alpha=0.5)
# axs[1].set_xscale("log")
# axs[1].set_xlabel(r"$q$")

if save:
    save_plot(
        fig,
        "Exponential_different_delta",
    )

plt.show()
