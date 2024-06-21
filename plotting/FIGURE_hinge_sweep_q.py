import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.signal import find_peaks

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

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8
N = 1000

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

multiplier = 1.0
second_multiplier = 0.7

fname_amp = (
    "./simulations/data/AMP_probit_Hinge_proj_q_sweep_delta_{:.3f}_d_2500_reps_10_alpha_2.000.npz"
)
fname_se = "./simulations/data/SE_probit_Hinge_proj_q_sweep_delta_{:.3f}_alpha_{:.3f}.npz"
deltas = [0.0, 0.01, 0.1, 1.0]
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
alpha = 2.0

fig, axs = plt.subplots(
    nrows=2,
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
    # load the AMP data
    data_amp = np.load(fname_amp.format(delta_noise))

    # plot the AMP data
    axs[0].errorbar(
        data_amp["qs_amp"],
        data_amp["gen_err_mean_amp"],
        yerr=data_amp["gen_err_std_amp"],
        # label=r"$\Delta = {:.2f}$".format(delta_noise),
        linestyle="",
        marker=".",
        color=c,
    )
    axs[1].errorbar(
        data_amp["qs_amp"],
        data_amp["train_err_mean_amp"],
        yerr=data_amp["train_err_std_amp"],
        linestyle="",
        marker=".",
        label=r"$\Delta = {:.2f}$".format(delta_noise),
        color=c,
    )
    # axs[1].errorbar(
    #     data_amp["qs_amp"],
    #     data_amp["iters_nb_mean_amp"],
    #     yerr=data_amp["iters_nb_std_amp"],
    #     color=c,
    #     label=r"$\Delta = {:.2f}$".format(delta_noise),
    #     marker=".",
    # )

    # load the SE data
    data_se = np.load(fname_se.format(delta_noise, alpha))

    # plot the SE data
    axs[0].plot(
        data_se["qs"],
        np.arccos(data_se["ms"] / np.sqrt(data_se["qs"])) / np.pi,
        linestyle="-",
        color=c,
    )
    axs[1].plot(
        data_se["qs"],
        data_se["training_error"],
        linestyle="--",
        color=c,
    )

# axs[0].set_ylim(0.0, 1.0)
axs[0].set_ylabel(r"$E_{\mathrm{estim}}$", labelpad=2.0)
axs[0].grid(which="both", axis="both", alpha=0.5)


axs[1].set_ylabel(r"$E_{\mathrm{train}}$")
# axs[1].set_ylim(0.0, 1000.0)
# axs[1].set_yscale("log")
axs[1].grid(which="both", axis="both", alpha=0.5)
axs[1].set_xscale("log")
axs[1].set_xlabel(r"$q$", labelpad=2.0)
axs[1].legend()

# Create legend element by element
legend_elements = []
for delta_noise, c in zip(deltas, color_list):
    legend_elements.append(
        mpl.lines.Line2D(
            [0],
            [0],
            linestyle="",
            marker="o",
            color=c,
            label=r"$\Delta = {:.2f}$".format(delta_noise),
        )
    )

# Add legend to the plot
axs[1].legend(handles=legend_elements, labelspacing=0.25)

if save:
    save_plot(
        fig,
        "Hinge_different_delta_separate",
    )

plt.show()
