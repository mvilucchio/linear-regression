import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.signal import find_peaks
from linear_regression.aux_functions.stability_functions import (
    stability_Logistic_no_noise_classif,
    stability_Logistic_probit_classif,
)
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
width = 1.25 * 458.63788

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8
N = 1000

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

fname_se_logistic = "./simulations/data/margin_Logistic_alpha_{:.2f}_delta_{:.2f}.npy"
fname_se_hinge = "./simulations/data/margin_Hinge_alpha_{:.2f}_delta_{:.2f}.npy"
fname_se_exponential = "./simulations/data/margin_Exponential_alpha_{:.2f}_delta_{:.2f}.npy"

deltas = [0.0, 0.01, 0.1, 1.0]
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
    print("---- ", delta_noise)

    data_se_logistic = np.load(fname_se_logistic.format(alpha, delta_noise))
    data_se_hinge = np.load(fname_se_hinge.format(alpha, delta_noise))
    data_se_exponential = np.load(fname_se_exponential.format(alpha, delta_noise))

    qs_log_exp = np.logspace(-1, np.log10(20), 50)
    qs_hinge = np.logspace(-1, np.log10(20), 200)
    interp_hinge = np.interp(qs_log_exp, qs_hinge, data_se_hinge)
    # interp_margin_log = np.interp(qs_hinge, qs_log_exp, data_se_logistic)
    # interp_margin_exp = np.interp(qs_hinge, qs_log_exp, data_se_exponential)

    axs.plot(
        qs_log_exp,
        data_se_exponential / interp_hinge,
        color=c,
        linestyle="solid",
        label=r"$\Delta = {:.2f}$".format(delta_noise),
    )

    axs.plot(qs_log_exp, data_se_logistic / interp_hinge, color=c, linestyle="dashed")

    # axs.plot(
    #     qs_log_exp,
    #     data_se_logistic,
    #     color=c,
    #     linestyle="dashed"
    # )

    # axs.plot(
    #     qs_log_exp,
    #     data_se_exponential,
    #     color=c,
    #     linestyle="dotted"
    # )

axs.set_xscale("log")
# axs.set_yscale("log")
axs.set_ylabel(r"Relative Average Margin", labelpad=1.0)
axs.set_xlabel(r"$q_{\mathrm{fix}}$", labelpad=0.0)
axs.grid(which="both", axis="both", alpha=0.5)
axs.legend()

if save:
    save_plot(
        fig,
        "Average_margin_comparison_division",
    )

plt.show()
