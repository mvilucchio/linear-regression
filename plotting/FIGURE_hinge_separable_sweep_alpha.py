import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.signal import find_peaks
from linear_regression.aux_functions.training_errors import training_error_Hinge_loss_no_noise

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
width = 1.5 * 458.63788

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8
N = 1000

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

fname = "./simulations/data/{}_classification_fixed_lambda_{:.2e}_delta_{:.2e}.npz"
reg_prams = [0.005, 0.05, 0.5, 5.0]
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

for reg_param in reg_prams:
    # load the data
    data = np.load(fname.format("Hinge", reg_param, 0.0))

    axs[0].plot(
        data["alphas"],
        data["angle_ts"],
        "-",
        label=r"$\lambda = {:.3f}$".format(reg_param),
    )

    axs[1].plot(
        data["alphas"],
        np.array(
            [
                training_error_Hinge_loss_no_noise(m, q, sigma)
                for m, q, sigma in zip(data["ms"], data["qs"], data["sigmas"])
            ]
        ),
        "-",
    )

# load the BO data
fname = "./simulations/data/BO_classification_fixed_lambda_0.00e+00_delta_0.00e+00.npz"
data = np.load(fname)
axs[0].plot(
    data["alphas"],
    data["angle_ts"],
    "--",
    color="k",
)

axs[0].set_ylabel(r"$E_{\mathrm{gen}}$")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].legend(loc="best")
axs[0].grid(which="both")

axs[1].set_ylabel(r"$E_{\mathrm{train}}$")
axs[1].set_xscale("log")
axs[1].set_xlabel(r"$\alpha$")
axs[1].grid(which="both")

if save:
    save_plot(
        fig,
        "Hinge_different_reg_param_alpha_sweep",
    )

plt.show()
