import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
from linear_regression.aux_functions.loss_functions import (
    mod_tukey_loss_cubic,
    huber_loss,
    l2_loss,
    cauchy_loss,
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


save = True
width = 433.62

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{mathrsfs}"
            r"\let\oldmathscr\mathscr"  # Save the original mathscr
            r"\renewcommand{\mathscr}[1]{\oldmathscr{#1}}"  # Define a local version
        ),
    }
)

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

zz = np.linspace(-5, 5, 1000)

τ = 1.5

# Huber
plt.plot(zz, huber_loss(0.0, zz, τ), label=r"Huber", color="C1")

# Tukey
plt.plot(zz, mod_tukey_loss_cubic(0.0, zz, τ, 0.01), label=r"Tukey", color="C0")

# Cauchy
plt.plot(zz, cauchy_loss(0.0, zz, τ), label=r"Cauchy", color="C2")

# L2
plt.plot(zz, l2_loss(0.0, zz), label=r"L2", color="C3")

# axs.set_ylabel(r"$E_{\mathrm{train}}$", labelpad=2.0)
axs.set_ylabel(r"$\mathscr{L}(z)$", labelpad=2.0)
axs.set_xlabel(r"$z$", labelpad=0.0)
axs.grid(which="both", axis="both", alpha=0.5)
# axs.set_xlim(0.0, 10)
axs.set_ylim(0.0, 5)
axs.legend()

# create a legend with the specific order
# handles, labels = axs.get_legend_handles_labels()
# order = [3, 0, 2, 1]
# axs.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

if save:
    save_plot(
        fig,
        "loss_presentaiton_Tukey_Huber",
    )

plt.show()
