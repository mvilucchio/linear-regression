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
width = 458.63788

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

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

x = np.arange(30)
y = np.exp(-0.1 * x) + 0.1

axs.plot(x, y, ".-")
axs.axhline(0.1, color="black", linestyle="--", label=r"$\mathcal{A}^\star$")

# axs.set_yscale("log")
axs.set_ylabel(r"AMP Energy", labelpad=2.0)
axs.set_xlabel(r"Iterations", labelpad=2.0)
axs.grid(
    which="both",
    axis="both",
    alpha=0.5,
)
axs.legend()
# axs.set_yticks([])
# axs.set_yticklabels([])

if save:
    save_plot(
        fig,
        "convergence_cartoon",
    )

plt.show()
