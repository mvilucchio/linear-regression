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
fig.subplots_adjust(left=0.17)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.98)
fig.subplots_adjust(right=0.96)


delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 1.0
alpha = 10.0

data_directory = "./data"
file_prefix = "aa_huber_1.000"
file_list = [
    file
    for file in os.listdir(data_directory)
    if file.startswith(file_prefix) and "_alpha_10.000_" in file
]
path_list = [os.path.join(data_directory, file) for file in file_list]

label_list = []
for file in file_list:
    label = file.split("reg_param_")[1].split("_")[0]
    label_list.append(float(label))

print(label_list)

data = []
for path in path_list:
    with open(path, "rb") as file:
        data.append(pickle.load(file))

colors = [f"C{i}" for i in range(len(data))]

for d, lb, c in zip(data, label_list, colors):
    # if lb == -0.5:
    #     continue
    axs.plot(d["qs"], d["training_errors"], label=f"$\\lambda = {lb:.1f}$", color=c)
    # axs.plot(d["q_true"], d["training_error_true"], "x", color=c)

# axs.set_title(r"$\alpha = {:.0f}$ $a = {:.0f}$ $\Delta_{{\mathrm{{IN}}}} = {:.1f}$ $\Delta_{{\mathrm{{OUT}}}} = {:.1f}$ $\beta = {:.1f}$ $\epsilon = {:.1f}$".format(alpha, a, delta_in, delta_out, beta, percentage), fontsize=9)
axs.set_xscale("log")
# axs.set_ylabel(r"$E_{\mathrm{train}}$", labelpad=2.0)
axs.set_ylabel(r"$\mathcal{A}^{\star}(q)$", labelpad=3.0)
axs.set_xlabel(r"$q$", labelpad=0.0)
axs.grid(which="both", axis="both", alpha=0.5)
axs.set_xlim(1e-1, 1e2)
axs.set_ylim(0.4, 2.5)
# axs.legend()

# create a legend with the specific order
handles, labels = axs.get_legend_handles_labels()
order = [1, 4, 0, 3, 2]
axs.legend(
    [handles[idx] for idx in order],
    [
        "$\\lambda > 0$",
        "$\\lambda = 0$",
        "$\\lambda \\in [\\lambda_c, 0]$",
        "$\\lambda = \\lambda_c$",
        "$\\lambda < \\lambda_c$",
    ],
)

if save:
    save_plot(
        fig,
        "different_reg_params_Huber",
    )

plt.show()
