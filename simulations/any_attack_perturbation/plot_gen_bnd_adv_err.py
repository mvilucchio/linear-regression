import matplotlib.pyplot as plt
import numpy as np
from os.path import join, exists
import os

IMGS_FOLDER = "./imgs"

alpha_min, alpha_max, n_alpha_pts = 0.01, 1, 50
reg_orders = [1, 2, 3]
eps_t = 0.3
eps_g = eps_t
reg_param = 1e-3
pstar = 1

run_experiments = False

data_folder = "./data/SE_any_norm"

file_name = f"SE_data_pstar_{pstar}_reg_order_{{}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.csv"

plt.style.use("./latex_ready.mplstyle")

columnwidth = 234.8775
fig_width_pt = columnwidth
inches_per_pt = 1.0 / 72.27
figwidth = fig_width_pt * inches_per_pt
figheight = figwidth * (5.0**0.5 - 1.0) / 2.0

# Test two subplots
fig, axs = plt.subplots(
    2,
    1,
    figsize=(figwidth, figheight),
    sharex=True,
    layout="constrained",
    gridspec_kw={"hspace": 0.0},
)
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

plt.style.use("./latex_ready.mplstyle")

for i, reg_order in enumerate(reg_orders):
    with open(join(data_folder, file_name.format(reg_order)), "rb") as f:
        data_se = np.loadtxt(f, delimiter=",", skiprows=1)

    alphas_se = data_se[:, 0]
    ms_se = data_se[:, 1]
    qs_se = data_se[:, 2]
    Vs_se = data_se[:, 3]
    Ps_se = data_se[:, 4]
    estim_errors_se = data_se[:, 5]
    adv_errors_se = data_se[:, 6]
    gen_errors_se = data_se[:, 7]

    axs[0].plot(
        alphas_se,
        gen_errors_se,
        "--",
        color=f"C{i}",
        label=f"$r$ = {reg_order}",
    )

    axs[0].plot(
        alphas_se,
        adv_errors_se,
        "-",
        color=f"C{i}",
        label=f"$r$ = {reg_order}",
    )

    axs[1].plot(
        alphas_se,
        adv_errors_se - gen_errors_se,
        "-",
        color=f"C{i}",
        label=f"$r$ = {reg_order}",
    )

names = [
    # "$E_{\\mathrm{gen}}$",
    "$E_{\\mathrm{adv}}$/$E_{\\mathrm{gen}}$",
    # "$E_{\\mathrm{adv}}$",
    # "$E$",
    # "$E_{\\mathrm{bnd}} = E_{\\mathrm{adv}} - E_{\\mathrm{gen}}$",
    "$E_{\\mathrm{bnd}}$",
]
limits = [[0.25, 0.6], [0.007, 0.13]]
x_lims = [[1e-2, 1], [1e-2, 1]]

for i, (nn, lms, xl) in enumerate(zip(names, limits, x_lims)):
    ax = axs[i]
    if i == 0:
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )
    if i == 1:
        ax.set_xlabel(r"$\alpha$", labelpad=0)
    ax.set_ylabel(nn)
    ax.set_xscale("log")
    ax.set_ylim(lms)
    ax.set_xlim(xl)
    if i == 1:
        ax.set_yscale("log")
        ax.legend()
    ax.grid(which="both")

# Add a legend in the first subplot where the lines are respectively dashed and solid with names as in legend_names with color 'k'
legend_names = [
    ("$E_{\\mathrm{gen}}$", "k", "--"),
    ("$E_{\\mathrm{adv}}$", "k", "-"),
]
axs[0].legend(
    handles=[
        plt.Line2D([0], [0], color=c, lw=1, linestyle=ls, label=l) for l, c, ls in legend_names
    ],
    loc="best",
    fontsize=8,
    ncol=2,
)

if not exists(IMGS_FOLDER):
    os.makedirs(IMGS_FOLDER)

plt.savefig(
    join(IMGS_FOLDER, f"SE_adv_gen_bnd_reg_param_{reg_param:.1e}_eps_{eps_t:.1e}.pdf"),
    format="pdf",
    dpi=300,
)


plt.show()
