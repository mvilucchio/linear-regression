import matplotlib.pyplot as plt
import numpy as np
from os.path import join, exists
from linear_regression.aux_functions.misc import classification_adversarial_error
import os


def log_log_linear_fit(x, y, base=10, return_points=False, extend_percent=0.1):
    x = np.array(x)
    y = np.array(y)

    log_x = np.log(x) / np.log(base)
    log_y = np.log(y) / np.log(base)

    A = np.vstack([log_x, np.ones(len(log_x))]).T
    m, c = np.linalg.lstsq(A, log_y, rcond=None)[0]

    coefficient = base**c

    if return_points:
        log_x_min, log_x_max = np.log10(min(x)), np.log10(max(x))
        log_x_range = log_x_max - log_x_min
        extended_log_x_min = log_x_min - extend_percent * log_x_range
        extended_log_x_max = log_x_max + extend_percent * log_x_range

        x_fit = np.logspace(extended_log_x_min, extended_log_x_max, 100)
        y_fit = coefficient * x_fit**m

        return m, coefficient, (x_fit, y_fit)
    else:
        return m, coefficient


IMGS_FOLDER = "./imgs"

alpha_min, alpha_max, n_alpha_pts = 0.005, 1, 250
reg_orders = [1, 2, 3]
eps_t = 0.3
eps_g = eps_t
reg_param = 1e-3
pstar = 1
alpha_cutoff = 0.008

run_experiments = True

data_folder = "./data/SE_any_norm"

file_name = f"SE_data_pstar_{pstar}_reg_order_{{}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}_overlaps.csv"

columnwidth = 469.75502
fig_width_pt = columnwidth
inches_per_pt = 1.0 / 72.27
figwidth = fig_width_pt * inches_per_pt
figheight = figwidth * (5.0**0.5 - 1.0) / 2.0

plt.style.use("./latex_ready.mplstyle")

# Test two subplots
fig, axs = plt.subplots(
    1,
    3,
    figsize=(figwidth, 0.6 * figheight),
    sharex=True,
    sharey=True,
    layout="constrained",
    gridspec_kw={"hspace": 0.0},
)
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)


for i, reg_order in enumerate(reg_orders):
    with open(join(data_folder, file_name.format(reg_order)), "rb") as f:
        data_se = np.loadtxt(f, delimiter=",", skiprows=1)

    alphas_se = data_se[:, 0]

    ms_se = data_se[:, 1]
    qs_se = data_se[:, 2]
    Vs_se = data_se[:, 3]
    ps_se = data_se[:, 4]

    mhats_se = data_se[:, 5]
    qhats_se = data_se[:, 6]
    Vhats_se = data_se[:, 7]
    phats_se = data_se[:, 8]

    adversarial_error = [
        classification_adversarial_error(m, q, p, eps_g, pstar)
        for m, q, p in zip(ms_se, qs_se, ps_se)
    ]
    generalisation_error = [np.arccos(m / np.sqrt(q)) / np.pi for m, q in zip(ms_se, qs_se)]

    overlaps_se = [ms_se, qs_se, ps_se, Vs_se, ms_se / np.sqrt(qs_se)]
    overlaps_hats_se = [mhats_se, qhats_se, phats_se, Vhats_se]

    names = ["$m$", "$q$", "$P$", "$V$", "$\\frac{{m}}{\\sqrt{q}}$"]

    for j, ov in enumerate(overlaps_se):
        axs[i].plot(
            alphas_se,
            ov,
            linestyle="-",
            color=f"C{j}",
            alpha=0.5,
        )
        ones_to_keep = np.where(alphas_se < alpha_cutoff)
        m, c, (x_lin, y_lin) = log_log_linear_fit(
            alphas_se[ones_to_keep], ov[ones_to_keep], return_points=True
        )
        axs[i].plot(
            x_lin,
            y_lin,
            linestyle="--",
            color=f"C{j}",
            label=f"{names[j]} $\\sim \\alpha^{{{m:.2f}}}$",
        )

    axs[i].set_xlabel(r"$\alpha$")
    axs[i].set_xscale("log")
    axs[i].set_yscale("log")
    axs[i].legend(loc="upper right")
    axs[i].grid(which="major", visible=True)
    axs[i].set_xlim((0.0036582940641780985, 1.4262742076220248))

    if i == 0:
        axs[i].set_ylabel("Overlaps")

    if i > 0:
        axs[i].tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

    axs[i].set_title(f"$r = {int(reg_order)}$")


plt.savefig(f"./imgs/overlaps_scalings_low_alpha_{reg_param:.1e}.pdf", format="pdf", dpi=300)

plt.show()
