import matplotlib.pyplot as plt
import numpy as np
from os.path import join, exists
import os
import itertools

data_folder_SE = "./data/SE_eps_sweep_2"

eps_min, eps_max = 0.001, 0.3
alphas = [0.1, 1.0]
reg_orders = [1, 2]
reg_param = 1e-2
pstar = 1

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

file_names = [
    f"SE_eps_sweep_pstar_{pstar}_reg_order_{reg_order}_alpha_{alpha:.3f}_reg_param_{reg_param:.1e}_eps_{eps_min:.2f}_{eps_max:.2f}.csv"
    for alpha, reg_order in itertools.product(alphas, reg_orders)
]

# plt.style.use("./latex_ready.mplstyle")

columnwidth = 234.8775
fig_width_pt = columnwidth
inches_per_pt = 1.0 / 72.27
figwidth = fig_width_pt * inches_per_pt
figheight = figwidth * (5.0**0.5 - 1.0) / 2.0
fig, ax = plt.subplots(1, 1, figsize=(figwidth, 0.87 * figheight), layout="constrained")

for i, (file_name, c, (alpha, reg_order)) in enumerate(
    zip(file_names, colors, itertools.product(alphas, reg_orders))
):

    data = np.genfromtxt(os.path.join(data_folder_SE, file_name), delimiter=",", skip_header=1)

    eps_vals = data[:, 0]

    ms_found = data[:, 1]
    qs_found = data[:, 2]
    Vs_found = data[:, 3]
    Ps_found = data[:, 4]

    mhats_found = data[:, 5]
    qhats_found = data[:, 6]
    Vhats_found = data[:, 7]
    Phats_found = data[:, 8]

    estim_errors_se = data[:, 9]
    adversarial_errors_found = data[:, 10]
    gen_errors_se = data[:, 11]

    ax.plot(
        adversarial_errors_found,
        gen_errors_se,
        "-",
        color=c,
        label=f"$\\alpha = {alpha}$ $r = {reg_order}$",
    )

ax.set_xlabel("$E_{\mathrm{adv}}$")
ax.set_ylabel("$E_{\mathrm{gen}}$")

ax.legend()
ax.grid(which="major")

plt.savefig("gen_vs_adv_err.pdf", bbox_inches="tight")

plt.show()
