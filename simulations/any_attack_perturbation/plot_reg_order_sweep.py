import os
import numpy as np
import matplotlib.pyplot as plt

data_folder_SE = "./data/SE_reg_order_sweep"

reg_param = 1e-4
pstar = 3
eps_vals = [0.01, 0.05333333, 0.09666667, 0.14, 0.18333333, 0.27, 0.31333333, 0.35666667]
# eps_vals = [0.01, 0.1, 0.2, 0.3, 0.4]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

file_names = [
    f"SE_reg_order_sweep_pstar_{pstar}_reg_order_1.0_3.0_alpha_0.100_reg_param_{reg_param:.1e}_eps_{eps:.2f}.csv"
    for eps in eps_vals
]

plt.style.use("./latex_ready.mplstyle")

columnwidth = 234.8775
fig_width_pt = columnwidth
inches_per_pt = 1.0 / 72.27
figwidth = fig_width_pt * inches_per_pt
figheight = figwidth * (5.0**0.5 - 1.0) / 2.0
fig, ax = plt.subplots(1, 1, figsize=(figwidth, figheight))

min_adversarial_errors = []
for i, (file_name, c, eps) in enumerate(zip(file_names, colors, eps_vals)):

    data = np.genfromtxt(os.path.join(data_folder_SE, file_name), delimiter=",", skip_header=1)

    reg_orders = data[:, 0]

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

    min_adversarial_error = np.min(adversarial_errors_found)
    min_adversarial_error_index = np.argmin(adversarial_errors_found)
    min_adversarial_errors.append(reg_orders[min_adversarial_error_index])

    # ax.plot(reg_orders, gen_errors_se, "-", color="tab:blue", linestyle=ls)
    ax.plot(reg_orders, adversarial_errors_found, "-", color=c)
    ax.plot(reg_orders[min_adversarial_error_index], min_adversarial_error, ".", color=c)

    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="none")
    xx = min(reg_orders) + 0.05 * (max(reg_orders) - min(reg_orders))
    ax.text(
        xx,
        adversarial_errors_found[-1],
        f"$\\varepsilon = $ {eps:.2f}",
        fontdict={"fontsize": 8},
        verticalalignment="top" if i > 2 else "bottom",
        horizontalalignment="left",
        bbox=props,
        color=c,
    )

ax.set_xlabel("$r$")
ax.set_ylabel("$E_{\\mathrm{adv}}$")
# ax.set_yscale("log")
# ax.set_ylim([0.2, 0.6])
# ax.legend([f"$\\varepsilon = {eps}$" for eps in eps_vals], loc="best")
ax.grid(True)
ax.set_xlim([min(reg_orders), max(reg_orders)])

ax.set_title(f"$p^\\star = {pstar:d}$")

additional_eps_vals = np.linspace(0.01, 0.4, 10)
print(additional_eps_vals)
additional_file_names = [
    f"SE_reg_order_sweep_pstar_{pstar}_reg_order_1.0_3.0_alpha_0.100_reg_param_{reg_param:.1e}_eps_{eps:.2f}.csv"
    for eps in additional_eps_vals
]

add_min_adversarial_errors = []
for file_name in additional_file_names:
    data = np.genfromtxt(os.path.join(data_folder_SE, file_name), delimiter=",", skip_header=1)

    reg_orders = data[:, 0]

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

    min_adversarial_error = np.min(adversarial_errors_found)
    min_adversarial_error_index = np.argmin(adversarial_errors_found)
    add_min_adversarial_errors.append(reg_orders[min_adversarial_error_index])

print("AAA")
axins = ax.inset_axes([0.4, 0.65, 0.3, 0.3])
axins.plot(additional_eps_vals, add_min_adversarial_errors, "k-")
axins.set_xlabel("$\\varepsilon$", fontsize=8, labelpad=0)
axins.set_ylabel("$r^\\star$", fontsize=8, labelpad=0)
axins.set_xlim([min(eps_vals), max(eps_vals)])
# axins.set_ylim([1, 2])
axins.tick_params(axis="both", which="major", labelsize=6)
axins.grid(True)

fig.set_constrained_layout(True)

plt.savefig(
    f"./imgs/reg_order_sweep_pstar_{pstar:d}_reg_param_{reg_param:.1e}.pdf", format="pdf", dpi=300
)

plt.show()
