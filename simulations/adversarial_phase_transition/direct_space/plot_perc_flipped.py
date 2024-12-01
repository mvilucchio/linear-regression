import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_direct_space,
)
import pickle
import os

alpha = 1.5
reg_param = 1e-3
ps = ["inf", 2, 3]
dimensions = [int(2**a) for a in range(10, 12)]
reps = 10
epsilon_t = 0.0
pstar_t = 1.0

eps_dense = np.logspace(-1.5, 1.5, 100)

colors = [f"C{i}" for i in range(len(dimensions))]
linestyles = ["-", "--", "-.", ":"]
markers = [".", "x", "1", "2", "+", "3", "4"]
assert len(linestyles) >= len(ps)
assert len(markers) >= len(ps)

data_folder = "./data/direct_space"
file_name = f"ERM_direct_space_perc_flipped_n_features_{{:d}}_alpha_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{{:.1e}}_eps_t_{{:.2f}}_pstar_t_{{}}.pkl"

for p, ls, mrk in zip(tqdm(ps, desc="p", leave=False), linestyles, markers):

    for n_features, c in zip(tqdm(dimensions, desc="n", leave=False), colors):

        with open(
            os.path.join(
                data_folder, file_name.format(n_features, alpha, p, reg_param, epsilon_t, pstar_t)
            ),
            "rb",
        ) as f:
            data = pickle.load(f)
            epss = data["epss"]
            vals = data["vals"]
            estim_vals_m = data["estim_vals_m"]
            estim_vals_q = data["estim_vals_q"]
            estim_vals_rho = data["estim_vals_rho"]
            mean_m = data["mean_m"]
            std_m = data["std_m"]
            mean_q = data["mean_q"]
            std_q = data["std_q"]
            mean_rho = data["mean_rho"]

        plt.errorbar(
            epss,
            np.mean(vals, axis=0),
            yerr=np.std(vals, axis=0),
            # label=f"n_features = {n_features:d}",
            marker=mrk,
            linestyle="None",
            color=c,
        )

    out = np.empty_like(eps_dense)

    for i, eps in enumerate(tqdm(eps_dense, desc="eps", leave=False)):
        out[i] = percentage_flipped_direct_space(mean_m, mean_q, mean_rho, eps, p)

    plt.plot(eps_dense, out, label=f"p = {p}", linestyle=ls, color="black", linewidth=1)

plt.title(f"Direct space $\\alpha$ = {alpha:.1f} $\\lambda$ = {reg_param:.1e}")
plt.xscale("log")
plt.xlabel(r"$\epsilon (\sqrt[p]{d} / \sqrt{d})$")
plt.ylabel("Percentage of flipped labels")
plt.grid()

handles = []
labels = []
for p, ls, mrk in zip(ps, linestyles, markers):
    handle = plt.Line2D([], [], linestyle=ls, linewidth=0.5, marker=mrk, color="black")
    handles.append(handle)
    labels.append(f"p = {p}")

for dim, c in zip(dimensions, colors):
    handle_dim = plt.Line2D([], [], linestyle="None", marker="o", color=c)
    handles.append(handle_dim)
    labels.append(f"d = {dim:d}")

plt.legend(handles, labels)

plt.show()
