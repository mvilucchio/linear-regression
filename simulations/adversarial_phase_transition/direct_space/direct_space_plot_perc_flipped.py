import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_direct_space,
)
from math import gamma
import jax.numpy as jnp
import jax
from jax.scipy.optimize import minimize as jax_minimize
from jax import grad, vmap
import pickle


alpha = 1.5
reg_param = 1.0
ps = ["inf", 2, 3, 5]
dimensions = [int(2**a) for a in range(11, 12)]
reps = 20

eps_dense = np.logspace(-2, 2, 100)

colors = [f"C{i}" for i in range(len(dimensions))]
linestyles = ["-", "--", "-.", ":"]
markers = [".", "x", "1", "2", "+", "3", "4"]
assert len(linestyles) >= len(ps)
assert len(markers) >= len(ps)

for p, ls, mrk in zip(tqdm(ps, desc="p", leave=False), linestyles, markers):

    for n_features, c in zip(tqdm(dimensions, desc="n", leave=False), colors):

        with open(
            f"./data/n_features_{n_features:d}_reps_{reps:d}_p_{p}_alpha_{alpha:.1f}.pkl", "rb"
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
        if p == "inf":
            out[i] = percentage_flipped_direct_space(mean_m, mean_q, mean_rho, eps, 1)
        else:
            out[i] = percentage_flipped_direct_space(mean_m, mean_q, mean_rho, eps, p / (p - 1))

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
