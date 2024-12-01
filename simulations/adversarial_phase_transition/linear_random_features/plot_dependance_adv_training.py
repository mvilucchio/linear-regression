from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_linear_features,
)
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

alpha = 2.0
gamma = 2.0
reps = 10
pstar_t = 1.0
p = "inf"
reg_param = 1e-3

data_folder = "./data/linear_random_features"
file_name = f"ERM_linear_RF_adv_transition_n_features_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_p_{p}_reg_param_{reg_param:.1e}_eps_t_{{:.2f}}_pstar_t_{pstar_t}.pkl"
dimensions = [int(2**a) for a in range(9, 12)]

eps_trainigs = [0.0, 0.1, 0.2]
markers = [".", "x", "1", "2", "+", "3", "4"]


eps_dense = np.logspace(-1, 1, 100)
out = np.empty_like(eps_dense)

for idx, e_t in enumerate(eps_trainigs):

    for i, (dimension, mks) in enumerate(zip(dimensions, markers)):
        with open(os.path.join(data_folder, file_name.format(dimension, e_t)), "rb") as f:
            data = pickle.load(f)

            # data = {
            #     "epss": epss,
            #     "vals": vals,
            #     "mean_m": mean_m,
            #     "std_m": std_m,
            #     "mean_q": mean_q,
            #     "std_q": std_q,
            #     "mean_rho": mean_rho,
            #     "std_rho": std_rho,
            # }

            epss_g = data["epss"]
            vals_g = data["vals"]
            mean_m = data["mean_m"]
            std_m = data["std_m"]
            mean_q = data["mean_q"]
            std_q = data["std_q"]
            mean_rho = data["mean_rho"]
            std_rho = data["std_rho"]

        plt.errorbar(
            epss_g,
            np.mean(vals_g, axis=0),
            yerr=np.std(vals_g, axis=0),
            linestyle="",
            color=f"C{idx}",
            marker=mks,
            label=f"eps_t = {e_t}",
        )

    for j, eps_i in enumerate(eps_dense):
        out[j] = percentage_flipped_linear_features(mean_m, mean_q, mean_rho, eps_i, p, gamma)

    plt.plot(eps_dense, out, color=f"C{idx}")


plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.xlabel(r"$\epsilon_g (\sqrt[p]{d} / \sqrt{d})$")
plt.ylabel("Percentage of flipped labels")
plt.grid(which="both")

plt.show()
