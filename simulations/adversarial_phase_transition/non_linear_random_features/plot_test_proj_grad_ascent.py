import matplotlib.pyplot as plt
import numpy as np
import os
from math import gamma as gamma_fun
from math import erf
import pickle
from linear_regression.aux_functions.percentage_flipped import percentage_flipped_linear_features

gamma, alpha, eps_training = 1.0, 1.0, 0.0

pstar_t = 1.0
reg_param = 1e-3
ps = [np.float32("inf")]
dimensions = [int(2**a) for a in range(8, 10)]
epss_dense = np.logspace(-1.5, 1.5, 50)
reps = 5

data_folder = "./data/non_linear_random_features"
file_name = f"test_linear_ERM_non_linear_rf_perc_misclass_n_features_{{:d}}_alpha_{{:.1f}}_gamma_{{:.1f}}_reps_{reps:d}_p_{{}}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}_pstar_t_{pstar_t}.pkl"


for p in ps:
    for d in dimensions:

        with open(
            os.path.join(data_folder, file_name.format(d, alpha, gamma, p)),
            "rb",
        ) as f:
            data = pickle.load(f)

        vals = data["vals"]
        epss = data["epss"]

        m = data["mean_m"]
        q = data["mean_q"]
        rho = data["mean_rho"]

        plt.errorbar(
            epss,
            np.mean(vals, axis=0),
            yerr=np.std(vals, axis=0) / np.sqrt(reps),
            label=f"d = {d} p = {p}",
        )

        out = np.empty_like(epss_dense)
        for i, eps in enumerate(epss_dense):
            out[i] = percentage_flipped_linear_features(m, q, rho, eps, p)

        plt.plot(epss_dense, out, label=f"d = {d} p = {p} dense")

plt.xscale("log")
plt.xlabel(r"$\varepsilon$")
plt.ylabel("Percentage of misclassified samples")

plt.legend()
plt.show()
