import numpy as np
import matplotlib.pyplot as plt
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
    data_generation_hastie,
)
from linear_regression.erm.metrics import (
    percentage_flipped_labels_estim,
    percentage_error_from_true,
)
from cvxpy import Variable, Minimize, Problem, norm
from scipy.optimize import minimize_scalar
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
)
from tqdm.auto import tqdm
import os
import sys
from itertools import product
import warnings
import pickle

warnings.filterwarnings("error")

eps_init, alpha, gamma, reg_param, eps_training = (1.0, 1.5, 2.5, 1e-2, 0.0)

# DO NOT CHANGE, NOT IMPLEMENTED FOR OTHERS
pstar_t = 1.0

dimensions = [int(2**a) for a in range(5, 10)]
reps = 50

vals = np.zeros((len(dimensions), reps))
vals_small = np.zeros((len(dimensions), reps))

qs = np.zeros((len(dimensions), reps))
qs_feature = np.zeros((len(dimensions), reps))
ms = np.zeros((len(dimensions), reps))

for k, d in enumerate(tqdm(dimensions, desc="dim", leave=False)):
    p = int(d / gamma)
    n = int(d * alpha)

    print(f"d: {d}, n: {n}, p: {p}")

    eps = eps_init / d

    j = 0
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
            measure_gen_no_noise_clasif,
            d=d,
            n=max(n, 1),
            n_gen=1000,
            measure_fun_args={},
            gamma=gamma,
            noi=True,
        )

        try:
            if eps_training == 0.0:
                w = find_coefficients_Logistic(ys, xs, reg_param)
            else:
                w = find_coefficients_Logistic_adv(
                    ys, xs, 0.5 * reg_param, eps_training, 2.0, pstar_t, F @ wstar
                )
        except ValueError as e:
            print("Error in finding coefficients:", e)
            continue

        # np.sum(w**2) / p +
        qs[k, j] = np.dot(F.T @ w, F.T @ w) / d
        qs_feature[k, j] = np.dot(w, w) / p
        ms[k, j] = np.dot(wstar, F.T @ w) / d

        yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)
        yhat_gen = np.sign(np.dot(xs_gen, w))

        delta = Variable(d)
        constraints = [norm(delta, "inf") <= eps, wstar.T @ delta == 0]

        wtilde = F.T @ w
        objective = Minimize(wtilde.T @ delta)

        problem = Problem(objective, constraints)
        problem.solve()

        # the value at the minimum
        vals[k, j] = problem.value

        def min_fun(kappa):
            return np.sum(np.abs(kappa * wstar + F.T @ w))

        res = minimize_scalar(min_fun, bracket=(-5e10, 5e10), method="brent", tol=1e-10)

        # value at the minimum
        vals_small[k, j] = -eps * res.fun

        j += 1

plt.errorbar(
    dimensions,
    np.mean(vals, axis=1),
    yerr=np.std(vals, axis=1),
    label="Linf",
    marker="o",
    markersize=5,
    linestyle="--",
)
plt.errorbar(
    dimensions + 2 * np.ones(len(dimensions)),
    np.mean(vals_small, axis=1),
    yerr=np.std(vals_small, axis=1),
    label="small",
    marker="o",
    markersize=5,
    linestyle="--",
)

# print(f"qs: {np.mean(qs)}, ms: {np.mean(ms)}, qs_feature: {np.mean(qs_feature)}")

# if gamma <= 1.0:
#     plt.axhline(
#         y=-eps_init * np.sqrt(np.mean(qs) - np.mean(ms) ** 2) * np.sqrt(2 / np.pi),
#         color="black",
#         linestyle="--",
#     )
# else:
#     # opt = minimize_scalar(
#     #     lambda kappa: np.sqrt(
#     #         (np.mean(qs) - gamma**2 * np.mean(ms) ** 2) + (np.mean(ms) + kappa / gamma) ** 2
#     #     )
#     #     + np.abs(kappa) * (1 - 1 / gamma),
#     #     bracket=(-5e3, 5e3),
#     #     method="brent",
#     # )
#     # optimal = opt.fun
#     # plt.axhline(
#     #     y=-eps_init * optimal * np.sqrt(2 / np.pi),
#     #     color="black",
#     #     linestyle="--",
#     # )
#     plt.axhline(
#         y=-eps_init * np.sqrt(np.mean(qs_feature) - np.mean(ms) ** 2) * np.sqrt(2 / np.pi) / gamma,
#         color="black",
#         linestyle="--",
#     )
plt.xlabel("d")
plt.xscale("log", base=2)
plt.legend()
plt.show()
