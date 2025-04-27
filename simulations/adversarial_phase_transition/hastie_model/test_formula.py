import numpy as np
import matplotlib.pyplot as plt
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation_hastie,
)
from scipy.optimize import minimize_scalar
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("error")

eps_init, gamma, reg_param, eps_training = (3.0, 1.0, 1e-2, 0.0)

# DO NOT CHANGE, NOT IMPLEMENTED FOR OTHERS
pstar_t = 1.0

d = int(2**10)
reps = 25

alphas = np.linspace(0.5, 2.0, 10)

vals = np.zeros((len(alphas), reps))
qs = np.zeros((len(alphas), reps))
qs_feature = np.zeros((len(alphas), reps))
ms = np.zeros((len(alphas), reps))


def min_fun(kappa, wstar, w, F):
    return np.sum(np.abs(kappa * wstar + F.T @ w))


for k, alpha in enumerate(tqdm(alphas, desc="alpha", leave=False)):
    p = int(d / gamma)
    n = int(d * alpha)

    # print(f"d: {d}, n: {n}, p: {p}")

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

        qs[k, j] = np.dot(F.T @ w, F.T @ w) / d
        qs_feature[k, j] = np.dot(w, w) / p
        ms[k, j] = np.dot(wstar, F.T @ w) / d

        res = minimize_scalar(min_fun, bracket=(-5e10, 5e10), method="brent", args=(wstar, w, F))

        vals[k, j] = -eps * res.fun

        j += 1

plt.errorbar(
    alphas,
    np.mean(vals, axis=1),
    yerr=np.std(vals, axis=1),
    label="empirical",
)

if gamma <= 1.0:
    plt.plot(
        alphas,
        -eps_init * np.sqrt(np.mean(qs, axis=1) - np.mean(ms, axis=1) ** 2) * np.sqrt(2 / np.pi),
        label="theoretical",
    )
else:
    plt.plot(
        alphas,
        -eps_init
        * np.sqrt(np.mean(qs_feature, axis=1) - np.mean(ms, axis=1) ** 2)
        * np.sqrt(2 / np.pi)
        / gamma,
        label="theoretical",
    )
plt.xlabel("alpha")
plt.legend()
plt.show()
