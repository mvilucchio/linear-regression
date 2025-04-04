from linear_regression.data.generation import data_generation, measure_gen_decorrelated
from linear_regression.erm.erm_solvers import find_coefficients_mod_Tukey, find_coefficients_Huber
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle


import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regression.Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np


def V_order_param(m, q, V):
    return V


def q_order_param(m, q, V):
    return q


def m_order_param(m, q, V):
    return m


alpha_min, alpha_max = 0.1, 10
n_alpha_pts = 100
delta_in, delta_out, percentage, beta = 1.0, 1.0, 0.1, 0.0
reg_param = 0.1
a = 1.0

(alphas_se, (ms_se, qs_se, Vs_se)) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": a,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[m_order_param, q_order_param, V_order_param],
    funs_args=[{}, {}, {}],
    update_funs_args=None,
)

plt.figure()
plt.plot(alphas_se, 1 - 2 * ms_se + qs_se, label="SE")

reps = 10
n_alpha_pts = 5
d = 500

alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
ms = np.empty((n_alpha_pts, 2))
qs = np.empty((n_alpha_pts, 2))
estim_error = np.empty((n_alpha_pts, 2))
gen_error = np.empty((n_alpha_pts, 2))

for i, alpha in enumerate(tqdm(alphas)):
    n = int(alpha * d)

    m_list, q_list, estim_error_list, gen_error_list = [], [], [], []
    for rep in tqdm(range(reps), leave=False):
        xs, ys, xs_gen, ys_gen, wstar = data_generation(
            measure_gen_decorrelated,
            d,
            n,
            1000,
            (delta_in, delta_out, percentage, beta),
        )

        w = find_coefficients_Huber(ys, xs, reg_param, a, inital_w=wstar + np.random.randn(d) * 1.0)

        m, q = np.dot(w, wstar) / d, np.sum(w**2) / d
        m_list.append(m)
        q_list.append(q)

        estim_error_list.append(np.sum((w - wstar) ** 2) / d)
        gen_error_list.append(np.mean((xs_gen @ w - ys_gen) ** 2))

    ms[i] = np.mean(m_list), np.std(m_list)
    qs[i] = np.mean(q_list), np.std(q_list)
    estim_error[i] = np.mean(estim_error_list), np.std(estim_error_list)
    gen_error[i] = np.mean(gen_error_list), np.std(gen_error_list)


plt.errorbar(alphas, estim_error[:, 0], yerr=estim_error[:, 1], fmt=".-", label="estim error")
plt.xscale("log")
plt.xlabel(r"$\alpha$")
plt.yscale("log")
plt.legend()

plt.show()
