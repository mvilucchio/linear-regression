from linear_regression.data.generation import data_generation, measure_gen_decorrelated
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import vectorize, float32, float64
from tqdm import tqdm
import pickle
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (
    f_hat_mod_Tukey_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.amp.amp_funcs import GAMP_fullyTAP
from linear_regression.aux_functions.moreau_proximals import proximal_Tukey_modified_quad
import linear_regression.aux_functions.prior_regularization_funcs as priors
import linear_regression.aux_functions.likelihood_channel_functions as like
import numpy as np


def V_order_param(m, q, V):
    return V


def q_order_param(m, q, V):
    return q


def m_order_param(m, q, V):
    return m


@vectorize(
    [
        float32(float32, float32, float32, float32, float32),
        float64(float64, float64, float64, float64, float64),
    ]
)
def f_out_mod_Tukey(y: float, omega: float, V: float, tau: float, c: float) -> float:
    return (proximal_Tukey_modified_quad(y, omega, V, tau, c) - omega) / V


alpha_min, alpha_max = 2.16, 20
n_alpha_pts = 15
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.0
reg_param = 0.1
tau = 1.0
c = 0.01

data_folder = "./data/mod_Tukey_decorrelated_noise"
file_name = f"test_se_tukey_{alpha_min}_{alpha_max}_{n_alpha_pts}_{delta_in}_{delta_out}_{percentage}_{beta}_{reg_param}_{tau}_{c}.pkl"


# check if the file exists, if this is the case do not run it
if not os.path.exists(os.path.join(data_folder, file_name)):
    # init_cond = (0.07304981546047486, 0.0760530419161838, 9.103303146814449)
    init_cond = (0.9677292548128574, 0.9526508037535942, 0.22209856977678497)

    (alphas_se, (ms_se, qs_se, Vs_se)) = alsw.sweep_alpha_fixed_point(
        f_L2_reg,
        f_hat_mod_Tukey_decorrelated_noise,
        alpha_min,
        alpha_max,
        n_alpha_pts,
        {"reg_param": reg_param},
        {
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
            "tau": tau,
            "c": c,
        },
        initial_cond_fpe=init_cond,
        funs=[m_order_param, q_order_param, V_order_param],
        funs_args=[{}, {}, {}],
        update_funs_args=None,
        decreasing=True,
    )

    m_hat_se = np.empty_like(alphas_se)
    q_hat_se = np.empty_like(alphas_se)
    V_hat_se = np.empty_like(alphas_se)

    for i in tqdm(range(n_alpha_pts), leave=False, desc="SE"):
        m_hat_se[i], q_hat_se[i], V_hat_se[i] = f_hat_mod_Tukey_decorrelated_noise(
            ms_se[i],
            qs_se[i],
            Vs_se[i],
            alphas_se[i],
            delta_in,
            delta_out,
            percentage,
            beta,
            tau,
            c,
        )

    data_se = {
        "alphas": alphas_se,
        "ms": ms_se,
        "qs": qs_se,
        "Vs": Vs_se,
        "m_hat": m_hat_se,
        "q_hat": q_hat_se,
        "V_hat": V_hat_se,
    }

    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data_se, f)

with open(os.path.join(data_folder, file_name), "rb") as f:
    data_se = pickle.load(f)
    alphas_se = data_se["alphas"]
    ms_se = data_se["ms"]
    qs_se = data_se["qs"]
    Vs_se = data_se["Vs"]
    m_hat_se = data_se["m_hat"]
    q_hat_se = data_se["q_hat"]
    V_hat_se = data_se["V_hat"]

print("Results SE")
for i in range(n_alpha_pts):
    print(f"alpha: {alphas_se[i]:.3f}, m: {ms_se[i]:.3f}, q: {qs_se[i]:.3f}, V: {Vs_se[i]:.3f}")

plt.figure()
# plt.plot(alphas_se, 1 - 2 * ms_se + qs_se, label="SE")
plt.plot(alphas_se, qs_se, label="q")
plt.plot(alphas_se, ms_se, label="m")
plt.plot(alphas_se, Vs_se, label="V")


n_alpha_pts = 10
# reps = 5
# ds = [500]
# for d in ds:
#     alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
#     ms = np.empty((n_alpha_pts, 2))
#     qs = np.empty((n_alpha_pts, 2))
#     estim_error = np.empty((n_alpha_pts, 2))
#     gen_error = np.empty((n_alpha_pts, 2))

#     for i, alpha in enumerate(tqdm(alphas)):
#         n = int(alpha * d)

#         m_list, q_list, estim_error_list, gen_error_list = [], [], [], []
#         for rep in tqdm(range(reps), leave=False):
#             xs, ys, xs_gen, ys_gen, wstar = data_generation(
#                 measure_gen_decorrelated,
#                 d,
#                 n,
#                 1000,
#                 (delta_in, delta_out, percentage, beta),
#             )

#             # try:
#             #     w = find_coefficients_Huber(
#             #         ys, xs, reg_param, tau, inital_w=wstar + np.random.randn(d) * 1.0
#             #     )
#             # except ValueError:
#             #     pass

#             w_init = ms_se[i] * wstar + np.sqrt(qs_se[i] - ms_se[i] ** 2) * np.random.randn(d)
#             # w_init = 10.0 * np.random.randn(d)

#             w = GAMP_fullyTAP(
#                 priors.f_w_L2_regularization,
#                 f_out_mod_Tukey,
#                 ys,
#                 xs,
#                 (reg_param,),
#                 (tau, c),
#                 w_init,
#                 (Vs_se[i], V_hat_se[i]),
#                 1e-4,
#                 1000,
#                 wstar=wstar,
#             )

#             m, q = np.dot(w, wstar) / d, np.sum(w**2) / d
#             m_list.append(m)
#             q_list.append(q)

#             estim_error_list.append(np.sum((w - wstar) ** 2) / d)
#             gen_error_list.append(np.mean((xs_gen @ w - ys_gen) ** 2))

#         ms[i] = np.mean(m_list), np.std(m_list)
#         qs[i] = np.mean(q_list), np.std(q_list)
#         estim_error[i] = np.mean(estim_error_list), np.std(estim_error_list)
#         gen_error[i] = np.mean(gen_error_list), np.std(gen_error_list)

#     print("Results ERM")
#     for i in range(n_alpha_pts):
#         print(f"alpha: {alphas[i]:.3f}, m: {ms[i, 0]:.3f}, q: {qs[i, 0]:.3f}")

#     plt.errorbar(alphas, estim_error[:, 0], yerr=estim_error[:, 1], fmt=".-", label=f"d = {d}")

plt.xscale("log")
plt.xlabel(r"$\alpha$")
plt.yscale("log")
plt.legend()

plt.show()
