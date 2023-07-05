import linear_regression.sweeps.eps_sweep as epsw
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L2_loss import f_hat_L2_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_BO import f_BO, f_hat_BO_decorrelated_noise
import numpy as np
from linear_regression.aux_functions.misc import estimation_error
from scipy.optimize import curve_fit


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha, delta_in, delta_out, beta = 100.0, 1.0, 5.0, 0.0
eps_min, eps_max, n_eps_pts = 0.0000000001, 0.0001, 200

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        initial_condition = [m, q, sigma]
        break

(
    epsilons,
    e_gen_hub,
    (reg_params_opt_hub, hub_params_opt),
    (ms_hub, qs_hub, sigmas_hub),
) = epsw.sweep_eps_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    [0.5, 1.0],
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=initial_condition,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[list(), list(), list()],
    decreasing=True,
)

print("Huber done")

mhats_hub, qhats_hub, sigmahats_hub = np.empty_like(ms_hub), np.empty_like(qs_hub), np.empty_like(sigmas_hub)
for idx, (m, q, sigma) in enumerate(zip(ms_hub, qs_hub, sigmas_hub)):
    mhats_hub[idx], qhats_hub[idx], sigmahats_hub[idx] = f_hat_Huber_decorrelated_noise(
        m, q, sigma, alpha, delta_in, delta_out, epsilons[idx], beta, hub_params_opt[idx]
    )

np.savetxt(
    "sweep_eps_pres_huber_very_small.csv",
    np.vstack([epsilons, e_gen_hub, reg_params_opt_hub, hub_params_opt, ms_hub, qs_hub, sigmas_hub, mhats_hub, qhats_hub, sigmahats_hub]).T,
    delimiter=",",
    header="epsilons, e_gen_hub, reg_params_opt_hub, hub_params_opt, ms_hub, qs_hub, sigmas_hub, mhats_hub, qhats_hub, sigmahats_hub",
)

print("hat evaluated")

m_0, q_0, sigma_0 = fixed_point_finder(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    initial_condition,
    {"reg_param": delta_in},
    {"alpha": alpha, "delta_in": delta_in, "delta_out": delta_out, "percentage": 0.0, "beta": beta},
)

mhat_0, qhat_0, sigmahat_0 = f_hat_L2_decorrelated_noise(
    m_0, q_0, sigma_0, alpha, delta_in, delta_out, 0.0, beta
)

print(f"m_0 = {m_0}, q_0 = {q_0}, sigma_0 = {sigma_0}, mhat_0 = {mhat_0}, qhat_0 = {qhat_0}, sigmahat_0 = {sigmahat_0}")

# epsilons, (e_gen_BO,) = epsw.sweep_eps_fixed_point(
#     f_BO,
#     f_hat_BO_decorrelated_noise,
#     eps_min,
#     eps_max,
#     n_eps_pts,
#     {"reg_param": 3.0},
#     {
#         "alpha": alpha,
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": 0.3,
#         "beta": beta,
#     },
#     initial_cond=initial_condition,
#     funs=[estimation_error],
#     funs_args=[{}],
#     update_funs_args=None,
#     decreasing=True,
# )

print("BO done")

mhat_0 = alpha / (sigma_0 + 1)  # np.sqrt(2 * (delta_in + 1 + q_0 - 2 * m_0))
print(f"mhat_0 = {mhat_0}, mhat_Hub = {mhats_hub[0]}")

A = (sigma_0 + 1) / np.sqrt(2 * (delta_in + 1 + q_0 - 2 * m_0))
print(f"B0 = {A}")

# print(A**2 * alpha / (sigma_0 + 1))

# ----------------------------

# plt.figure(figsize=(7,7))

# # plt.plot(epsilons, np.abs((sigmas_hub - sigma_0)), label=r"$\Sigma$")
# # plt.plot(epsilons, np.abs((reg_params_opt_hub - delta_in)), label=r"$\lambda$")
# # plt.plot(epsilons, np.abs((reg_params_opt_hub - delta_in) / (sigmas_hub - sigma_0)))
# # plt.plot(epsilons, np.abs((sigma - sigma_0) / (ms)))
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.xlabel(r"$\epsilon$")
# plt.grid()

# plt.show()

# for ys, y0, s, params in zip(
#     [ms_hub, qs_hub, sigmas_hub, reg_params_opt_hub, mhats_hub, qhats_hub, sigmahats_hub],
#     [m_0, q_0, sigma_0, delta_in, mhat_0, qhat_0, sigmahat_0],
#     ["m", "q", "sigma", "lambda", "mhat", "qhat", "sigmahat"],
#     [
#         [0.97952813, -0.97952813, 0.5],
#         [0.98198347, -1.2219939, 0.5],
#         [1.07055057, -1.24931989, 0.5],
#         [0.9841116, 1.85911129, 0.5],
#         [1.06805681, 2.74581722, 0.5],
#         [1.06805681, 2.74581722, 0.5],
#         [1.11703011, 2.79551758, 0.5],
#     ],
# ):
#     plt.figure(figsize=(7, 7))

#     # Define the linear function to fit
#     def linear_function(x, a, b, c):
#         return -b * (x**a) * (np.log10(x) ** c)
    
#     lower_bounds, upper_bounds = [0.5, -1.0, 0.4], [1.5, 1.0, 0.6]
#     # Perform the curve fitting
#     # coeff = np.polyfit(np.log10(epsilons), np.log10(np.abs(ys - y0)), 1)
#     params, cov_matrix = curve_fit(linear_function, epsilons, ys - y0, p0=params, bounds=(lower_bounds, upper_bounds))

#     print(f"{s} coeff = {params}, cov = {np.sqrt(cov_matrix)}")
#     # poly = np.poly1d(params)
#     log_y_fitted = linear_function(epsilons, *params)

#     plt.subplot(211)
#     plt.title(s)
#     # plt.plot(epsilons, e_gen_hub - e_gen_BO, ".-", label="diff")
#     plt.plot(epsilons, (ys - y0), ".", label="m")
#     plt.plot(epsilons, log_y_fitted, "-", label="fit")

#     plt.xlabel(r"$\epsilon$")
#     plt.xscale("log")
#     # plt.yscale("log")
#     plt.grid()
#     plt.legend()

#     plt.subplot(212)
#     plt.plot(epsilons, (log_y_fitted - (ys - y0)), ".")
#     plt.xscale("log")
#     plt.grid()

#     plt.show()

# plt.plot(epsilons, np.abs(qs_hub - q_0), ".-", label="q")
# plt.plot(epsilons, np.abs(sigmas_hub - sigma_0), ".-", label="sigma")
# plt.plot(epsilons, np.abs(reg_params_opt_hub - delta_in), ".-", label="lambda")

# print(f"fits m : {}")
# print(f"fits q : {np.polyfit(np.log(epsilons), np.log(np.abs(qs_hub - q_0)), 1)}")
# print(f"fits sigma : {np.polyfit(np.log(epsilons), np.log(np.abs(sigmas_hub - sigma_0)), 1)}")
# print(f"fits lambda : {np.polyfit(np.log(epsilons), np.log(np.abs(reg_params_opt_hub - delta_in)), 1)}")

# # plt.plot(epsilons, np.abs(mhats_hub - mhat_0), ".-", label="mhat")
# # plt.plot(epsilons, np.abs(qhats_hub - qhat_0), ".-", label="qhat")
# # plt.plot(epsilons, np.abs(sigmahats_hub - sigmahat_0), ".-", label="sigmahat")


# print(f"lambda at 0 : {reg_params_opt_hub[0]}")
# # plt.plot(epsilons, e_gen_BO, label="BO")
# plt.xlabel(r"$\epsilon$")
# plt.ylabel(r"$E_{gen}$")
# plt.xscale("log")
# plt.yscale("log")
# plt.grid()
# plt.legend()


# x, y = np.sqrt(np.log10(1 / epsilons) + 1.0), hub_params_opt

x, y = epsilons, e_gen_hub

degree = 1
coefficients = np.polyfit(x, y, degree)

print(f"coefficients = {coefficients}")

# Create a polynomial object from the coefficients
poly = np.poly1d(coefficients)

# Generate data for the fitted line
x_fitted = np.linspace(0, 5, 100)
y_fitted = poly(x)

plt.figure(figsize=(7,7))

plt.subplot(211)
plt.xscale("log")
plt.plot(x, y, label="data")
plt.plot(x, poly(x), label="fit")
plt.legend()
plt.grid()

plt.subplot(212)
# plt.plot(epsilons, reg_params_opt_hub, label="lambda")
# plt.plot(x, y, ".", label="a")
# plt.plot(x_fitted, y_fitted, label="fit")
plt.plot(x, (y - poly(x)), ".", label="diff")
plt.xscale("log")
# plt.xlabel(r"$\epsilon$")
# plt.ylabel(r"$\lambda_{opt}$")
# plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.grid()

# print(f"max diff : {np.max(np.abs(y - poly(x)))}")

plt.show()
