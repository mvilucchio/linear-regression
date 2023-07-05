import linear_regression.sweeps.eps_sweep as epsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_BO import f_BO, f_hat_BO_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO
import numpy as np


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha, delta_in, delta_out, beta = 10.0, 1.0, 1.0, 0.0
eps_min, eps_max, n_eps_pts = 0.0001, 1.0, 500
n_eps_pts_BO = 100
# n_eps_pts = n_eps_pts_BO

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        initial_condition = [m, q, sigma]
        break

fname_add = "_deltain_{}_deltaout_{}_beta_{}".format(delta_in, delta_out, beta)

epsilons_l2, e_gen_l2, reg_params_opt_l2, (ms_l2, qs_l2, sigmas_l2) = epsw.sweep_eps_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    delta_in,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
    linear=True
)

np.savez(
    "eps_sweep_L2_linear" + fname_add + ".npz",
    epsilons=epsilons_l2,
    e_gen=e_gen_l2,
    reg_params_opt=reg_params_opt_l2,
    ms=ms_l2,
    qs=qs_l2,
    sigmas=sigmas_l2,
)

print("L2 done")

epsilons_l1, e_gen_l1, reg_params_opt_l1, (ms_l1, qs_l1, sigmas_l1) = epsw.sweep_eps_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L1_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    0.5,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
    linear=True
)

np.savez(
    "eps_sweep_L1_linear" + fname_add + ".npz",
    epsilons=epsilons_l1,
    e_gen=e_gen_l1,
    reg_params_opt=reg_params_opt_l1,
    ms=ms_l1,
    qs=qs_l1,
    sigmas=sigmas_l1,
)

print("L1 done")

(
    epsilons_hub,
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
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
    linear=True
)

np.savez(
    "eps_sweep_Huber_linear" + fname_add + ".npz",
    epsilons=epsilons_hub,
    e_gen=e_gen_hub,
    reg_params_opt=reg_params_opt_hub,
    hub_params_opt=hub_params_opt,
    ms=ms_hub,
    qs=qs_hub,
    sigmas=sigmas_hub,
)

print("Huber done")

(
    epsilons_BO,
    (gen_error_BO_old, qs_BO),
) = epsw.sweep_eps_fixed_point(
    f_BO,
    f_hat_BO_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts_BO,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond=(0.6, 0.01, 0.9),
    funs=[gen_error_BO, q_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta}, {}],
    update_funs_args=[True, False],
    decreasing=False,
    linear=True
)

np.savez(
    "eps_sweep_BO_linear" + fname_add + ".npz",
    epsilons=epsilons_BO,
    e_gen=gen_error_BO_old,
    qs=qs_BO,
)

print("BO done")

# ----------------------------

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    r"$\alpha = {}$, $\beta = {}$, $\Delta_{{in}} = {}$, $\Delta_{{in}} = {}$".format(alpha, beta, delta_in, delta_out)
)
plt.plot(epsilons_l2, e_gen_l2, label="L2")
plt.plot(epsilons_l1, e_gen_l1, label="L1")
plt.plot(epsilons_hub, e_gen_hub, label="Huber")
plt.plot(epsilons_BO, gen_error_BO_old, label="BO")

plt.axvline(0.5, color="black", linestyle="--")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$E_{gen}^{excess}$")
plt.ylim(1e-2, 1e0)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(212)
# plt.plot(epsilons, e_gen_l2-gen_error_BO_old, label="L2")
# plt.plot(epsilons, e_gen_l1-gen_error_BO_old, label="L1")
# plt.plot(epsilons, e_gen_hub-gen_error_BO_old, label="Huber")
# plt.plot(epsilons_BO, gen_error_BO_old, label="BO")
plt.plot(epsilons_l2, reg_params_opt_l2, label="L2")
plt.plot(epsilons_l1, reg_params_opt_l1, label="L1")
plt.plot(epsilons_hub, reg_params_opt_hub, label="Huber $\\lambda$")
plt.plot(epsilons_hub, hub_params_opt, label="Huber $a$")

# plt.axvline(0.5, color="black", linestyle="--")
plt.xlabel(r"$\epsilon$")
# plt.xscale("log")
plt.ylim(-1, 7)
# plt.yscale("log")
plt.legend()
plt.grid()

# plt.subplot(313)
# plt.plot(epsilons_l2, np.arccos(ms_l2/np.sqrt(qs_l2)) / np.pi, '--', label="L2 angle")
# plt.plot(epsilons_l2, qs_l2, label="L2 q")
# plt.plot(epsilons_l2, ms_l2, label="L2 m")

# plt.plot(epsilons_l1, np.arccos(ms_l1/np.sqrt(qs_l1)) / np.pi, '--', label="L1 angle")

# plt.plot(epsilons_hub, np.arccos(ms_hub/np.sqrt(qs_hub)) / np.pi, '--', label="Huber angle")

# # plt.plot(epsilons_BO, gen_error_BO_old, label="BO")

# plt.axvline(0.5, color="black", linestyle="--")
# plt.xlabel(r"$\epsilon$")
# plt.ylim(1e-2, 1e0)
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.grid()


plt.show()
