import linear_regression.sweeps.delta_out_sweep as dosw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_BO import f_BO, f_hat_BO_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np
from linear_regression.aux_functions.misc import excess_gen_error, estimation_error, gen_error_BO
from linear_regression.aux_functions.stability_functions import (
    stability_L2_decorrelated_regress,
    stability_L1_decorrelated_regress,
    stability_Huber_decorrelated_regress,
)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha, delta_in, percentage, beta = 10.0, 2.0, 0.1, 0.2
delta_out_min, delta_out_max, n_delta_out_pts = 0.01, 1_000, 300
n_delta_out_pts_BO = 150

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out_max * q:
        initial_condition = [m, q, sigma]
        break

fname_add = "_deltain_{:.1f}_percentage_{:.1f}_beta_{:.1f}".format(delta_in, percentage, beta)

delta_outs_l2, e_gen_l2, reg_params_opt_l2, (ms_l2, qs_l2, sigmas_l2) = dosw.sweep_delta_out_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    0.1,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_max,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
    decreasing=True,
)

np.savez(
    "delta_out_sweep_L2" + fname_add + ".npz",
    delta_outs=delta_outs_l2,
    e_gen=e_gen_l2,
    reg_params_opt=reg_params_opt_l2,
    ms=ms_l2,
    qs=qs_l2,
    sigmas=sigmas_l2,
)

print("L2 done")

delta_outs_l1, e_gen_l1, reg_params_opt_l1, (ms_l1, qs_l1, sigmas_l1) = dosw.sweep_delta_out_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L1_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    0.5,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_max,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
    decreasing=True,
)

np.savez(
    "delta_out_sweep_L1" + fname_add + ".npz",
    delta_outs=delta_outs_l1,
    e_gen=e_gen_l1,
    reg_params_opt=reg_params_opt_l1,
    ms=ms_l1,
    qs=qs_l1,
    sigmas=sigmas_l1,
)

print("L1 done")

(
    delta_outs_hub,
    e_gen_hub,
    (reg_params_opt_hub, hub_params_opt),
    (ms_hub, qs_hub, sigmas_hub),
) = dosw.sweep_delta_out_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    [0.5, 1.0],
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_max,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=initial_condition,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
    decreasing=True,
)

np.savez(
    "delta_out_sweep_Huber" + fname_add + ".npz",
    delta_outs=delta_outs_hub,
    e_gen=e_gen_hub,
    reg_params_opt=reg_params_opt_hub,
    hub_params_opt=hub_params_opt,
    ms=ms_hub,
    qs=qs_hub,
    sigmas=sigmas_hub,
)

print("Huber done")

delta_outs_BO, (gen_error_BO_old, qs_BO) = dosw.sweep_delta_out_fixed_point(
    f_BO,
    f_hat_BO_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts_BO,
    {"reg_param": 3.0},
    {"alpha": alpha, "delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    initial_cond=(0.6, 0.01, 0.9),
    funs=[gen_error_BO, q_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta}, {}],
    update_funs_args=[True, False],
    decreasing=True,
)

np.savez(
    "delta_out_sweep_BO" + fname_add + ".npz",
    delta_outs=delta_outs_BO,
    gen_error=gen_error_BO_old,
    qs=qs_BO,
)

print("BO done")

# ----------------------------

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    r"$\alpha = {}$, $\beta = {}$, $\epsilon = {}$, $\Delta_{{in}} = {}$".format(alpha, beta, percentage, delta_in)
)
plt.plot(delta_outs_l2, e_gen_l2, label="L2")
plt.plot(delta_outs_l1, e_gen_l1, label="L1")
plt.plot(delta_outs_hub, e_gen_hub, label="Huber")
plt.plot(delta_outs_BO, gen_error_BO_old, label="BO")

# plt.axvline(delta_in, color="black", linestyle="--")
plt.xlabel(r"$\Delta_{out}$")
plt.ylabel(r"$E_{gen}$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(delta_outs_l2, reg_params_opt_l2, label="L2")
plt.plot(delta_outs_l1, reg_params_opt_l1, label="L1")
plt.plot(delta_outs_hub, reg_params_opt_hub, label="Huber lambda")
plt.plot(delta_outs_hub, hub_params_opt, label="Huber a")

# plt.axvline(delta_in, color="black", linestyle="--")
plt.ylim([-1, 10])
plt.xlabel(r"$\Delta_{out}$")
plt.ylabel(r"$\lambda_{opt}$")
plt.xscale("log")
plt.legend()
plt.grid()

# plt.subplot(313)
# # plt.plot(delta_outs_l2, np.arccos(ms_l2 / np.sqrt(qs_l2)) / np.pi, label="L2 angle")
# # plt.plot(delta_outs_l2, qs_l2, label="L2 q")
# # plt.plot(delta_outs_l2, ms_l2, label="L2 m")
# # plt.plot(delta_outs_l1, np.arccos(ms_l1 / np.sqrt(qs_l1)) / np.pi, label="L1 angle")
# plt.plot(delta_outs_hub, ms_hub, label="Huber m")
# plt.plot(delta_outs_hub, qs_hub, label="Huber q")
# plt.plot(delta_outs_hub, sigmas_hub, label="Huber sigma")
# plt.legend()
# plt.grid()

# # plt.plot(epsilons_BO, gen_error_BO_old, label="BO")

# # plt.axvline(delta_in, color="black", linestyle="--")
# plt.xlabel(r"$\epsilon$")
# plt.ylabel("Angle")
# # plt.ylim(1e-2, 1e0)
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.grid()

plt.show()
