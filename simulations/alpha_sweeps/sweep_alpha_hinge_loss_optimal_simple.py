import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO, angle_teacher_student
from linear_regression.fixed_point_equations.regression.Hinge_loss import (
    f_hat_Hinge_decorrelated_noise,
    f_hat_Hinge_single_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np

def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_noise = 1.0
alpha_min, alpha_max, n_alpha_pts = 0.1, 10, 5

fname = "./simulations/data/{}_lambda_opt_delta_{:.2e}"

(
    alphas_Hinge,
    angle_ts_Hinge,
    reg_param_opt_Hinge,
    (sigmas_Hinge, qs_Hinge, ms_Hinge),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_Hinge_single_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    0.1,
    {"reg_param": 0.1},
    {"delta": delta_noise},
    initial_cond_fpe=(0.07780, 0.29, 0.89),
    f_min=angle_teacher_student,
    f_min_args={},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-6,
)


np.savez(
    fname.format("Hinge", delta_noise),
    alphas=alphas_Hinge,
    angle_ts=angle_ts_Hinge,
    reg_param_opt=reg_param_opt_Hinge,
    sigmas=sigmas_Hinge,
    qs=qs_Hinge,
    ms=ms_Hinge,
)

dat = np.load(fname.format("Hinge", delta_noise) + ".npz")

# create a figure with two vertical subplots
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title("Hinge regression, $\\alpha$ sweep, $\\Delta = {}$".format(delta_noise))
plt.plot(dat["alphas"], dat["angle_ts"], "-.")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"Angle $\theta$")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(dat["alphas"], dat["reg_param_opt"], "-.", label=r"$\lambda$")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\lambda$")
plt.xscale("log")
# plt.yscale("log")
plt.grid()


plt.show()
