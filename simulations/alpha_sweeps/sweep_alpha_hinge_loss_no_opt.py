import linear_regression.sweeps.alpha_sweeps as alsw
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
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO, angle_teacher_student
from linear_regression.fixed_point_equations.fpe_Hinge_loss import f_hat_Hinge_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np
from linear_regression.aux_functions.stability_functions import (
    stability_L2_decorrelated_regress,
    stability_L1_decorrelated_regress,
    stability_Huber_decorrelated_regress,
)


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 2.0, 0.3, 1.0
alpha_min, alpha_max, n_alpha_pts = 0.01, 1000, 50
reg_param = 30.0
delta_eff = (1 - percentage) * delta_in + percentage * delta_out

fname = "./simulations/data/{}_fixed_lambda_{:.2e}_delta_in_{:.2e}_delta_out_{:.2e}_percentage_{:.2e}_beta_{:.2e}"

(
    alphas_Hinge,
    (estim_error_Hinge, sigmas_Hinge, qs_Hinge, ms_Hinge),
) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_Hinge_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.07780, 0.29, 0.89),
    funs=[estimation_error, sigma_order_param, q_order_param, m_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta}, {}, {}, {}],
)


np.savez(
    fname.format("Hinge", reg_param, delta_in, delta_out, percentage, beta),
    alphas=alphas_Hinge,
    gen_error=estim_error_Hinge,
    sigmas=sigmas_Hinge,
    qs=qs_Hinge,
    ms=ms_Hinge,
)

dat = np.load(fname.format("Hinge", reg_param, delta_in, delta_out, percentage, beta) + ".npz")

# create a figure with two vertical subplots
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title("Hinge regression, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(delta_in, delta_out, beta))
plt.plot(dat["alphas"], dat["gen_error"], '-.')
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{estim}$")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(dat["alphas"], np.arccos(dat["ms"]/ np.sqrt(dat["qs"])) / np.pi, "-.",label=r"Angle $\theta$")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\theta / \pi$")
plt.xscale("log")
plt.yscale("log")
plt.grid()


plt.show()