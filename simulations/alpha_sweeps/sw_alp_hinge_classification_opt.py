import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.classification.Hinge_loss import f_hat_Hinge_single_noise_classif, f_hat_Hinge_no_noise_classif
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_noise = 1.0
alpha_min, alpha_max, n_alpha_pts = 1, 10, 10

fname = "./simulations/data/{}_lambda_opt_delta_{:.2e}"

(
    alphas_Hinge,
    angle_ts_Hinge,
    reg_param_opt_Hinge,
    (sigmas_Hinge, qs_Hinge, ms_Hinge),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_Hinge_no_noise_classif,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    0.0001,
    {"reg_param": 0.01},
    {}, # {"delta": delta_noise},
    initial_cond_fpe=(0.01, 0.1, 0.9),
    f_min=angle_teacher_student,
    f_min_args={},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-5,
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

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title("Hinge regression, $\\alpha$ sweep, $\\Delta = {}$".format(delta_noise))
plt.plot(dat["alphas"], dat["angle_ts"], "o-")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"Angle $\theta$")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(dat["alphas"], dat["reg_param_opt"], "o-", label=r"$\lambda$")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\lambda$")
plt.xscale("log")
plt.grid()


plt.show()
