from scipy.signal import find_peaks
from linear_regression.fixed_point_equations.classification.Exponential_loss import f_hat_Exponential_probit_classif
from linear_regression.sweeps.q_sweeps import sweep_q_fixed_point_proj_denoiser
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import f_projection_denoising
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.aux_functions.training_errors import training_error_Exponential_loss_probit
from linear_regression.aux_functions.stability_functions import stability_Exponential_no_noise_classif
import numpy as np
import matplotlib.pyplot as plt


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


q_min, q_max, n_q_pts = 1, 100, 30
alpha = 2.0

m, q, sigma = 0.3, 10.0, 0.9
reg_param = 0.0001

delta_noises = [0.01, 0.1]

plt.figure(figsize=(10, 7.5))
plt.title("Exponential Loss $\\alpha$ = {:.2f}".format(alpha))

for delta_noise in delta_noises:
    qs, (
        training_error_Exponential,
        angle_ts_Exponential,
        sigmas_Exponential,
        ms_Exponential,
    ) = sweep_q_fixed_point_proj_denoiser(
        f_hat_Exponential_probit_classif,
        q_min,
        q_max,
        n_q_pts,
        {"q_fixed": q_min},
        {"alpha": alpha, "delta": delta_noise},
        initial_cond_fpe=(m, q, sigma),
        funs=[training_error_Exponential_loss_probit, angle_teacher_student, sigma_order_param, m_order_param],
        funs_args=[{"delta" : delta_noise}, {}, {}, {}],
        update_funs_args=[False, False, False, False],
    )
    
    plt.subplot(2, 1, 1)
    next_color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.plot(qs, training_error_Exponential, "-", color=next_color, label="$\\Delta$ = {:.4f}".format(delta_noise))

    plt.subplot(2, 1, 2)
    plt.plot(qs, angle_ts_Exponential, "-", color=next_color, label="$\\Delta$ = {:.4f}".format(delta_noise))


plt.subplot(2, 1, 1)
plt.xlabel("q")
plt.ylabel("Training Error")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.subplot(2, 1, 2)
plt.xlabel("q")
plt.ylabel("Generalisation Error")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.show()
