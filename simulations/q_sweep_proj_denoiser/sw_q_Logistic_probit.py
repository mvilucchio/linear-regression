from scipy.signal import find_peaks
from linear_regression.fixed_point_equations.classification.Logistic_loss import f_hat_Logistic_probit_classif
from linear_regression.sweeps.q_sweeps import sweep_q_fixed_point_proj_denoiser
from linear_regression.aux_functions.stability_functions import stability_Logistic_probit_classif
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import f_projection_denoising
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.aux_functions.training_errors import training_error_Logistic_loss_probit
import numpy as np
import matplotlib.pyplot as plt


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


q_min, q_max, n_q_pts = 0.05, 10, 30
alpha = 1.0
reg_param = 0.001
delta_noises = [0.01, 0.1]

while True:
    m_init, q_init, sigma_init = np.random.rand(3)
    if m_init**2 < q_init:
        break

plt.figure(figsize=(10, 7.5))
plt.subplot(2, 1, 1)
plt.title("Logistic Loss $\\alpha$ = {:.2f}".format(alpha))

for delta_noise in delta_noises:
    qs, (training_error_Logistic, angle_ts_Logistic, sigmas_Logistic, ms_Logistic) = sweep_q_fixed_point_proj_denoiser(
        f_hat_Logistic_probit_classif,
        q_min,
        q_max,
        n_q_pts,
        {"q_fixed": q_min},
        {"alpha": alpha, "delta": delta_noise},
        initial_cond_fpe=(m_init, q_init, sigma_init),
        funs=[training_error_Logistic_loss_probit, angle_teacher_student, sigma_order_param, m_order_param],
        funs_args=[{"delta" : delta_noise}, {}, {}, {}],
        update_funs_args=[False, False, False, False],
    )

    plt.subplot(2, 1, 1)
    next_color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.plot(qs, training_error_Logistic, "-", color=next_color,  label="$\\Delta$ = {:.4f}".format(delta_noise))

    plt.subplot(2, 1, 2)
    plt.plot(qs, angle_ts_Logistic, "-", color=next_color, label="$\\Delta$ = {:.4f}".format(delta_noise))

plt.subplot(2, 1, 1)
plt.xlabel("q")
plt.ylabel("Training Error")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.subplot(2, 1, 2)
plt.xlabel("q")
plt.ylabel("Generalization Error")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.show()
