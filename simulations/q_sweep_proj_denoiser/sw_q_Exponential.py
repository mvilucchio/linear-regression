from scipy.signal import find_peaks
from linear_regression.fixed_point_equations.classification.Exponential_loss import f_hat_Exponential_no_noise_classif
from linear_regression.sweeps.q_sweeps import sweep_q_fixed_point_proj_denoiser
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import f_projection_denoising
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.aux_functions.training_errors import training_error_Exponential_loss_no_noise
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

qs, (
    training_error_Exponential,
    angle_ts_Exponential,
    sigmas_Exponential,
    ms_Exponential,
) = sweep_q_fixed_point_proj_denoiser(
    f_hat_Exponential_no_noise_classif,
    q_min,
    q_max,
    n_q_pts,
    {"q_fixed": q_min},
    {"alpha": alpha},
    initial_cond_fpe=(m, q, sigma),
    funs=[training_error_Exponential_loss_no_noise, angle_teacher_student, sigma_order_param, m_order_param],
    funs_args=[{}, {}, {}, {}],
    update_funs_args=[False, False, False, False],
)

m_true, q_true, sigma_true = fixed_point_finder(
    f_L2_reg,
    f_hat_Exponential_no_noise_classif,
    (m, q, sigma),
    {"reg_param": reg_param},
    {"alpha": alpha},
)

print("m_true = {:.7f} q_true = {:.7f} sigma_true = {:.7f}".format(m_true, q_true, sigma_true))

stab_values = np.empty_like(qs)
for idx, (m, q, sigma) in enumerate(zip(ms_Exponential, qs, sigmas_Exponential)):
    print(idx)
    stab_values[idx] = stability_Exponential_no_noise_classif(q, m, sigma, alpha)

# create a plot with an upper panel with the training error and a lower panel with the angle
plt.figure(figsize=(10, 7.5))
plt.title("Exponential Loss")
plt.subplot(2, 1, 1)
plt.plot(qs, training_error_Exponential, label="Exponential")
plt.axvline(x=q_true, color="k", linestyle="--", label="q $\\lambda$ = {:.4f}".format(reg_param))
plt.plot(qs, stab_values, ".-", label="Stability")
plt.xlabel("q")
plt.ylabel("Training Error")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.subplot(2, 1, 2)
plt.plot(qs, angle_ts_Exponential, label="Exponential")
plt.axvline(x=q_true, color="k", linestyle="--", label="q $\\lambda$ = {:.4f}".format(reg_param))
plt.xlabel("q")
plt.ylabel("Angle")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.show()
