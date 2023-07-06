from scipy.signal import find_peaks
from linear_regression.fixed_point_equations.classification.Logistic_loss import f_hat_Logistic_no_noise_classif
from linear_regression.sweeps.q_sweeps import sweep_q_fixed_point_proj_denoiser
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import f_projection_denoising
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.aux_functions.training_errors import training_error_Logistic_loss_no_noise
import numpy as np
import matplotlib.pyplot as plt

q_min, q_max, n_q_pts = 1, 100, 50
alpha = 2.0

m, q, sigma = 0.3, 10.0, 0.9

qs, (training_error_Logistic, angle_ts_Logistic) = sweep_q_fixed_point_proj_denoiser(
    f_hat_Logistic_no_noise_classif,
    q_min,
    q_max,
    n_q_pts,
    {"q_fixed": q_min},
    {"alpha": alpha},
    initial_cond_fpe=(m, q, sigma),
    funs=[training_error_Logistic_loss_no_noise, angle_teacher_student],
    funs_args=[{}, {}],
    update_funs_args=[False, False],
)

m_true, q_true, sigma_true = fixed_point_finder(
    f_L2_reg,
    f_hat_Logistic_no_noise_classif,
    (m, q, sigma),
    {"reg_param": 0.0001},
    {"alpha": alpha},
)

print("m_true = {:.7f} q_true = {:.7f} sigma_true = {:.7f}".format(m_true, q_true, sigma_true))


# create a plot with an upper panel with the training error and a lower panel with the angle
plt.figure(figsize=(10, 7.5))
plt.title("Logistic Loss")
plt.subplot(2, 1, 1)
plt.plot(qs, training_error_Logistic, label="Logistic")
plt.axvline(x=q_true, color="k", linestyle="--", label="True q")
plt.xlabel("q")
plt.ylabel("Training Error")
plt.xscale("log")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(qs, angle_ts_Logistic, label="Logistic")
plt.axvline(x=q_true, color="k", linestyle="--", label="True q")
plt.xlabel("q")
plt.ylabel("Angle")
plt.xscale("log")
plt.legend()
plt.grid()

plt.show()
