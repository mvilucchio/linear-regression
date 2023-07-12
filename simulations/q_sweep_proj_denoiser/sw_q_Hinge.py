from scipy.signal import find_peaks
from linear_regression.fixed_point_equations.classification.Hinge_loss import f_hat_Hinge_no_noise_classif
from linear_regression.aux_functions.stability_functions import stability_Hinge_no_noise_classif
from linear_regression.sweeps.q_sweeps import sweep_q_fixed_point_proj_denoiser
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg

from linear_regression.aux_functions.training_errors import training_error_Hinge_loss_no_noise
import numpy as np
import matplotlib.pyplot as plt


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


q_min, q_max, n_q_pts = 0.01, 7, 200
alpha = 2.0

m, q, sigma = 0.3, 10.0, 0.9
# reg_param = 0.0001

qs, (training_error_Hinge, angle_ts_Hinge, sigmas_Hinge, ms_Hinge) = sweep_q_fixed_point_proj_denoiser(
    f_hat_Hinge_no_noise_classif,
    q_min,
    q_max,
    n_q_pts,
    {"q_fixed": q_min},
    {"alpha": alpha},
    initial_cond_fpe=(m, q, sigma),
    funs=[training_error_Hinge_loss_no_noise, angle_teacher_student, sigma_order_param, m_order_param],
    funs_args=[{}, {}, {}, {}],
    update_funs_args=[False, False, False, False],
)

stab_values = np.empty_like(qs)
for idx, (m, q, sigma) in enumerate(zip(ms_Hinge, qs, sigmas_Hinge)):
    stab_values[idx] = stability_Hinge_no_noise_classif(q, m, sigma, alpha)

plt.figure(figsize=(10, 7.5))
plt.title("Hinge Loss $\\alpha$ = {:.2f}".format(alpha))

plt.subplot(2, 1, 1)
plt.plot(qs, training_error_Hinge, ".-", label="Hinge")

# reg_params = [0.00001]
# for reg_param in reg_params:
#     m_true, q_true, sigma_true = fixed_point_finder_strict(
#         f_L2_reg,
#         f_hat_Hinge_no_noise_classif,
#         (m, q, sigma),
#         {"reg_param": reg_param},
#         {"alpha": alpha},
#     )
#     next_color = next(plt.gca()._get_lines.prop_cycler)["color"]
#     plt.axvline(x=q_true, color=next_color, linestyle="--", label="q $\\lambda$ = {:.3f}".format(reg_param))

plt.plot(qs, stab_values, ".-", label="Stability")
plt.ylabel("Training Error")
plt.xlabel("q")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.subplot(2, 1, 2)
plt.plot(qs, angle_ts_Hinge, ".-", label="Hinge")
# plt.axvline(x=q_true, color="k", linestyle="--", label="q $\\lambda$ = {:.4f}".format(reg_param))
plt.ylabel("Generalization Error")
plt.xlabel("q")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
# plt.ylim([-0.5, 2])
plt.grid(which="both")


plt.show()
