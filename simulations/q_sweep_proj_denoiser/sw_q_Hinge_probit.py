from scipy.signal import find_peaks
from linear_regression.fixed_point_equations.classification.Hinge_loss import f_hat_Hinge_probit_classif
from linear_regression.aux_functions.stability_functions import stability_Hinge_probit_classif
from linear_regression.sweeps.q_sweeps import sweep_q_fixed_point_proj_denoiser
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.training_errors import training_error_Hinge_loss_probit
import numpy as np
import matplotlib.pyplot as plt


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


q_min, q_max, n_q_pts = 0.01, 10, 200
alpha = 2.0

m, q, sigma = 0.3, 10.0, 0.9

deltas_noise = [0.001, 0.01, 0.1]

plt.figure(figsize=(10, 7.5))
plt.subplot(2, 1, 1)
plt.title("Hinge Loss $\\alpha$ = {:.2f}".format(alpha))

for delta_noise in deltas_noise:
    plt.subplot(2, 1, 1)
    qs, (training_error_Hinge, angle_ts_Hinge, sigmas_Hinge, ms_Hinge) = sweep_q_fixed_point_proj_denoiser(
        f_hat_Hinge_probit_classif,
        q_min,
        q_max,
        n_q_pts,
        {"q_fixed": q_min},
        {"alpha": alpha, "delta": delta_noise},
        initial_cond_fpe=(m, q, sigma),
        funs=[training_error_Hinge_loss_probit, angle_teacher_student, sigma_order_param, m_order_param],
        funs_args=[{"delta": delta_noise}, {}, {}, {}],
        update_funs_args=[False, False, False, False],
    )

    next_color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.plot(qs, training_error_Hinge, "-", color=next_color, label="$\\Delta$ = {:.4f}".format(delta_noise))

    # stab_values = np.empty_like(qs)
    # for idx, (m, q, sigma) in enumerate(zip(ms_Hinge, qs, sigmas_Hinge)):
    #     stab_values[idx] = stability_Hinge_probit_classif(q, m, sigma, alpha, delta_noise)

    # plt.plot(qs, stab_values, "--", color=next_color, label="Stability")

    plt.subplot(2, 1, 2)
    plt.plot(qs, angle_ts_Hinge, "-", color=next_color, label="$\\Delta$ = {:.4f}".format(delta_noise))


plt.subplot(2, 1, 1)
plt.ylabel("Training Error")
plt.xlabel("q")
plt.xscale("log")
plt.legend()
plt.grid(which="both")

plt.subplot(2, 1, 2)
# plt.axvline(x=q_true, color="k", linestyle="--", label="q $\\lambda$ = {:.4f}".format(reg_param))
plt.ylabel("Generalisation Error")
plt.xlabel("q")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
# plt.ylim([-0.5, 2])
plt.grid(which="both")

plt.show()
