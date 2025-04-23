from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import (
    f_projection_denoising,
)
import pickle
from linear_regression.erm.numerics import (
    erm_weight_finding,
)
from linear_regression.data.generation import data_generation
from scipy.signal import find_peaks
from linear_regression.aux_functions.loss_functions import huber_loss
from linear_regression.aux_functions.stability_functions import (
    stability_Huber_decorrelated_regress,
)
from linear_regression.data.generation import measure_gen_decorrelated
from linear_regression.fixed_point_equations.regression.Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.sweeps.q_sweeps import sweep_q_fixed_point_proj_denoiser
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.training_errors import training_error_huber_loss
from linear_regression.erm.erm_solvers import (
    find_coefficients_vanilla_GD,
    _loss_and_gradient_Huber,
)
from linear_regression.data.generation import measure_gen_decorrelated
from linear_regression.erm.metrics import train_error_data, q_real_overlaps
import numpy as np
import pickle
import matplotlib.pyplot as plt

delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 1.0
alpha = 30.0
reg_param = -2.5
max_iters_GD = 1000

n_features = 1_000
q_min_num, q_max_num = 0.5, 100
n_num = 10

qs_list = np.logspace(np.log10(q_min_num), np.log10(q_max_num), n_num)

if __name__ == "__main__":
    xs, ys, _, _, theta_0_teacher = data_generation(
        measure_gen_decorrelated,
        n_features=n_features,
        n_samples=max(int(np.around(n_features * alpha)), 1),
        n_generalization=1,
        measure_fun_args=(delta_in, delta_out, percentage, beta),
    )

    for q in qs_list:
        w_init = np.random.normal(0, 1, n_features)

        w_init = w_init / np.linalg.norm(w_init) * np.sqrt(q * n_features)

        print(f"Initial norm of w: {np.linalg.norm(w_init)} q: {q}")

        w_found, loss_vals, q_vals, estimation_error_vals = (
            find_coefficients_vanilla_GD(
                ys,
                xs,
                reg_param,
                _loss_and_gradient_Huber,
                (a,),
                1e-3,
                w_init=w_init,
                max_iters=max_iters_GD,
                save_run=True,
                ground_truth_theta=theta_0_teacher,
            )
        )

        plt.subplot(2, 1, 1)
        plt.plot(loss_vals, label=f"q = {q:.3f}")

        plt.subplot(2, 1, 2)
        plt.plot(q_vals, label=f"q = {q:.3f}")

        results = {
            'w_found': w_found,
            'loss_vals': loss_vals,
            'q_vals': q_vals,
            'estimation_error_vals': estimation_error_vals,
            'w_star': theta_0_teacher
        }

        pickle.dump(results, open(
            "./data/Huber_vanilla_GD_start_qinit_{:.3f}_alpha_{:.3f}_reg_{:.3f}_nfeatures_{:d}_deltain_{:.3f}_deltaout_{:.3f}_percentage_{:.3f}_beta_{:.3f}_a_{:.3f}.pkl".format(
                q, alpha, reg_param, n_features, delta_in, delta_out, percentage, beta, a
            ),
            "wb"
        ))

    plt.subplot(2, 1, 1)
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("qs")
    plt.legend()
    plt.grid()

    plt.show()