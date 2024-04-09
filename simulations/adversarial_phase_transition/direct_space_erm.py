import matplotlib.pyplot as plt
import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import percentage_flipped_labels
from linear_regression.erm.erm_solvers import find_coefficients_Logistic
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.classification.Logistic_loss import (
    f_hat_Logistic_no_noise_classif,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.percentage_flipped import percentage_flipped_direct_space

if __name__ == "__main__":
    alpha = 5
    reg_param = 1e-4
    ord_ = 2
    reps = 5
    
    print("The direction is given by the estimated label")

    for n_features in [1024, 2048]:
        epss = np.logspace(-2, 2, 20)
        scaling = (ord_ - 1.0) / ord_ if ord_ < np.inf else 1.0
        # this rescaling works in the Clare case
        epss_rescaled = epss  # / n_features**scaling

        print(
            f"Generating data ... n_featueres = {n_features:d} n_samples = {int(n_features * alpha):d}"
        )

        vals = np.empty((reps, len(epss)))
        estim_vals_m = np.empty((reps,))
        estim_vals_q = np.empty((reps,))
        estim_vals_rho = np.empty((reps,))

        for j in range(reps):
            xs, ys, _, _, teacher_vector = data_generation(
                measure_gen_no_noise_clasif,
                n_features=n_features,
                n_samples=max(int(n_features * alpha), 1),
                n_generalization=1,
                measure_fun_args={},
                hidden_model=False,
            )

            estimated_theta = find_coefficients_Logistic(ys, xs, reg_param)

            estim_vals_rho[j] = np.sum(teacher_vector **2) / n_features
            estim_vals_m[j] = np.sum(teacher_vector * estimated_theta) / n_features
            estim_vals_q[j] = np.sum(estimated_theta**2) / n_features

            yhat = np.repeat(
                np.sign(xs @ estimated_theta).reshape(-1, 1), n_features, axis=1
            )

            direction_adv = (
                estimated_theta
                - ((teacher_vector @ estimated_theta) / np.sum(teacher_vector**2))
                * teacher_vector
            )

            direction_adv_norm = direction_adv / np.linalg.norm(direction_adv, ord=ord_)

            adv_perturbation = -(yhat * direction_adv_norm[None, :])

            for i, eps_i in enumerate(epss_rescaled):
                flipped = percentage_flipped_labels(
                    ys,
                    xs,
                    estimated_theta,
                    teacher_vector,
                    xs + eps_i * adv_perturbation,
                )

                vals[j, i] = flipped

        plt.errorbar(
            epss,
            np.mean(vals, axis=0),
            yerr=np.std(vals, axis=0),
            label=f"n_features = {n_features:d}",
            marker=".",
        )

        mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
        mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
        mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)

        print(
            f"Estimated m = {mean_m:.3f} ± {std_m:.3f} q = {mean_q:.3f} ± {std_q:.3f} rho = {mean_rho:.3f} ± {std_rho:.3f}"
        )

    # theoretical computation of m and q
    m_t, q_t, _ = fixed_point_finder(
        f_L2_reg,
        f_hat_Logistic_no_noise_classif,
        [0.1, 1.0, 0.1],
        {"reg_param": reg_param},
        {"alpha": alpha},
    )

    eps_dense = np.logspace(-2, 2, 100)
    out = np.empty_like(eps_dense)

    for i, eps_i in enumerate(eps_dense):
        out[i] = percentage_flipped_direct_space(mean_m, mean_q, mean_rho, eps_i)

    plt.plot(eps_dense, out, label="Theoretical", linestyle="--", color="black")

    plt.title(f"Direct space $\\alpha$ = {alpha:.1f} $\\lambda$ = {reg_param:.1e}")
    plt.xscale("log")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Percentage of flipped labels")
    plt.grid()
    plt.legend()
    plt.show()
