import matplotlib.pyplot as plt
import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from linear_regression.erm.metrics import percentage_flipped_labels
from linear_regression.erm.erm_solvers import find_coefficients_Logistic
from linear_regression.data.adversarial_generation import (
    adversarial_direction_generation,
)
from linear_regression.erm.loss_functions import logistic_loss, Dx_logistic_loss
from numba import njit


@njit
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit
def D_sigmoid(x):
    return np.exp(x) / (1 + np.exp(x)) ** 2


if __name__ == "__main__":
    alpha = 5.0
    gamma = 1.0
    reg_param = 1e-3
    ord_ = 2

    for n_features in [256, 512, 1024]:
        print(f"Generating data n_featueres = {n_features:d}")

        vs, xs, ys, _, _, _, teacher_vector, projector = data_generation(
            measure_gen_no_noise_clasif,
            n_features=n_features,
            n_samples=max(int(n_features * alpha), 1),
            n_generalization=1,
            measure_fun_args={},
            hidden_model=True,
            hidden_fun=sigmoid,
            overparam_ratio=gamma,
        )

        estimated_theta = find_coefficients_Logistic(ys, vs, reg_param)

        epss = np.logspace(-2, 4, 30)
        percentage_of_labels_flipped = np.empty_like(epss)

        scaling = (ord_ - 1.0) / ord_ if ord_ < np.inf else 1.0
        epss_rescaled = epss / n_features**scaling

        for i, eps_i in enumerate(epss_rescaled):

            _, _, direction_advs = adversarial_direction_generation(
                xs,
                Dx_logistic_loss,
                (ys, projector, estimated_theta, sigmoid, D_sigmoid),
                teacher_vector,
                orthogonal_projetion=True,
                ord=ord_,
                hidden_model=True,
                ratio_hidden=gamma,
                hidden_fun=sigmoid,
                proj_mat_hidden=projector,
            )

            adv_perturbation = (
                eps_i / np.linalg.norm(direction_advs, ord=ord_, axis=1)
            )[:, None] * direction_advs

            xs_perturbed = xs + adv_perturbation

            vs_perturbed = sigmoid(
                (xs_perturbed @ projector.T / np.sqrt(n_features))
            ) / np.sqrt(n_features * gamma)

            flipped = percentage_flipped_labels(
                ys, vs, estimated_theta, teacher_vector, vs_perturbed
            )

            percentage_of_labels_flipped[i] = flipped

        plt.plot(
            epss, percentage_of_labels_flipped, label=f"n_features = {n_features:d}"
        )

    plt.title(f"$\\alpha$ = {alpha:.1f}")
    plt.xscale("log")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Percentage of flipped labels")
    plt.grid()
    plt.legend()
    plt.show()
