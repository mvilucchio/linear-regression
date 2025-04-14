import matplotlib.pyplot as plt
import os
import numpy as np

alpha_min, alpha_max, n_alphas = 0.1, 5.0, 20
gammas = [0.5, 1.0, 2.0]
d = 500
delta = 0.1
reps = 20
eps_t = 0.1
pstar_t = 1.0
reg_param = 1e-2
pstar = 1.0

alphas = np.linspace(alpha_min, alpha_max, n_alphas)

plt.figure(figsize=(15, 6))

for k, gamma in enumerate(gammas):
    data_folder = f"./data/hastie_model_training"
    file_name = f"ERM_training_gamma_{{gamma:.2f}}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv".format(
        gamma=gamma
    )
    file_path = os.path.join(data_folder, file_name)

    data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    alphas = data[:, 0]
    gen_errors_mean, gen_errors_std = data[:, 7], data[:, 8]
    adv_errors_mean, adv_errors_std = data[:, 9], data[:, 10]
    flipped_fairs_mean, flipped_fairs_std = data[:, 11], data[:, 12]
    misclas_fairs_mean, misclas_fairs_std = data[:, 13], data[:, 14]

    plt.subplot(1, 4, 1)
    plt.errorbar(
        alphas,
        gen_errors_mean,
        yerr=gen_errors_std,
        label=f"$\\gamma = {gamma:.1f}$",
    )

    plt.subplot(1, 4, 2)
    plt.errorbar(
        alphas,
        adv_errors_mean,
        yerr=adv_errors_std,
        label=f"$\\gamma = {gamma:.1f}$",
    )

    plt.subplot(1, 4, 3)
    plt.errorbar(
        alphas,
        flipped_fairs_mean,
        yerr=flipped_fairs_std,
        label=f"$\\gamma = {gamma:.1f}$",
    )

    plt.subplot(1, 4, 4)
    plt.errorbar(
        alphas,
        misclas_fairs_mean,
        yerr=misclas_fairs_std,
        label=f"$\\gamma = {gamma:.1f}$",
    )

plt.subplot(1, 4, 1)
plt.xlabel(r"$\alpha$")
plt.ylabel("Generalization Error")
plt.grid(which="both")
plt.ylim(0, 1)
plt.legend(loc="best", fancybox=True, shadow=True)

plt.subplot(1, 4, 2)
plt.xlabel(r"$\alpha$")
plt.ylabel("Adversarial Error")
plt.grid(which="both")
plt.ylim(0, 1)
plt.legend(loc="best", fancybox=True, shadow=True)

plt.subplot(1, 4, 3)
plt.xlabel(r"$\alpha$")
plt.ylabel("Flipped Fairness")
plt.grid(which="both")
plt.ylim(0, 1)
plt.legend(loc="best", fancybox=True, shadow=True)

plt.subplot(1, 4, 4)
plt.xlabel(r"$\alpha$")
plt.ylabel("Misclassification Fairness")
plt.grid(which="both")
plt.ylim(0, 1)
plt.legend(loc="best", fancybox=True, shadow=True)

plt.suptitle(f"Linear Random Features q = inf $\\lambda$ = {reg_param:.1e}")
plt.tight_layout()

plt.show()
