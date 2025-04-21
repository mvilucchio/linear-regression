import matplotlib.pyplot as plt
import os
import numpy as np

gamma_min, gamma_max, n_gammas_ERM, n_gammas_SE = 0.5, 2.0, 15, 60
alphas = [0.5, 1.0, 2.0]
d = 300
delta = 0.0
reps = 20
eps_t = 0.1
pstar_t = 1.0
reg_param = 1e-3
pstar = 1.0

# Define colors for each alpha value
colors = [f"C{k}" for k in range(len(alphas))]

plt.figure(figsize=(15, 6))

for k, alpha in enumerate(alphas):
    data_folder = f"./data/hastie_model_training"
    file_name_ERM = f"ERM_training_alpha_{{alpha:.2f}}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas_ERM:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv".format(
        alpha=alpha
    )
    file_path = os.path.join(data_folder, file_name_ERM)

    data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    gammas = data[:, 0]
    m_mean, m_std = data[:, 1], data[:, 2]
    q_mean, q_std = data[:, 3], data[:, 4]
    P_mean, P_std = data[:, 5], data[:, 6]
    gen_errors_mean, gen_errors_std = data[:, 7], data[:, 8]
    adv_errors_mean, adv_errors_std = data[:, 9], data[:, 10]
    flipped_fairs_mean, flipped_fairs_std = data[:, 11], data[:, 12]
    misclas_fairs_mean, misclas_fairs_std = data[:, 13], data[:, 14]

    plt.subplot(2, 4, 1)
    plt.errorbar(
        gammas,
        gen_errors_mean,
        yerr=gen_errors_std,
        label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 2)
    plt.errorbar(
        gammas,
        adv_errors_mean,
        yerr=adv_errors_std,
        label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 3)
    plt.errorbar(
        gammas,
        flipped_fairs_mean,
        yerr=flipped_fairs_std,
        label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 4)
    plt.errorbar(
        gammas,
        misclas_fairs_mean,
        yerr=misclas_fairs_std,
        label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 5)
    plt.errorbar(
        gammas,
        m_mean,
        yerr=m_std,
        label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 6)
    plt.errorbar(
        gammas,
        q_mean,
        yerr=q_std,
        label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 7)
    plt.errorbar(
        gammas,
        P_mean,
        yerr=P_std,
        label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    file_name_SE = f"SE_training_alpha_{alpha:.2f}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas_SE:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"
    file_path = os.path.join(data_folder, file_name_SE)

    data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    gammas = data[:, 0]
    m = data[:, 1]
    q = data[:, 2]
    P = data[:, 4]
    gen_err = data[:, 7]
    adv_err = data[:, 6]
    flipped_fair = data[:, 8]
    misclas_fair = data[:, 9]

    plt.subplot(2, 4, 1)
    plt.plot(
        gammas,
        gen_err,
        label=f"$\\alpha = {alpha:.1f}$ (SE)",
        linestyle="--",
        color=colors[k],
    )

    plt.subplot(2, 4, 2)
    plt.plot(
        gammas,
        adv_err,
        label=f"$\\alpha = {alpha:.1f}$ (SE)",
        linestyle="--",
        color=colors[k],
    )

    plt.subplot(2, 4, 3)
    plt.plot(
        gammas,
        flipped_fair,
        label=f"$\\alpha = {alpha:.1f}$ (SE)",
        linestyle="--",
        color=colors[k],
    )

    plt.subplot(2, 4, 4)
    plt.plot(
        gammas,
        misclas_fair,
        label=f"$\\alpha = {alpha:.1f}$ (SE)",
        linestyle="--",
        color=colors[k],
    )

    plt.subplot(2, 4, 5)
    plt.plot(
        gammas,
        m,
        label=f"$\\alpha = {alpha:.1f}$ (SE)",
        linestyle="--",
        color=colors[k],
    )

    plt.subplot(2, 4, 6)
    plt.plot(
        gammas,
        q,
        label=f"$\\alpha = {alpha:.1f}$ (SE)",
        linestyle="--",
        color=colors[k],
    )

    plt.subplot(2, 4, 7)
    plt.plot(
        gammas,
        P,
        label=f"$\\alpha = {alpha:.1f}$ (SE)",
        linestyle="--",
        color=colors[k],
    )

# Create custom legend with only one entry per alpha value
plt.subplot(2, 4, 1)
plt.xlabel(r"$\gamma$")
plt.ylabel("Generalization Error")
plt.grid(which="both")
plt.ylim(0, 1)

# Create a custom legend
from matplotlib.lines import Line2D

custom_lines = []
for k, alpha in enumerate(alphas):
    custom_lines.append(
        Line2D(
            [0],
            [0],
            color=colors[k],
            marker="o",
            linestyle="",
            markersize=5,
            label=f"$\\alpha = {alpha:.1f}$ (ERM)",
        )
    )
    custom_lines.append(
        Line2D([0], [0], color=colors[k], linestyle="--", label=f"$\\alpha = {alpha:.1f}$ (SE)")
    )
plt.legend(handles=custom_lines, loc="best", fancybox=True, shadow=True)

plt.subplot(2, 4, 2)
plt.xlabel(r"$\gamma$")
plt.ylabel("Adversarial Error")
plt.grid(which="both")
plt.ylim(0, 1)

plt.subplot(2, 4, 3)
plt.xlabel(r"$\gamma$")
plt.ylabel("Flipped Fairness")
plt.grid(which="both")
plt.ylim(0, 1)

plt.subplot(2, 4, 4)
plt.xlabel(r"$\gamma$")
plt.ylabel("Misclassification Fairness")
plt.grid(which="both")
plt.ylim(0, 1)
plt.legend(loc="best", fancybox=True, shadow=True)

plt.subplot(2, 4, 5)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$m$")
plt.grid(which="both")
plt.legend(loc="best", fancybox=True, shadow=True)

plt.subplot(2, 4, 6)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$q$")
plt.grid(which="both")
plt.legend(loc="best", fancybox=True, shadow=True)

plt.subplot(2, 4, 7)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$P$")
plt.grid(which="both")
plt.legend(loc="best", fancybox=True, shadow=True)

plt.suptitle(f"Linear Random Features q = inf $\\lambda$ = {reg_param:.1e}")
plt.tight_layout()

plt.show()
