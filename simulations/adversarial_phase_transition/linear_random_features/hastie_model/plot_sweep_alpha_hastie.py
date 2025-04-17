import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

alpha_min, alpha_max, n_alphas_ERM, n_alphas_SE = 0.1, 5.0, 20, 60
gammas = [0.5, 1.0, 2.0]
d = 1000
delta = 0.0
reps = 20
eps_t = 0.1
pstar_t = 1.0
reg_param = 1e-3
pstar = 1.0

# Define colors for each gamma value
colors = [f"C{k}" for k in range(len(gammas))]

plt.figure(figsize=(15, 6))

for k, gamma in enumerate(gammas):
    data_folder = f"./data/hastie_model_training"
    file_name_ERM = f"ERM_training_gamma_{{gamma:.2f}}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_ERM:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv".format(
        gamma=gamma
    )
    file_path = os.path.join(data_folder, file_name_ERM)

    data = np.loadtxt(file_path, delimiter=",", skiprows=1)

    alphas = data[:, 0]
    m_mean, m_std = data[:, 1], data[:, 2]
    q_mean, q_std = data[:, 3], data[:, 4]
    P_mean, P_std = data[:, 5], data[:, 6]
    gen_errors_mean, gen_errors_std = data[:, 7], data[:, 8]
    adv_errors_mean, adv_errors_std = data[:, 9], data[:, 10]
    flipped_fairs_mean, flipped_fairs_std = data[:, 11], data[:, 12]
    misclas_fairs_mean, misclas_fairs_std = data[:, 13], data[:, 14]

    plt.subplot(2, 4, 1)
    plt.errorbar(
        alphas,
        gen_errors_mean,
        yerr=gen_errors_std,
        label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 2)
    plt.errorbar(
        alphas,
        adv_errors_mean,
        yerr=adv_errors_std,
        label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 3)
    plt.errorbar(
        alphas,
        flipped_fairs_mean,
        yerr=flipped_fairs_std,
        label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 4)
    plt.errorbar(
        alphas,
        misclas_fairs_mean,
        yerr=misclas_fairs_std,
        label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 5)
    plt.errorbar(
        alphas,
        m_mean,
        yerr=m_std,
        label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 6)
    plt.errorbar(
        alphas,
        q_mean,
        yerr=q_std,
        label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    plt.subplot(2, 4, 7)
    plt.errorbar(
        alphas,
        P_mean,
        yerr=P_std,
        label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        color=colors[k],
        fmt=".",
    )

    # Load SE data if available
    file_name_SE = f"SE_training_gamma_{gamma:.2f}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas_SE:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"
    file_path = os.path.join(data_folder, file_name_SE)

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        alphas_se = data[:, 0]
        m = data[:, 1]
        q = data[:, 2]
        P = data[:, 4]
        gen_err = data[:, 7]
        adv_err = data[:, 6]
        flipped_fair = data[:, 8]
        misclas_fair = data[:, 9]

        plt.subplot(2, 4, 1)
        plt.plot(
            alphas_se,
            gen_err,
            label=f"$\\gamma = {gamma:.1f}$ (SE)",
            linestyle="--",
            color=colors[k],
        )

        plt.subplot(2, 4, 2)
        plt.plot(
            alphas_se,
            adv_err,
            label=f"$\\gamma = {gamma:.1f}$ (SE)",
            linestyle="--",
            color=colors[k],
        )

        plt.subplot(2, 4, 3)
        plt.plot(
            alphas_se,
            flipped_fair,
            label=f"$\\gamma = {gamma:.1f}$ (SE)",
            linestyle="--",
            color=colors[k],
        )

        plt.subplot(2, 4, 4)
        plt.plot(
            alphas_se,
            misclas_fair,
            label=f"$\\gamma = {gamma:.1f}$ (SE)",
            linestyle="--",
            color=colors[k],
        )

        plt.subplot(2, 4, 5)
        plt.plot(
            alphas_se,
            m,
            label=f"$\\gamma = {gamma:.1f}$ (SE)",
            linestyle="--",
            color=colors[k],
        )

        plt.subplot(2, 4, 6)
        plt.plot(
            alphas_se,
            q,
            label=f"$\\gamma = {gamma:.1f}$ (SE)",
            linestyle="--",
            color=colors[k],
        )

        plt.subplot(2, 4, 7)
        plt.plot(
            alphas_se,
            P,
            label=f"$\\gamma = {gamma:.1f}$ (SE)",
            linestyle="--",
            color=colors[k],
        )
    except (FileNotFoundError, IOError):
        print(f"SE data file not found: {file_path}")

# Create custom legend with only one entry per gamma value
plt.subplot(2, 4, 1)
plt.xlabel(r"$\alpha$")
plt.ylabel("Generalization Error")
plt.grid(which="both")
plt.ylim(0, 1)

# Create a custom legend
custom_lines = []
for k, gamma in enumerate(gammas):
    custom_lines.append(
        Line2D(
            [0],
            [0],
            color=colors[k],
            marker="o",
            linestyle="",
            markersize=5,
            label=f"$\\gamma = {gamma:.1f}$ (ERM)",
        )
    )
    custom_lines.append(
        Line2D([0], [0], color=colors[k], linestyle="--", label=f"$\\gamma = {gamma:.1f}$ (SE)")
    )
plt.legend(handles=custom_lines, loc="best", fancybox=True, shadow=True)

plt.subplot(2, 4, 2)
plt.xlabel(r"$\alpha$")
plt.ylabel("Adversarial Error")
plt.grid(which="both")
plt.ylim(0, 1)

plt.subplot(2, 4, 3)
plt.xlabel(r"$\alpha$")
plt.ylabel("Flipped Fairness")
plt.grid(which="both")
plt.ylim(0, 1)

plt.subplot(2, 4, 4)
plt.xlabel(r"$\alpha$")
plt.ylabel("Misclassification Fairness")
plt.grid(which="both")
plt.ylim(0, 1)

plt.subplot(2, 4, 5)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$m$")
plt.grid(which="both")

plt.subplot(2, 4, 6)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$q$")
plt.grid(which="both")

plt.subplot(2, 4, 7)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$P$")
plt.grid(which="both")

plt.suptitle(f"Linear Random Features q = inf $\\lambda$ = {reg_param:.1e}")
plt.tight_layout()

plt.show()
