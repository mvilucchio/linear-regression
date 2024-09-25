import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "./data/SE_any_norm_cluster"
imgs_dir = "./imgs/SE_alpha_sweep_cluster"

files = os.listdir(data_dir)

reg_order = [1, 2, 3]
pstar = 2
file_names = [
    f"SE_data_pstar_{pstar}_reg_order_{r}_pstar_{pstar}_alpha_0.005_100.000_reg_param_1.0e-01_eps_t_g_3.0e-01_3.0e-01.csv"
    for r in reg_order
]

for file in file_names:
    with open(os.path.join(data_dir, file), "rb") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1)

    alphas = data[:, 0]

    ms_found = data[::-1, 1]
    qs_found = data[::-1, 2]
    Vs_found = data[::-1, 3]
    Ps_found = data[::-1, 4]

    mhats_found = data[::-1, 5]
    qhats_found = data[::-1, 6]
    Vhats_found = data[::-1, 7]
    Phats_found = data[::-1, 8]

    estim_errors_se = data[::-1, 9]
    adversarial_errors_found = data[::-1, 10]
    gen_errors_se = data[::-1, 11]

    fig, axs = plt.subplots(2, 1, figsize=(5, 7))

    axs[0].plot(alphas, adversarial_errors_found, label="E_{{adv}}")
    axs[0].plot(alphas, gen_errors_se, label="E_{{gen}}")
    axs[0].set_xlabel("$\\alpha$")
    axs[0].set_ylabel("Error")
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_ylim(1e-1, 6e-1)
    axs[0].set_xlim(1e-2, 1)
    axs[0].grid()
    axs[0].set_title(f"Reg order: {reg_order[file_names.index(file)]} - pstar: {pstar}")

    axs[1].plot(alphas, ms_found, label="m")
    axs[1].plot(alphas, qs_found, label="q")
    axs[1].plot(alphas, Vs_found, label="V")
    axs[1].plot(alphas, Ps_found, label="P")
    axs[1].plot(alphas, ms_found / np.sqrt(qs_found), label="m/sqrt(q)")
    axs[1].set_xlabel("$\\alpha$")
    axs[1].set_ylabel("Overlap")
    axs[1].legend()
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].grid()

    plt.show()
    # plt.savefig(f"./imgs/SE_alpha_sweep_cluster/{file.split('.')[0]}.png")
    # plt.close()
