import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "./data/SE_alpha_sweep"
imgs_dir = "./imgs/SE_alpha_sweep"

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

files = os.listdir(data_dir)

# reg_order = [1, 2, 3]
# pstar = 2
# file_names = [
#     f"SE_data_pstar_{pstar}_reg_order_{r}_pstar_{pstar}_alpha_0.005_100.000_reg_param_1.0e-01_eps_t_g_3.0e-01_3.0e-01.csv"
#     for r in reg_order
# ]

for file in files:
    with open(os.path.join(data_dir, file), "rb") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1)

    # remove the .csv extension
    file = file[:-4]
    file = file.split("_")
    print(file)

    pstar = int(file[4])
    reg_order = float(file[7])
    reg_param = float(file[13])
    eps_t = float(file[15])

    alphas = data[:, 0]

    ms_found = data[:, 1]
    qs_found = data[:, 2]
    Vs_found = data[:, 3]
    Ps_found = data[:, 4]

    mhats_found = data[:, 5]
    qhats_found = data[:, 6]
    Vhats_found = data[:, 7]
    Phats_found = data[:, 8]

    estim_errors_se = data[:, 9]
    adversarial_errors_found = data[:, 10]
    gen_errors_se = data[:, 11]

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
    axs[0].set_title(
        f"r: {reg_order} $p^\\star$: {pstar} reg_param: {reg_param} $\\varepsilon$: {eps_t}"
    )

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

    plt.tight_layout()

    plt.savefig(os.path.join(imgs_dir, f"{file}.png"), format="png", dpi=300)
    plt.close()
