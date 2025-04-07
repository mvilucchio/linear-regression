import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from os.path import join, exists

IMGS_FOLDER = "./imgs"
if not exists(IMGS_FOLDER):
    os.makedirs(IMGS_FOLDER)

# Parameters matching those used in the MNIST ERM experiment
alpha_min, alpha_max, n_alpha_pts = 0.06, 1.0, 12
reg_orders = [1, 2, 3]
eps_t = 1.0
eps_g = 1.0
DIGIT_1, DIGIT_2 = 0, 1  # Update these to match your experiment

data_folder = "./data"
file_name = f"MNIST_ERM_data_Linf_reg_order_{{:d}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_{n_alpha_pts:d}_digits_{DIGIT_1}vs{DIGIT_2}_reps_{{:d}}_reg_param_{{:.1e}}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl"

# Set up the plot
plt.style.use("./plotting/latex_ready.mplstyle")  # Make sure this file exists

# Calculate dimensions for the figure
columnwidth = 234.8775
fig_width_pt = columnwidth
inches_per_pt = 1.0 / 72.27
figwidth = fig_width_pt * inches_per_pt
figheight = figwidth * (5.0**0.5 - 1.0) / 2.0

# Create figure with a single subplot
fig, ax = plt.subplots(figsize=(figwidth, figheight), layout="constrained")

# Number of repetitions in your experiment
reps = 10
# Regularization parameter to plot
reg_param = 1e-3  # Change this to the reg_param you want to visualize

# Load and plot data for each regularization order
for i, reg_order in enumerate(reg_orders):
    try:
        # Try to load the data
        with open(join(data_folder, file_name.format(reg_order, reps, reg_param)), "rb") as f:
            data_dict = pickle.load(f)

        # Extract data
        alphas = data_dict["alphas"]

        gen_error_mean = data_dict["gen_error_mean"]
        gen_error_std = data_dict["gen_error_std"]

        adv_error_mean = data_dict["adversarial_error_mean"]
        adv_error_std = data_dict["adversarial_error_std"]

        # Plot generalization error (dashed) and adversarial error (solid)
        # ax.plot(
        #     alphas,
        #     gen_error_mean,
        #     "--",
        #     color=f"C{i}",
        #     label=f"$r$ = {reg_order}, $E_{{\\mathrm{{gen}}}}$",
        # )

        # ax.plot(
        #     alphas,
        #     adv_error_mean,
        #     "-",
        #     color=f"C{i}",
        #     label=f"$r$ = {reg_order}, $E_{{\\mathrm{{rob}}}}$",
        # )

        # Add error bars if needed
        ax.errorbar(
            alphas,
            gen_error_mean,
            yerr=gen_error_std / np.sqrt(reps),
            linestyle="--",
            marker=".",
            markersize=2,
            color=f"C{i}",
        )

        ax.errorbar(
            alphas,
            adv_error_mean,
            yerr=adv_error_std / np.sqrt(reps),
            linestyle="-",
            marker=".",
            markersize=2,
            color=f"C{i}",
        )

    except FileNotFoundError:
        print(f"File for reg_order={reg_order} not found, skipping.")
    except Exception as e:
        print(f"Error loading reg_order={reg_order}: {e}")

# Configure plot styling
# Set these limits based on your data ranges or keep them adaptive
# y_lim = [0.0, 0.5]  # y-axis limits
x_lim = [6e-2, 1]  # x-axis limits
# If you want auto limits, comment out the limit setting below

# Configure the plot
ax.set_xlabel(r"$\alpha = n / d$", labelpad=0)
ax.set_xscale("log")
# Uncomment these if you want fixed limits
# ax.set_ylim(y_lim)
ax.set_xlim(x_lim)
ax.grid(which="major", visible=True, axis="both")
# ax.legend(loc="best", fontsize=8)

# legend for line styles
ax.plot([], [], "--", color="black", label="$E_{{\\mathrm{{gen}}}}$")
ax.plot([], [], "-", color="black", label="$E_{{\\mathrm{{rob}}}}$")
ax.legend(loc="best", fontsize=8)

# legend for colors
ax.plot([], [], color="C0", label="$r = 1$")
ax.plot([], [], color="C1", label="$r = 2$")
ax.plot([], [], color="C2", label="$r = 3$")
ax.legend(loc="best", fontsize=8)

ax.set_title(
    f"MNIST {DIGIT_1}vs{DIGIT_2}, $\\varepsilon = {eps_t:.1f}$, $\\lambda = 10^{{{int(np.log10(reg_param)):d}}}$"
)

# Save the figure
plt.savefig(
    join(
        IMGS_FOLDER, f"MNIST_{DIGIT_1}vs{DIGIT_2}_errors_lambda_{reg_param:.0e}_eps_{eps_t:.1e}.pdf"
    ),
    format="pdf",
    dpi=300,
)

plt.show()
