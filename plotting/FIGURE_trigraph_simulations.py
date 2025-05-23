import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import glob


IMG_DIRECTORY = "./imgs"


def save_plot(fig, name, formats=["pdf"], date=True):
    for f in formats:
        fig.savefig(
            os.path.join(IMG_DIRECTORY, "{}".format(name) + "." + f),
            format=f,
        )


def set_size(width, fraction=1, subplots=(1, 1)):
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * (golden_ratio) * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


save = True
width = 433.62

plt.style.use("./plotting/latex_ready.mplstyle")

tuple_size = set_size(width, fraction=0.50)


fig, axs = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(tuple_size[0], 0.75 * tuple_size[0]),
    gridspec_kw={"hspace": 0, "wspace": 0},
)
fig.subplots_adjust(left=0.14)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.98)
fig.subplots_adjust(right=0.96)

fig.delaxes(axs[0, 0])

data_path = "./data/huber_1.000_reg_param_-2.500_alpha_30.000_delta_in_1.000_delta_out_5.000_percentage_0.300_beta_0.000.pkl"
with open(data_path, "rb") as file:
    data = pickle.load(file)


axs[1, 0].plot(data["training_errors"], data["qs"], "r-")
axs[1, 0].plot(data["training_error_true"], data["q_true"], "xr")
axs[1, 0].invert_xaxis()
axs[1, 0].set_xlim([2.5, -0.4])
# axs[1, 0].set_title("Bottom Left Plot")

file_list = glob.glob("./data/*reg_-2.500_*")
q_init_values = []
for file_path in file_list:
    q_init_value = float(file_path.split("qinit_")[1].split("_")[0])
    print(q_init_value)
    q_init_values.append(float(q_init_value))

colorlist = [f"C{idx}" for idx in range(len(file_list))]

for file_path, q0, c in zip(file_list, q_init_values, colorlist):
    with open(file_path, "rb") as file:
        data_1 = pickle.load(file)

    axs[0, 1].plot(data_1["estimation_error_vals"], "-", label=q0, color=c)
    # axs[0, 1].set_title("Top Right Plot")

    axs[1, 1].plot(data_1["q_vals"], "-", label=q0, color=c)
    # axs[1, 1].set_title("Bottom Right Plot")

    closest_idx = np.argmin(np.abs(data["qs"] - data_1["q_vals"][0]))

    axs[1, 0].plot(data["training_errors"][closest_idx], data["qs"][closest_idx], ".", color=c)

    tmp = np.linspace(data["training_errors"][closest_idx], -3, 3)
    tmp2 = data["qs"][closest_idx] * np.ones_like(tmp)
    axs[1, 0].plot(tmp, tmp2, "--", color=c)


for ax in axs[:, 1]:
    ax.sharex(axs[0, 1])

for ax in axs[1, :]:
    ax.sharey(axs[1, 0])

axs[1, 1].set_xlabel("GD Iters.", labelpad=0.0)
axs[1, 1].set_xlim([0, 1000])

# axs[1, 0].set_xlabel(r"$E_{\mathrm{train}}$", labelpad=0.0)
axs[1, 0].set_xlabel(r"$\tilde{\mathcal{A}}(q)$", labelpad=0.0)
axs[1, 0].set_ylabel(r"$q$")
axs[1, 0].set_ylim([0.2, 350])

# axs[0, 1].set_ylabel(r"$E_{\mathrm{estim}}$")
axs[0, 1].set_ylabel(r"$\frac{1}{d} \|\hat{\mathbf{w}} - \mathbf{w}_\star\|_2^2$")
axs[0, 1].set_yscale("log")
# axs[0,1].set_ylim([-10, 150])

legend_ax = fig.add_subplot(2, 2, 1)
legend_ax.axis("off")

axs[0, 1].xaxis.set_tick_params(which="both", bottom=False, labelbottom=False)
axs[1, 1].yaxis.set_tick_params(which="both", left=False, labelleft=False)

axs[1, 1].yaxis.grid(True)


axs[1, 0].set_yscale("log")

handles, labels = [], []
for ax in [
    axs[1, 1],
]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

# legend_ax.legend(handles, labels, loc="center", title="Legend")
# plt.tight_layout()

if save:
    save_plot(
        fig,
        "triplot_huber_loss_GD",
    )

plt.show()
