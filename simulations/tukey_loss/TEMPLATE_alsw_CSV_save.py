import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from tqdm import tqdm

from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.regression.Tukey_loss import f_hat_Tukey_decorrelated_noise_TI, RS_decorrelated_noise_TI_l2_reg
from linear_regression.aux_functions.misc import excess_gen_error, estimation_error
from linear_regression.utils.errors import ConvergenceError
from linear_regression.fixed_point_equations import TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE, BLEND_FPE

# Loss specific hyperparameters
loss_fun_name = "Tukey"
loss_parameters = {"tau": 1.0}

# Regularization hyperparameter
reg_fun_name = "L2"
reg_param = 2.0  # Lambda

# Decorrelated noise parameters
noise ={"Delta_in": 0.1, "Delta_out": 1.0, "percentage": 0.1, "beta": 0.0}

# --- Alpha Sweep Parameters ---

alpha_min = 10
alpha_max = 100000
n_alpha_pts = 10000
decreasing_alpha = True

initial_cond_fpe = (0.35947940,0.12170537,3.04138236e-01)

# --- Plotting parameters ---

save_plot = True

# Choose what to plot (possible values: "m", "q", "V", "RS", "m_hat", "q_hat", "V_hat", "excess_gen_error", "estim_error", "time")
plotted_values = ["m", "q", 
                  "V", "RS",
                   "m_hat", "q_hat", "V_hat", 
                   "excess_gen_error", "estim_error", 
                   "time"]

if save_plot:

    plot_folder = f"./imgs/alpha_sweeps_{loss_fun_name}_{reg_fun_name}_decorrelated_noise"

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_subfolder = f"./imgs/alpha_sweeps_{loss_fun_name}_{reg_fun_name}_decorrelated_noise/loss_param_{loss_parameters['tau']:.1f}_reg_param_{reg_param:.1f}_noise_{noise['Delta_in']:.2f}_{noise['Delta_out']:.2f}_{noise['percentage']:.2f}_{noise['beta']:.2f}"

    if not os.path.exists(plot_subfolder):
        os.makedirs(plot_subfolder)

    figures_folder = f"./imgs/alpha_sweeps_{loss_fun_name}_{reg_fun_name}_decorrelated_noise/loss_param_{loss_parameters['tau']:.1f}_reg_param_{reg_param:.1f}_noise_{noise['Delta_in']:.2f}_{noise['Delta_out']:.2f}_{noise['percentage']:.2f}_{noise['beta']:.2f}/alsw_alpha_min_{alpha_min:.1f}_max_{alpha_max:.1f}_n_pts_{n_alpha_pts}"

    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

# --- Precision parameters ---

# Used for RS stability condition computation
integration_bound_RS = 5
integration_epsabs_RS = 1e-7
integration_epsrel_RS = 1e-4

# Fixed Point Finder precision parameters (optional)
abs_tol_FPE = TOL_FPE
min_iter_FPE = MIN_ITER_FPE
max_iter_FPE = MAX_ITER_FPE
blend_FPE = BLEND_FPE

# --- Save configuration ---

save_CSV = True
save_pickle = True

data_folder = f"./data/alpha_sweeps_{loss_fun_name}_{reg_fun_name}_decorrelated_noise"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_subfolder = f"./data/alpha_sweeps_{loss_fun_name}_{reg_fun_name}_decorrelated_noise/loss_param_{loss_parameters['tau']:.1f}_reg_param_{reg_param:.1f}_noise_{noise['Delta_in']:.2f}_{noise['Delta_out']:.2f}_{noise['percentage']:.2f}_{noise['beta']:.2f}"

if not os.path.exists(data_subfolder):
    os.makedirs(data_subfolder)

file_name_base = f"alsw_alpha_min_{alpha_min:.1f}_max_{alpha_max:.1f}_n_pts_{n_alpha_pts}"
file_path_CSV = os.path.join(data_subfolder, file_name_base + ".csv")
file_path_pkl = os.path.join(data_subfolder, file_name_base + ".pkl")

# --- Initialization ---
alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

if decreasing_alpha:
    alphas = alphas[::-1]  # Reverse order for decreasing alpha sweep

# Outputs
ms_results = np.full(n_alpha_pts, np.nan)
qs_results = np.full(n_alpha_pts, np.nan)
Vs_results = np.full(n_alpha_pts, np.nan)
m_hat_results = np.full(n_alpha_pts, np.nan)
q_hat_results = np.full(n_alpha_pts, np.nan)
V_hat_results = np.full(n_alpha_pts, np.nan)
excess_gen_error_results = np.full(n_alpha_pts, np.nan)
estim_error_results = np.full(n_alpha_pts, np.nan)
rs_values_results = np.full(n_alpha_pts, np.nan)
time_results = np.full(n_alpha_pts, np.nan)

# CSV Header
header_CSV = "alpha,m,q,V,m_hat,q_hat,V_hat,gen_error,estim_error,rs_value,time_sec\n"

with open(file_path_CSV, "w") as f:
    f.write(header_CSV)

# --- Alpha sweep with incremental save ---

print(f"Starting alpha sweep for {loss_fun_name} loss with parameters: {loss_parameters}")
print(f"Using {reg_fun_name} regularization with parameter: {reg_param}")
print(f"Alpha range: alpha_min={alpha_min}, alpha_max={alpha_max}, n_alpha_pts={n_alpha_pts}")
print(f"Noise model parameters: {noise}")

current_initial_cond = tuple(initial_cond_fpe)
f_kwargs = {"reg_param": reg_param}

for idx, alpha in enumerate(tqdm(alphas, desc="Alpha Sweep")):

    f_hat_kwargs = {
        "alpha": alpha,
        **loss_parameters,
        **noise
    }

    m, q, V, m_hat, q_hat, V_hat, excess_gen_err, estim_err, rs_value = (np.nan,) * 9 # Initialize to NaN
    success = False

    try:
        point_start_time = time.time()
        # Fixed-point iteration
        m, q, V = fixed_point_finder(
            f_func=f_L2_reg,
            f_hat_func=f_hat_Tukey_decorrelated_noise_TI,
            initial_condition=current_initial_cond,
            f_kwargs=f_kwargs,
            f_hat_kwargs=f_hat_kwargs,
            abs_tol=abs_tol_FPE,
            min_iter=min_iter_FPE,
            max_iter=max_iter_FPE,
            # update_function=lambda x : 0,
            # args_update_function=(BLEND_FPE,),
            # error_function=None,
            verbose=False
        )
        point_end_time = time.time()
        point_duration = point_end_time - point_start_time

        if np.all(np.isfinite([m, q, V])):
            m_hat, q_hat, V_hat = f_hat_Tukey_decorrelated_noise_TI(m, q, V, **f_hat_kwargs)

            # Compute generalization error
            excess_gen_err = excess_gen_error(m, q, V, **noise)

            # Compute estimation error
            estim_err = estimation_error(m, q, V, **noise)

            # Compute RS stability condition
            try:
                rs_kwargs = {
                    "alpha": alpha,
                    **loss_parameters,
                    "reg_param":reg_param,
                    **noise,
                    "integration_bound": integration_bound_RS,
                    "integration_epsabs": integration_epsabs_RS,
                    "integration_epsrel": integration_epsrel_RS
                }
                rs_value = RS_decorrelated_noise_TI_l2_reg(m, q, V, **rs_kwargs)
            except Exception as e_rs:
                print(f"\nWarning: RS computation failed for alpha={alpha:.2e}: {e_rs}")

            # Success
            if np.all(np.isfinite([m, q, V, m_hat, q_hat, V_hat, excess_gen_err, estim_err, rs_value])):
                success = True
                current_initial_cond = (m, q, V)  # Warm start for next alpha

    except (ConvergenceError, ValueError, FloatingPointError, OverflowError) as e:
        print(f"\nWarning: FPE failed for alpha={alpha:.2e}. Error: {type(e).__name__}")
    except Exception as e:
        print(f"\nUnexpected error for alpha={alpha:.2e}: {e}")

    # --- Save incremental results ---
    if success:
        ms_results[idx] = m
        qs_results[idx] = q
        Vs_results[idx] = V
        m_hat_results[idx] = m_hat
        q_hat_results[idx] = q_hat
        V_hat_results[idx] = V_hat
        excess_gen_error_results[idx] = excess_gen_err
        estim_error_results[idx] = estim_err
        rs_values_results[idx] = rs_value
        time_results[idx] = point_duration

        try:
            with open(file_path_CSV, "a") as f:
                rs_str = f"{rs_value:.8e}" if np.isfinite(rs_value) else "0.0"
                f.write(f"{alpha:.8e},{m:.8e},{q:.8e},{V:.8e},{m_hat:.8e},{q_hat:.8e},{V_hat:.8e},{excess_gen_err:.8e},{estim_err:.8e} {rs_str},{point_duration:.4f}\n")
                f.flush()
        except IOError as e:
            print(f"Error writing to {file_path_CSV}: {e}")

# --- Final save with pickle ---

final_results_dict = {
    "description": f"Alpha sweep for {loss_fun_name} loss with {reg_fun_name} regularization",
    "loss_fun": loss_fun_name,
    "loss_parameters": loss_parameters,
    "reg_fun": reg_fun_name,
    "reg_param": reg_param,
    "noise_model": "decorrelated_noise",
    "noise_parameters": noise,
    "integration_bound": integration_bound_RS,
    "integration_epsabs": integration_epsabs_RS,
    "integration_epsrel": integration_epsrel_RS,
    "alpha_range": {"min": alpha_min, "max": alpha_max, "n_points": n_alpha_pts},
    "alphas": alphas,
    "ms": ms_results,
    "qs": qs_results,
    "Vs": Vs_results,
    "m_hats": m_hat_results,
    "q_hats": q_hat_results,
    "V_hats": V_hat_results,
    "excess_gen_error": excess_gen_error_results,
    "estim_error": estim_error_results,
    "rs_values": rs_values_results,
    "times_sec": time_results
}

print(f"Saving final results to {file_path_pkl}...")
try:
    with open(file_path_pkl, "wb") as f:
        pickle.dump(final_results_dict, f)
    print("Final save completed.")
except Exception as e:
    print(f"Error during final pickle save: {e}")

print("Alpha sweep computation finished.")

# --- Plotting results ---

if save_plot:
    print("Plotting results...")

    def safe_plot(x, y, label, **kwargs):
        if np.all(np.isnan(y)):
            print(f"Skipping plot for {label}: all values are NaN.")
            return
        plt.plot(x, y, label=label, **kwargs)

    # ---- PLOT: m and q ----
    if any(k in plotted_values for k in ["m", "q"]):
        plt.figure()
        if "m" in plotted_values:
            safe_plot(alphas, ms_results, label="m", linestyle='-')
        if "q" in plotted_values:
            safe_plot(alphas, qs_results, label="q", linestyle='--')
        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("Order parameters")
        plt.title("m and q vs alpha")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "m_q_vs_alpha.png"))
        plt.close()

    # ---- PLOT: V and RS ----
    if any(k in plotted_values for k in ["V", "RS"]):
        plt.figure()
        if "V" in plotted_values:
            safe_plot(alphas, Vs_results, label="V", linestyle='-')
        if "RS" in plotted_values:
            safe_plot(alphas, rs_values_results, label="RS", linestyle='--')
        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("V / RS")
        plt.title("V and RS vs alpha")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "V_RS_vs_alpha.png"))
        plt.close()

    # ---- PLOT: m_hat, q_hat, V_hat ----
    if any(k in plotted_values for k in ["m_hat", "q_hat", "V_hat"]):
        plt.figure()
        if "m_hat" in plotted_values:
            safe_plot(alphas, m_hat_results, label="m_hat", linestyle='-')
        if "q_hat" in plotted_values:
            safe_plot(alphas, q_hat_results, label="q_hat", linestyle='--')
        if "V_hat" in plotted_values:
            safe_plot(alphas, V_hat_results, label="V_hat", linestyle='-.')
        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("Estimates from f_hat")
        plt.title("m_hat, q_hat, V_hat vs alpha")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "mhat_qhat_Vhat_vs_alpha.png"))
        plt.close()

    # ---- PLOT: Excess gen error and estimation error ----
    if any(k in plotted_values for k in ["excess_gen_error", "estim_error"]):
        plt.figure()
        if "excess_gen_error" in plotted_values:
            safe_plot(alphas, excess_gen_error_results, label="Excess Gen Error", linestyle='-')
        if "estim_error" in plotted_values:
            safe_plot(alphas, estim_error_results, label="Estimation Error", linestyle='--')
        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("Error")
        plt.title("Generalization and Estimation Error vs alpha")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "gen_estim_error_vs_alpha.png"))
        plt.close()

    # ---- PLOT: Time ----
    if "time" in plotted_values:
        plt.figure()
        safe_plot(alphas[1:], time_results[1:], label="Time per point (s)", linestyle='-')
        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("Time (seconds)")
        plt.title("Computation time vs alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "time_vs_alpha.png"))
        plt.close()
