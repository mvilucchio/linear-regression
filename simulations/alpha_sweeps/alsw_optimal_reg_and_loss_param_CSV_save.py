# Matéo begins
# This file contains the implementation an alpha sweep with optimal loss and regularization parameters with decorrelated noise model.
# Barriers can be added for the V and RS conditions.
# Plots and CSV files can be saved.
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from tqdm import tqdm

from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.regression.translation_invariant_losses.Tukey_loss_TI import q_int_Tukey_decorrelated_noise_TI_r, V_int_Tukey_decorrelated_noise_TI_r, RS_Tukey_decorrelated_noise_TI_l2_reg
from linear_regression.fixed_point_equations.regression.translation_invariant_losses.f_hat_mixture_of_Gaussian import f_hat_decorrelated_noise_TI
from linear_regression.aux_functions.misc import excess_gen_error, estimation_error, angle_teacher_student
from linear_regression.fixed_point_equations.optimality_finding import find_optimal_reg_and_loss_param_function
from linear_regression.utils.errors import ConvergenceError, MinimizationError
from linear_regression.fixed_point_equations import TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE, BLEND_FPE

# Loss specific hyperparameters
loss_fun_name = "Tukey"
loss_param_name = "tau"
min_loss_param = 1e-9 #tau_min
loss_parameters = {"q_int_loss_decorrelated_noise_x": q_int_Tukey_decorrelated_noise_TI_r,
                   "V_int_loss_decorrelated_noise_x" : V_int_Tukey_decorrelated_noise_TI_r,
                   "even_loss":True} # Additional parameters for the loss function, if any

# Regularization hyperparameter
reg_fun_name = "L2"
min_reg_param = 1e-9 #lambda_min

# Decorrelated noise parameters
noise ={"Delta_in": 0.1, "Delta_out": 1.0, "percentage": 0.1, "beta": 0.0}

# --- Alpha Sweep Parameters ---

alpha_min = 50
alpha_max = 10000
n_alpha_pts = 200
decreasing_alpha = True

initial_cond_fpe = (0.9,0.82,3.04138236e-05)
initial_guess_reg_param = 1000
initial_guess_loss_param = 1.0

f_min = excess_gen_error # Function to minimize
f_min_args = {**noise} # Arguments for f_min except m,q,V
barrier_V_threshold = 1.25 # Threshold for V. Can be None for no barrier
barrier_RS_threshold = 1.0 # Threshold for RS condition. Can be None for no barrier
barrier_penalty = 1e10 # Penalty for violating barriers

# --- Plotting parameters ---

save_plots = True

# Choose what to plot (possible values: "m", "q", "V", "RS", "m_hat", "q_hat", "V_hat", "opt_reg_param", "opt_loss_param", "excess_gen_error", "estim_error", "angle_teacher_student", "time")
plotted_values = ["m", "q", 
                  "V", "RS", "angle_teacher_student",
                   "m_hat", "q_hat", "V_hat", 
                   "opt_reg_param",
                   "opt_loss_param",
                   "excess_gen_error", "estim_error", 
                   "time"]

folder_name = f"alsw_{loss_fun_name}_{reg_fun_name}_decorrelated_noise_opt_reg_and_loss_param_{f_min.__name__}"
subfolder_name = f"noise_{noise['Delta_in']:.2f}_{noise['Delta_out']:.2f}_{noise['percentage']:.2f}_{noise['beta']:.2f}"
file_name_base = f"alpha_min_{alpha_min:.0e}_max_{alpha_max:.0e}_n_pts_{n_alpha_pts}_min_reg_and_loss_param_{min_reg_param:.0e}_{min_loss_param:.0e}"

if save_plots:

    plot_folder = f"./imgs/{folder_name}"

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_subfolder = f"{plot_folder}/{subfolder_name}"

    if not os.path.exists(plot_subfolder):
        os.makedirs(plot_subfolder)

    figures_folder = f"{plot_subfolder}/{file_name_base}"

    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

# --- Precision parameters ---

# Integration parameters
integration_bound = 5
integration_epsabs = 1e-7
integration_epsrel = 1e-4

# Fixed Point Finder precision parameters (optional)
abs_tol_FPE = TOL_FPE
min_iter_FPE = MIN_ITER_FPE
max_iter_FPE = MAX_ITER_FPE
blend_FPE = BLEND_FPE

# --- Save configuration ---

save_CSV = True
save_pickle = True

if save_CSV or save_pickle:
    data_folder = f"./data/{folder_name}"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    data_subfolder = f"{data_folder}/{subfolder_name}"

    if not os.path.exists(data_subfolder):
        os.makedirs(data_subfolder)

    file_path_CSV = os.path.join(data_subfolder, file_name_base + ".csv")
    file_path_pkl = os.path.join(data_subfolder, file_name_base + ".pkl")

# --- Initialization ---
alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

if decreasing_alpha:
    alphas = alphas[::-1]  # Reverse order for decreasing alpha sweep

# Outputs
opt_reg_param_results = np.full(n_alpha_pts, np.nan)
opt_loss_param_results = np.full(n_alpha_pts, np.nan)
ms_results = np.full(n_alpha_pts, np.nan)
qs_results = np.full(n_alpha_pts, np.nan)
Vs_results = np.full(n_alpha_pts, np.nan)
m_hat_results = np.full(n_alpha_pts, np.nan)
q_hat_results = np.full(n_alpha_pts, np.nan)
V_hat_results = np.full(n_alpha_pts, np.nan)
excess_gen_error_results = np.full(n_alpha_pts, np.nan)
estim_error_results = np.full(n_alpha_pts, np.nan)
angle_teacher_student_results = np.full(n_alpha_pts, np.nan)
rs_values_results = np.full(n_alpha_pts, np.nan)
time_results = np.full(n_alpha_pts, np.nan)

# CSV Header
if save_CSV:
    header_CSV = "alpha,opt_reg_param,opt_loss_param,m,q,V,m_hat,q_hat,V_hat,gen_error,estim_error,angle_teacher_student,rs_value,time_sec\n"
    with open(file_path_CSV, "w") as f:
        f.write(header_CSV)

# --- Alpha sweep with incremental save ---

print(f"Starting alpha sweep for {loss_fun_name} loss with optimal {loss_param_name} for {f_min.__name__}")
print(f"Using {reg_fun_name} regularization with optimal parameter for {f_min.__name__}")
print(f"Alpha range: alpha_min={alpha_min}, alpha_max={alpha_max}, n_alpha_pts={n_alpha_pts}")
print(f"Noise model parameters: {noise}")

current_initial_cond = tuple(initial_cond_fpe)
f_kwargs = {"reg_param": initial_guess_reg_param}
f_hat_kwargs = {loss_param_name: initial_guess_loss_param}
funs = [excess_gen_error, estimation_error, angle_teacher_student]
funs_args = [{**noise}, {}, {}]
last_opt_reg_param = initial_guess_reg_param
last_opt_loss_param = initial_guess_loss_param

for idx, alpha in enumerate(tqdm(alphas, desc="Alpha Sweep")):

    f_hat_kwargs = {
        "alpha": alpha,
        **loss_parameters,
        **noise
    }

    opt_reg_param, opt_loss_param, m, q, V, m_hat, q_hat, V_hat, excess_gen_err, estim_err, rs_value = (np.nan,) * 11 # Initialize to NaN
    success = False

    point_start_time = time.time()
    try:
        # Attempt to find optimal regularization parameter
        (opt_reg_param, opt_loss_param), (m, q, V, m_hat, q_hat, V_hat), errors = \
            find_optimal_reg_and_loss_param_function(
                f_func=f_L2_reg,
                f_hat_func=f_hat_decorrelated_noise_TI,
                f_kwargs=f_kwargs,
                f_hat_kwargs=f_hat_kwargs,
                initial_guess_reg_and_loss_param=[last_opt_reg_param, last_opt_loss_param],
                initial_cond_fpe=current_initial_cond,
                loss_param_name=loss_param_name,
                alpha_key='alpha',
                funs=funs,
                funs_args=funs_args,
                f_min=f_min,
                f_min_args=f_min_args,
                barrier_V_threshold=barrier_V_threshold,
                barrier_RS_threshold=barrier_RS_threshold,
                barrier_penalty=barrier_penalty,
                RS_func= RS_Tukey_decorrelated_noise_TI_l2_reg,
                min_reg_param= min_reg_param,
                min_loss_param=min_loss_param,
                verbose=False
            )

        point_end_time = time.time()
        point_duration = point_end_time - point_start_time

        # Unpack errors
        excess_gen_err = errors[0]
        estim_err = errors[1]
        angle_teacher_student_val = errors[2]

        if np.all(np.isfinite([m, q, V, m_hat, q_hat, V_hat, excess_gen_err, estim_err])):
            try:
                rs_kwargs = {
                    "alpha": alpha,
                    **loss_parameters,
                    "reg_param":opt_reg_param,
                    loss_param_name: opt_loss_param,
                    **noise,
                    "integration_bound": integration_bound,
                    "integration_epsabs": integration_epsabs,
                    "integration_epsrel": integration_epsrel
                }
                rs_value = RS_Tukey_decorrelated_noise_TI_l2_reg(m, q, V, **rs_kwargs)
            except Exception as e_rs:
                print(f"\nWarning: RS computation failed for alpha={alpha:.2e}: {e_rs}")

            # Success
            if np.isfinite(rs_value):
                success = True
                current_initial_cond = (m, q, V)  # Warm start for next alpha
                last_opt_reg_param = opt_reg_param  # Update last optimal reg param
                last_opt_loss_param = opt_loss_param  # Update last optimal loss param

    except (ConvergenceError, ValueError, FloatingPointError, OverflowError, MinimizationError) as e:
        print(f"\nWarning: optimization failed for alpha={alpha:.2e}. Error: {type(e).__name__}")
    except Exception as e:
        print(f"\nUnexpected error for alpha={alpha:.2e}: {e}")

    # --- Save incremental results ---
    if success:
        opt_reg_param_results[idx] = opt_reg_param
        opt_loss_param_results[idx] = opt_loss_param
        ms_results[idx] = m
        qs_results[idx] = q
        Vs_results[idx] = V
        m_hat_results[idx] = m_hat
        q_hat_results[idx] = q_hat
        V_hat_results[idx] = V_hat
        excess_gen_error_results[idx] = excess_gen_err
        estim_error_results[idx] = estim_err
        angle_teacher_student_results[idx] = angle_teacher_student_val
        rs_values_results[idx] = rs_value
        time_results[idx] = point_duration

        if save_CSV:
            try:
                with open(file_path_CSV, "a") as f:
                    rs_str = f"{rs_value:.8e}" if np.isfinite(rs_value) else "0.0"
                    f.write(f"{alpha:.8e},{opt_reg_param:.8e},{opt_loss_param:.8e},{m:.8e},{q:.8e},{V:.8e},{m_hat:.8e},{q_hat:.8e},{V_hat:.8e},{excess_gen_err:.8e},{estim_err:.8e},{angle_teacher_student_val},{rs_str},{point_duration:.4f}\n")
                    f.flush()
            except IOError as e:
                print(f"Error writing to {file_path_CSV}: {e}")

# --- Final save with pickle ---
if save_pickle:
    final_results_dict = {
        "description": f"Alpha sweep for {loss_fun_name} loss with {reg_fun_name} regularization, optimal regularization and loss ({loss_param_name}) parameters for {f_min}.",
        "loss_fun": loss_fun_name,
        "loss_parameters": loss_parameters,
        "reg_fun": reg_fun_name,
        "f_min": f_min.__name__,
        "f_min_args": f_min_args,
        "barrier_V_threshold": barrier_V_threshold,
        "barrier_RS_threshold": barrier_RS_threshold,
        "barrier_penalty": barrier_penalty,
        "initial_cond_fpe": initial_cond_fpe,
        "initial_guess_reg_param": initial_guess_reg_param,
        "min_reg_param": min_reg_param,
        "noise_model": "decorrelated_noise",
        "noise_parameters": noise,
        "integration_bound": integration_bound,
        "integration_epsabs": integration_epsabs,
        "integration_epsrel": integration_epsrel,
        "alpha_range": {"min": alpha_min, "max": alpha_max, "n_points": n_alpha_pts},
        "alphas": alphas,
        "opt_reg_params": opt_reg_param_results,
        "opt_loss_params": opt_loss_param_results,
        "ms": ms_results,
        "qs": qs_results,
        "Vs": Vs_results,
        "m_hats": m_hat_results,
        "q_hats": q_hat_results,
        "V_hats": V_hat_results,
        "excess_gen_error": excess_gen_error_results,
        "estim_error": estim_error_results,
        "angle_teacher_student": angle_teacher_student_results,
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

if save_plots:
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

    # ---- PLOT: V, RS and angle ----
    if any(k in plotted_values for k in ["V", "RS", "angle_teacher_student"]):
        plt.figure()
        if "V" in plotted_values:
            safe_plot(alphas, Vs_results, label="V", linestyle='-')
        if "RS" in plotted_values:
            safe_plot(alphas, rs_values_results, label="RS", linestyle='--')
        if "angle_teacher_student" in plotted_values:
            safe_plot(alphas, angle_teacher_student_results, label="Angle Teacher-Student", linestyle='-.')
        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("V / RS / Angle")
        plt.title("V, RS and angle vs alpha")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "V_RS_angle_vs_alpha.png"))
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
        plt.title("m_hat, q_hat and V_hat vs alpha")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "mqV_hats_vs_alpha.png"))
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

    # ---- PLOT: Optimal regularization parameter ----
    if "opt_reg_param" in plotted_values:
        plt.figure()
        if all(opt_reg_param_results > 0):
            safe_plot(alphas, opt_reg_param_results, label="Optimal Reg Param", linestyle='-')
        else:
            safe_plot(alphas, np.abs(opt_reg_param_results), label="Optimal Reg Param (abs)", linestyle='-')

        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("Optimal Regularization Parameter")
        plt.title("Optimal Regularization Parameter vs alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "opt_reg_param_vs_alpha.png"))
        plt.close()

    # ---- PLOT: Optimal loss parameter ----
    if "opt_loss_param" in plotted_values:
        plt.figure()
        safe_plot(alphas, opt_loss_param_results, label="Optimal Loss Param", linestyle='-')
        plt.xscale("log")
        plt.xlabel(r"$\alpha$")
        plt.yscale("log")
        plt.ylabel("Optimal Loss Parameter")
        plt.title("Optimal Loss Parameter vs alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, "opt_loss_param_vs_alpha.png"))
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
