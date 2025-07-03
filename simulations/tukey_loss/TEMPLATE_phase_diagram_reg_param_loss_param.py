import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from tqdm import tqdm
import matplotlib.colors as mcolors

from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.regression.translation_invariant_losses.f_hat_mixture_of_Gaussian import f_hat_decorrelated_noise_TI
from linear_regression.fixed_point_equations.regression.translation_invariant_losses.Tukey_loss_TI import q_int_Tukey_decorrelated_noise_TI_r, V_int_Tukey_decorrelated_noise_TI_r, RS_Tukey_decorrelated_noise_TI_l2_reg
from linear_regression.aux_functions.misc import excess_gen_error, estimation_error
from linear_regression.utils.errors import ConvergenceError
from linear_regression.fixed_point_equations import TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE, BLEND_FPE

# =============================================================================
# Simulation parameters
# =============================================================================

# Loss configuration
loss_fun_name = "Tukey"
loss_parameters = {"q_int_loss_decorrelated_noise_x": q_int_Tukey_decorrelated_noise_TI_r,
                   "m_int_loss_decorrelated_noise_x": None,
                   "V_int_loss_decorrelated_noise_x" : V_int_Tukey_decorrelated_noise_TI_r} # Except the parameter which is sweeped

# Regularization configuration
reg_fun_name = "L2"

# Fixed alpha
alpha = 10.0

# Decorrelated noise parameters
noise = {
    "Delta_in": 0.1,
    "Delta_out": 1.0,
    "percentage": 0.1,
    "beta": 0.0,
}

# lambda - loss_param sweep parameters

reg_param_min = 0.5
reg_param_max = 2.0
n_reg_param_pts = 100
use_reg_logspace = False

loss_param_min = 0.4
loss_param_max = 2.0
n_loss_param_pts = 100
use_loss_logspace = False

initial_cond_fpe = (0.9, 0.8, 0.06)

# Integration parameters (not used for flat tailed losses)
integration_bound = 5
integration_epsabs = 1e-7
integration_epsrel = 1e-4

# -----------------------------------------------------------------------------
# Plotting control
# -----------------------------------------------------------------------------
save_plots = True

# Choose what to plot (possible values: "m", "q", "V", "RS", "m_hat", "q_hat", "V_hat", "excess_gen_error", "estim_error", "time")
plotted_values = [
    "m", "q", 
    "V", "RS",
    "m_hat", "q_hat", "V_hat",
    "excess_gen_error", "estim_error", 
    "time"
]

# =============================================================================
# Directory and file paths
# =============================================================================

save_CSV = True
save_PKL = True

# Base folders
if save_CSV or save_PKL:
    data_folder = f"./data/phase_diagrams_reg_param_loss_param_{loss_fun_name}_{reg_fun_name}_decorrelated_noise"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    subfolder_name = f"{data_folder}/alpha_{alpha:.1f}_noise_{noise['Delta_in']:.2f}_{noise['Delta_out']:.2f}_{noise['percentage']:.2f}_{noise['beta']:.2f}"
    if not os.path.exists(subfolder_name):
        os.makedirs(subfolder_name)

    file_base = f"phase_diagram_reg_param_{reg_param_min:.1f}-{reg_param_max:.1f}_nReg_{n_reg_param_pts}_loss_param_{loss_param_min:.1f}-{loss_param_max:.1f}_nLoss_{n_loss_param_pts}"
    csv_path = os.path.join(subfolder_name, file_base + ".csv")
    pkl_path = os.path.join(subfolder_name, file_base + ".pkl")

if save_plots:
    plot_folder = f"./imgs/phase_diagrams_reg_param_loss_param_{loss_fun_name}_{reg_fun_name}_decorrelated_noise"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_subfolder = (
        f"{plot_folder}/alpha_{alpha:.1f}_noise_{noise['Delta_in']:.2f}_{noise['Delta_out']:.2f}_{noise['percentage']:.2f}_{noise['beta']:.2f}"
    )
    if not os.path.exists(plot_subfolder):
        os.makedirs(plot_subfolder)

    figures_folder = (
        f"{plot_subfolder}/phase_diagram_reg_param_{reg_param_min:.1f}-{reg_param_max:.1f}_nReg_{n_reg_param_pts}_loss_param_{loss_param_min:.1f}-{loss_param_max:.1f}_nLoss_{n_loss_param_pts}"
    )
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

# =============================================================================
# Build grids for reg_param and loss_param
# =============================================================================

if use_reg_logspace:
    reg_params_grid = np.logspace(np.log10(reg_param_min), np.log10(reg_param_max), n_reg_param_pts)
else:
    reg_params_grid = np.linspace(reg_param_min, reg_param_max, n_reg_param_pts)

if use_loss_logspace:
    loss_param_grid = np.logspace(np.log10(loss_param_min), np.log10(loss_param_max), n_loss_param_pts)
else:
    loss_param_grid = np.linspace(loss_param_min, loss_param_max, n_loss_param_pts)

# We will iterate in decreasing order for warm starts
reg_params_iter = reg_params_grid[::-1]
loss_params_iter = loss_param_grid[::-1]

# =============================================================================
# Initialize storage arrays
# =============================================================================

# Outputs
ms_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
qs_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
Vs_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
m_hat_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
q_hat_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
V_hat_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
excess_gen_error_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
estim_error_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
rs_values_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)
time_results = np.full((n_reg_param_pts,n_loss_param_pts), np.nan)

# =============================================================================
# Prepare CSV header
# =============================================================================

if save_CSV:
    header = "reg_param,loss_param,m,q,V,m_hat,q_hat,V_hat,excess_gen_error,estim_error,RS,time_sec\n"
    with open(csv_path, "w") as f:
        f.write(header)

# =============================================================================
# Fixed-point solver precision settings
# =============================================================================

abs_tol_FPE = TOL_FPE
min_iter_FPE = MIN_ITER_FPE
max_iter_FPE = MAX_ITER_FPE
blend_FPE = BLEND_FPE

# =============================================================================
# Main nested sweep: reg_param x loss_param
# =============================================================================

print(f"Starting phase diagram computation for loss='{loss_fun_name}', alpha={alpha:.1f}")
print(f"Reg parameter range: [{reg_param_min}, {reg_param_max}], loss parameter range: [{loss_param_min}, {loss_param_max}]")
print(f"Noise model parameters: {noise}")

# Track warm starts: keep last valid initial condition per column
last_valid_cond_prev_col = tuple(initial_cond_fpe)

for i_iter, reg_param in enumerate(tqdm(reg_params_iter, desc="Reg Param Sweep")):

    i_idx = n_reg_param_pts - 1 - i_iter  # Reverse index for warm start

    # Initialize the warm start for this column
    current_initial_cond = last_valid_cond_prev_col
    last_valid_cond_this_col = None

    for j_iter, loss_param in enumerate(tqdm(loss_params_iter, desc=f"  loss_param (reg={reg_param:.3e})", position=1, leave=False)):

        j_idx = n_loss_param_pts - 1 - j_iter  # Reverse index for warm start

        # Prepare f and f_hat kwargs for this point
        f_kwargs = {"reg_param": reg_param}
        f_hat_kwargs = {
            "alpha": alpha,
            **noise,
            "tau": loss_param, # Change the name of the loss parameter
            **loss_parameters
        }

        m, q, V, m_hat, q_hat, V_hat, excess_gen_err, estim_err, RS_value = (np.nan,)*9
        success = False

        try:
            start_time = time.time()
            # Solve fixed-point equations for (m, q, V)
            m, q, V = fixed_point_finder(
                f_func=f_L2_reg,
                f_hat_func=f_hat_decorrelated_noise_TI,
                initial_condition=current_initial_cond,
                f_kwargs=f_kwargs,
                f_hat_kwargs=f_hat_kwargs,
                abs_tol=abs_tol_FPE,
                min_iter=min_iter_FPE,
                max_iter=max_iter_FPE,
                # update_function=lambda new, old: tuple(blend_FPE * n + (1 - blend_FPE) * o for n, o in zip(new, old)),
                # args_update_function=(blend_FPE,),
                verbose=False
            )

            end_time = time.time()
            point_duration = end_time - start_time

            # If the solver returned finite values, compute the rest
            if np.all(np.isfinite([m, q, V])):
                # Compute the "hat" variables
                m_hat, q_hat, V_hat = f_hat_decorrelated_noise_TI(m, q, V, **f_hat_kwargs)

                # Compute generalization and estimation errors
                excess_gen_err = excess_gen_error(m, q, V, **noise)
                estim_err = estimation_error(m, q, V, **noise)

                # Compute RS stability condition
                try:
                    rs_kwargs = {
                        "alpha": alpha,
                        "reg_param": reg_param,
                        "tau": loss_param, # Change the name of the loss parameter
                        **noise,
                        **loss_parameters,
                        "integration_bound": integration_bound,
                        "integration_epsabs": integration_epsabs,
                        "integration_epsrel": integration_epsrel,
                    }
                    RS_value = RS_Tukey_decorrelated_noise_TI_l2_reg(m, q, V, **rs_kwargs)
                except Exception as e_rs:
                    print(f"\nWarning: RS computation failed for reg={reg_param:.2e}, loss_param={loss_param:.2f}: {e_rs}")

                # Check finiteness before declaring success
                if np.all(np.isfinite([m, q, V, m_hat, q_hat, V_hat, excess_gen_err, estim_err, RS_value])):
                    success = True
                    last_valid_cond_this_col = (m, q, V)
                    current_initial_cond = (m, q, V)

        except (ConvergenceError, ValueError, FloatingPointError, OverflowError) as e:
            print(f"\nWarning: FPE failed for reg={reg_param:.2e}, loss_param={loss_param:.2f}: {type(e).__name__}")
        except Exception as e:
            print(f"\nUnexpected error at reg={reg_param:.2e}, loss_param={loss_param:.2f}: {e}")

        if success:
            ms_results[(i_idx, j_idx)] = m
            qs_results[(i_idx, j_idx)] = q
            Vs_results[(i_idx, j_idx)] = V
            m_hat_results[(i_idx, j_idx)] = m_hat
            q_hat_results[(i_idx, j_idx)] = q_hat
            V_hat_results[(i_idx, j_idx)] = V_hat
            excess_gen_error_results[(i_idx, j_idx)] = excess_gen_err
            estim_error_results[(i_idx, j_idx)] = estim_err
            rs_values_results[(i_idx, j_idx)] = RS_value
            time_results[(i_idx, j_idx)] = point_duration
            # Append to CSV

            if save_CSV:
                try:
                    with open(csv_path, "a") as f:
                        f.write(
                            f"{reg_param:.8e},{loss_param:.8e},"
                            f"{m:.8e},{q:.8e},{V:.8e},"
                            f"{m_hat:.8e},{q_hat:.8e},{V_hat:.8e},"
                            f"{excess_gen_err:.8e},{estim_err:.8e},{RS_value:.8e},{point_duration:.4f}\n"
                        )
                        f.flush()
                except IOError as io_e:
                    print(f"Error writing to CSV at reg={reg_param:.2e}, loss_param={loss_param:.2f}: {io_e}")

    # Update warm start for next reg_param column
    if last_valid_cond_this_col is not None:
        last_valid_cond_prev_col = last_valid_cond_this_col

# =============================================================================
# Final pickle save
# =============================================================================

if save_PKL:
    final_results = {
        "description": f"Phase diagram for {loss_fun_name} with {reg_fun_name} reg at alpha={alpha:.1f}",
        "alpha": alpha,
        "noise": noise,
        "loss_parameters": loss_parameters,
        "reg_param_min": reg_param_min,
        "reg_param_max": reg_param_max,
        "n_reg_param_pts": n_reg_param_pts,
        "use_reg_logspace": use_reg_logspace,
        "loss_param_min": loss_param_min,
        "loss_param_max": loss_param_max,
        "n_loss_param_pts": n_loss_param_pts,
        "use_loss_logspace": use_loss_logspace,
        "reg_params_grid": reg_params_grid,
        "loss_param_grid": loss_param_grid,
        "ms": ms_results,
        "qs": qs_results,
        "Vs": Vs_results,
        "m_hats": m_hat_results,
        "q_hats": q_hat_results,
        "V_hats": V_hat_results,
        "excess_gen_error": excess_gen_error_results,
        "estim_error": estim_error_results,
        "rs_values": rs_values_results,
        "time_sec": time_results,
    }

    print(f"Saving final results to {pkl_path}...")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(final_results, f)
        print("Pickle save completed.")
    except Exception as e:
        print(f"Error during final pickle save: {e}")

print("Phase diagram computation finished.")

# =============================================================================
# Plotting results
# =============================================================================

if save_plots:
    print("Plotting summary figures...")

    X, Y = np.meshgrid(loss_param_grid, reg_params_grid)

    def safe_plot(x_vals, y_vals, label, **kwargs):
        if np.all(np.isnan(y_vals)):
            print(f"Skipping plot for {label}: all values are NaN.")
            return
        plt.plot(x_vals, y_vals, label=label, **kwargs)

    if "m" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                ms_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(ms_results[np.isfinite(ms_results)]), 
                #                      vmax=np.nanmax(ms_results[np.isfinite(ms_results)]))
            )
            plt.colorbar(im, label="m")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of m over (reg_param, loss_param)")

            levels = 5
            CS = plt.contour(
            X, Y, ms_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "m_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot m heatmap: {e}")

    if "q" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                qs_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(qs_results[np.isfinite(qs_results)]), 
                #                      vmax=np.nanmax(qs_results[np.isfinite(qs_results)]))
            )
            plt.colorbar(im, label="q")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of q over (reg_param, loss_param)")

            levels = 5
            CS = plt.contour(
            X, Y, qs_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "q_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot q heatmap: {e}")

    if "V" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                Vs_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(Vs_results[np.isfinite(Vs_results)]), 
                #                      vmax=np.nanmax(Vs_results[np.isfinite(Vs_results)]))
            )
            plt.colorbar(im, label="V")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of V over (reg_param, loss_param)")
            
            levels = 5
            CS = plt.contour(
            X, Y, Vs_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "V_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot V heatmap: {e}")

    if "RS" in plotted_values:
        try:
            rs_min = min(rs_values_results[np.isfinite(rs_values_results)&(rs_values_results > 0)])
            rs_max = max(rs_values_results[np.isfinite(rs_values_results)])
            rs_values_results[~ (np.isfinite(rs_values_results)&(rs_values_results > 0))] = rs_max
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                rs_values_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                norm=mcolors.LogNorm(vmin=rs_min,
                                     vmax=rs_max)
            )
            plt.colorbar(im, label="RS stability condition")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of RS stability condition")

            X, Y = np.meshgrid(loss_param_grid, reg_params_grid)
            levels = [0.10, 0.50, 1.00]
            CS = plt.contour(
            X, Y, rs_values_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "RS_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot RS heatmap: {e}")

    if "m_hat" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                m_hat_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(m_hat_results[np.isfinite(m_hat_results)]), 
                #                      vmax=np.nanmax(m_hat_results[np.isfinite(m_hat_results)]))
            )
            plt.colorbar(im, label="m_hat")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of m_hat over (reg_param, loss_param)")

            levels = 5
            CS = plt.contour(
            X, Y, m_hat_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "m_hat_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot m_hat heatmap: {e}")
    
    if "q_hat" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                q_hat_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(q_hat_results[np.isfinite(q_hat_results)]), 
                #                      vmax=np.nanmax(q_hat_results[np.isfinite(q_hat_results)]))
            )
            plt.colorbar(im, label="q_hat")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of q_hat over (reg_param, loss_param)")

            levels = 5
            CS = plt.contour(
            X, Y, q_hat_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "q_hat_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot q_hat heatmap: {e}")
    
    if "V_hat" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                V_hat_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(V_hat_results[np.isfinite(V_hat_results)]), 
                #                      vmax=np.nanmax(V_hat_results[np.isfinite(V_hat_results)]))
            )
            plt.colorbar(im, label="V_hat")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of V_hat over (reg_param, loss_param)")

            levels = 5
            CS = plt.contour(
            X, Y, V_hat_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "V_hat_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot V_hat heatmap: {e}")
    
    if "excess_gen_error" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                excess_gen_error_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(excess_gen_error_results[np.isfinite(excess_gen_error_results)]), 
                #                      vmax=np.nanmax(excess_gen_error_results[np.isfinite(excess_gen_error_results)]))
            )
            plt.colorbar(im, label="Excess Generalization Error")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of Excess Generalization Error")

            levels = 5
            CS = plt.contour(
            X, Y, excess_gen_error_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "excess_gen_error_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot excess generalization error heatmap: {e}")
    
    if "estim_error" in plotted_values:
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                estim_error_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(estim_error_results[np.isfinite(estim_error_results)]), 
                #                      vmax=np.nanmax(estim_error_results[np.isfinite(estim_error_results)]))
            )
            plt.colorbar(im, label="Estimation Error")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of Estimation Error")

            levels = 5
            CS = plt.contour(
            X, Y, estim_error_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "estim_error_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot estimation error heatmap: {e}")

    if "time" in plotted_values:
        try:
            time_results = np.where(time_results > np.nanpercentile(time_results, 99), np.nan, time_results)
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                time_results,
                origin="lower",
                aspect="auto",
                extent=[loss_param_min, loss_param_max, reg_param_min, reg_param_max],
                # norm=mcolors.LogNorm(vmin=np.nanmin(time_results[np.isfinite(time_results)]),
                #                      vmax=np.nanmax(time_results[np.isfinite(time_results)]))
            )
            plt.colorbar(im, label="Time per point (s)")
            plt.xlabel("loss_param")
            plt.ylabel("Regularization parameter (lambda)")
            plt.title("Heatmap of computation time")

            levels = 5
            CS = plt.contour(
            X, Y, time_results,
            levels=levels,
            colors='black',
            linewidths=1.0
            )
            plt.clabel(CS, inline=True, fmt="%.2f", fontsize=12, colors="black")

            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, "time_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot time heatmap: {e}")

    print(f"All figures saved in '{figures_folder}'.")
