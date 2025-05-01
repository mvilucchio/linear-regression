import numpy as np
import os
import pickle
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- Imports directs depuis votre package ---
# Sweeps
from linear_regression.sweeps.alpha_sweeps import sweep_alpha_optimal_lambda_hub_param_fixed_point
# Fonctions SE
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import f_hat_xigamma_mod_Tukey_decorrelated_noise
# Observables & Erreurs
from linear_regression.aux_functions.observables_state_evolution import (
    gen_error,
    estimation_error,
    m_overlap,
    q_overlap,
    V_overlap,
    excess_gen_error
)
# Fonction de stabilité RS (E2)
from linear_regression.aux_functions.stability_functions import RS_E2_xigamma_mod_Tukey_decorrelated_noise
# Résolution Point Fixe et Optimisation
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.optimality_finding import find_optimal_reg_and_huber_parameter_function
from linear_regression.utils.errors import ConvergenceError, MinimizationError

# Fonctions ERM et Génération de données
from linear_regression.data.generation import data_generation, measure_gen_decorrelated
from linear_regression.erm.erm_solvers import find_coefficients_mod_Tukey
# ---------------------------------------------

integration_bound = 7 # Limite d'intégration pour RS_E2

# --- Wrapper pour f_hat utilisant 'a' comme 'tau' ---
def f_hat_tukey_wrapper_for_tau_opt(m, q, V, a, # Reçoit 'a' du sweep
                                     c=0.0,       # c est fixé à 0
                                     **kwargs):   # Reçoit alpha, delta_in, etc.
    """
    Wrapper pour f_hat_xigamma_mod_Tukey_decorrelated_noise.
    Utilise l'argument 'a' fourni par le sweep comme paramètre 'tau'.
    Fixe c=0.
    """
    return f_hat_xigamma_mod_Tukey_decorrelated_noise(
        m=m, q=q, V=V,
        tau=a, # Utilise 'a' comme tau
        c=c,   # c est fixé
        **kwargs
    )
# -----------------------------------------------------

# --- Wrapper pour la fonction objectif (retourne E_gen_xs) ---
#     (Utilise 'a' comme 'tau' pour le calcul de E2)
def excess_gen_error_wrapper_for_lambda_tau_opt(m, q, V, alpha, a, # Reçoit 'a' (tau)
                                                 barrier_threshold_V=5./4.,
                                                 barrier_threshold_RS=1.0,
                                                 penalty=1e10,
                                                 # Args pour RS_E2 et gen_error (via kwargs):
                                                 delta_in=0.0, delta_out=0.0, percentage=0.0, beta=0.0, c=0.0,
                                                 **kwargs):
    """
    Wrapper pour excess_gen_error avec barrières V et RS.
    Utilise l'argument 'a' comme 'tau' pour le calcul de E2.
    'alpha' et 'a' sont passés par l'optimiseur sous-jacent.
    """
    # 1. Barrière V
    if V >= barrier_threshold_V: return penalty
    # 2. Barrière RS
    try:
        rs_e2_args = {
            'm': m, 'q': q, 'V': V, 'delta_in': delta_in, 'delta_out': delta_out,
            'percentage': percentage, 'beta': beta,
            'tau': a, # Utilise 'a' comme tau
            'c': c,
            'integration_bound': integration_bound
        }
        E2 = RS_E2_xigamma_mod_Tukey_decorrelated_noise(**rs_e2_args)
        if not np.isfinite(E2) or E2 < 0 : return penalty # E2 doit être positif
        RS = alpha * (V**2) * E2
        if RS >= barrier_threshold_RS: return penalty
    except Exception as e_rs:
        print(f"\t\t  -> Erreur calcul E2/RS: {e_rs}. Pénalité.")
        return penalty
    # Si OK, retourne l'erreur excédentaire
    excess_gen_error_args = {
         'm': m, 'q': q, 'V': V, 'delta_in': delta_in, 'delta_out': delta_out,
         'percentage': percentage, 'beta': beta
    }
    return excess_gen_error(**excess_gen_error_args)
# -----------------------------------------------------------

# --- Définition des Paramètres ---
alpha_min, alpha_max = 100, 100
n_alpha_pts = 1
# Paramètres bruit/modèle
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0.0
# Paramètre c
c = 0.0
# Estimations initiales pour lambda ET tau
initial_guess_lambda = alpha_min/20
initial_guess_tau = 1.0
# Paramètres optimisation/barrières
min_reg_param_bound = 1e-8
min_tau_bound = 1e-5 # Borne pour tau
barrier_V_threshold = 1.24 #5.0 / 4.0
barrier_RS_threshold = 1.0
barrier_penalty = 1e10
# Paramètres plot/sauvegarde
plot_gen_error_vs_lambda = True
plot_every_n_alpha = 1
lambda_pts_plot_gen_error = 3
lambda_size_plot_gen_error = 20
tau_pts_plot_gen_error = 3
tau_size_plot_gen_error = 5
csv_save_enabled = True
force_recompute = True

# --- Configuration Fichiers ---
data_folder = "./data/mod_Tukey_c0_lambda_tau_opt_multi_barrier" # Nouveau dossier
os.makedirs(data_folder, exist_ok=True)
param_string = f"alpha_{alpha_min:.1f}_{alpha_max:.1f}_n_{n_alpha_pts}_delta_{delta_in:.1f}_{delta_out:.1f}_eps_{percentage:.1f}_beta_{beta:.1f}_c_{c:.1f}"
barrier_string = f"barrierV{barrier_V_threshold:.2f}_RS{barrier_RS_threshold:.1f}"
file_name_base = f"optimal_lambda_tau_se_tukey_xigamma_{param_string}_{barrier_string}" # Nom de fichier mis à jour
file_name_se_pkl = f"{file_name_base}.pkl"
file_name_se_csv = f"{file_name_base}.csv"
full_path_se_pkl = os.path.join(data_folder, file_name_se_pkl)
full_path_se_csv = os.path.join(data_folder, file_name_se_csv)
print(f"Utilisation du fichier Pickle: {full_path_se_pkl}")
print(f"Utilisation du fichier CSV: {full_path_se_csv}")

# --- Initialisation des listes de résultats ---
all_alphas = []
all_min_obj_values = [] # Stockera min(E_gen_xs)
all_reg_params_opt = [] # Stockera lambda optimal
all_tau_params_opt = []  # Stockera tau optimal
all_gen_errors = []     # Stockera E_gen totale à l'optimum
all_estim_errors = []
all_ms = []
all_qs = []
all_Vs = []
loaded_from_pickle = False
loaded_from_csv = False

# --- Chargement des Données Existantes ---
if not force_recompute:
    # 1. Essayer Pickle
    try:
        if os.path.exists(full_path_se_pkl):
            print(f"Chargement Pickle depuis {full_path_se_pkl}")
            with open(full_path_se_pkl, "rb") as f: data_se = pickle.load(f)
            # Vérification basique des paramètres (ajouter si besoin)
            if data_se.get("c") == c and data_se.get("delta_in") == delta_in and data_se.get("delta_out") == delta_out and \
               data_se.get("percentage") == percentage and data_se.get("beta") == beta and \
               data_se.get("barrier_V_threshold") == barrier_V_threshold and data_se.get("barrier_RS_threshold") == barrier_RS_threshold:
                all_alphas = list(data_se["alphas"])
                all_min_obj_values = list(data_se["min_objective_value"])
                all_reg_params_opt = list(data_se["reg_params_opt"])
                all_tau_params_opt = list(data_se["tau_params_opt"]) # Charger tau
                all_gen_errors = list(data_se["gen_error"])
                all_estim_errors = list(data_se["estim_error"])
                all_ms = list(data_se["ms"])
                all_qs = list(data_se["qs"])
                all_Vs = list(data_se["Vs"])
                barrier_V_threshold = data_se.get("barrier_V_threshold", 5./4.)
                barrier_RS_threshold = data_se.get("barrier_RS_threshold", 1.0)
                loaded_from_pickle = True
                print("Chargement Pickle OK.")
            else: print("Paramètres Pickle incompatibles.")
    except Exception as e_pkl: print(f"Erreur chargement Pickle: {e_pkl}"); all_alphas = []

    # 2. Essayer CSV si Pickle échoue
    if not loaded_from_pickle:
        try:
            if os.path.exists(full_path_se_csv):
                print(f"Chargement CSV depuis {full_path_se_csv}")
                # Réinitialise avant remplissage
                all_alphas, all_min_obj_values, all_reg_params_opt, all_tau_params_opt = [], [], [], []
                all_gen_errors, all_estim_errors, all_ms, all_qs, all_Vs = [], [], [], [], []
                with open(full_path_se_csv, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
                    print(f"  En-tête CSV: {header}") # Vérifie l'en-tête
                    for i, row in enumerate(reader):
                        try: # Conversion float
                            all_alphas.append(float(row[0]))
                            all_min_obj_values.append(float(row[1]))
                            all_reg_params_opt.append(float(row[2]))
                            all_tau_params_opt.append(float(row[3])) # Lit tau
                            all_gen_errors.append(float(row[4]))
                            all_estim_errors.append(float(row[5]))
                            all_ms.append(float(row[6]))
                            all_qs.append(float(row[7]))
                            all_Vs.append(float(row[8]))
                        except (ValueError, IndexError) as e_row: print(f"  -> Ligne CSV {i+2} invalide: {row} ({e_row})")
                if all_alphas: loaded_from_csv = True; print(f"Chargement CSV OK ({len(all_alphas)} points).")
                else: print(f"Fichier CSV vide ou invalide.")
            else: print(f"Fichier CSV non trouvé.")
        except Exception as e_csv: print(f"Erreur chargement CSV: {e_csv}"); all_alphas = []
else:
    print("Forçage du recalcul.")

# --- Calcul Théorique (State Evolution) ---
if not loaded_from_pickle and not loaded_from_csv:
    print(f"\nRecalcul SE complet (optimisation lambda et tau)...")
    init_cond_fpe = (0.9, 0.9, 0.1)
    initial_guesses = (initial_guess_lambda, initial_guess_tau)

    # Prépare le fichier CSV
    if csv_save_enabled:
        try:
            with open(full_path_se_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([ # Nouvel en-tête
                    "alpha", "min_objective", "reg_param_opt", "tau_param_opt",
                    "gen_error", "estim_error", "m", "q", "V"
                ])
            print(f"Fichier CSV créé/réinitialisé: {full_path_se_csv}")
        except IOError as e: print(f"Attention: Impossible d'écrire CSV: {e}"); csv_save_enabled = False

    # Arguments fixes
    f_kwargs = {}
    f_hat_kwargs = {
        "delta_in": delta_in, "delta_out": delta_out,
        "percentage": percentage, "beta": beta, "c": c,
        "integration_bound": integration_bound, "epsabs": 1e-12, "epsrel": 1e-8
    }
    f_min_args_wrapper = { 
         "delta_in": delta_in, "delta_out": delta_out,
         "percentage": percentage, "beta": beta, "c": c,
         "barrier_threshold_V": barrier_V_threshold,
         "barrier_threshold_RS": barrier_RS_threshold,
         "penalty": barrier_penalty,
         'integration_bound': integration_bound,
    }
    observables = [excess_gen_error, estimation_error, m_overlap, q_overlap, V_overlap]
    observables_args = [ # Args pour les observables finales
        {"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
        {}, {}, {}, {}
    ]

    # Boucle principale
    alphas_to_process = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
    old_initial_cond_fpe = init_cond_fpe
    old_reg_param_opt, old_tau_param_opt = initial_guesses

    # Réinitialise les listes
    all_alphas, all_min_obj_values, all_reg_params_opt, all_tau_params_opt = [], [], [], []
    all_gen_errors, all_estim_errors, all_ms, all_qs, all_Vs = [], [], [], [], []

    for idx, alpha_current in enumerate(alphas_to_process):
        print(f"\n--- Traitement alpha = {alpha_current:.3f} ({idx+1}/{n_alpha_pts}) ---")
        current_f_hat_kwargs = f_hat_kwargs.copy()
        current_f_hat_kwargs["alpha"] = alpha_current
        current_f_min_args = f_min_args_wrapper.copy()

        # --- Optimisation Principale (lambda ET tau) ---
        print(f"  -> Recherche de (lambda, tau) optimal (démarrage à {old_reg_param_opt:.5f}, {old_tau_param_opt:.5f})...")
        try:
            # Appel au sweep pour 2 paramètres (simulé pour un seul alpha)
            _alphas_sweep, f_min_val_sweep, (current_reg_param_opt_sweep, current_tau_opt_sweep), funs_values_sweep = \
                sweep_alpha_optimal_lambda_hub_param_fixed_point(
                    f_func=f_L2_reg,
                    f_hat_func=f_hat_tukey_wrapper_for_tau_opt,
                    alpha_min=alpha_current, alpha_max=alpha_current, n_alpha_pts=1,
                    inital_guess_params=(old_reg_param_opt, old_tau_param_opt),
                    f_kwargs=f_kwargs.copy(),
                    f_hat_kwargs=current_f_hat_kwargs,
                    initial_cond_fpe=old_initial_cond_fpe,
                    funs=observables,
                    funs_args=observables_args,
                    f_min=excess_gen_error_wrapper_for_lambda_tau_opt,
                    f_min_args=current_f_min_args,               # <<< Args pour le wrapper (sans alpha/a)
                    update_f_min_args=True,
                    min_reg_param=min_reg_param_bound,
                    min_huber_param=min_tau_bound,
                    decreasing=False,
                    update_fmin_huber_args=True,
                )
            # Extrait les résultats pour cet alpha unique
            f_min_val = f_min_val_sweep[0]
            current_reg_param_opt = current_reg_param_opt_sweep[0]
            current_tau_opt = current_tau_opt_sweep[0]
            current_funs_values = [val[0] for val in funs_values_sweep]
            m, q, V = current_funs_values[2], current_funs_values[3], current_funs_values[4] # Récupère m, q, V de l'optimum

            # Stockage des résultats
            all_alphas.append(alpha_current)
            all_min_obj_values.append(f_min_val)
            all_reg_params_opt.append(current_reg_param_opt)
            all_tau_params_opt.append(current_tau_opt)
            current_gen_error, current_estim_error, _, _, _ = current_funs_values # m,q,V déjà extraits
            all_gen_errors.append(current_gen_error)
            all_estim_errors.append(current_estim_error)
            all_ms.append(m)
            all_qs.append(q)
            all_Vs.append(V)

            print(f"  -> Trouvé lambda_opt = {current_reg_param_opt:.5f}, tau_opt = {current_tau_opt:.5f} (Min Excess Egen={f_min_val:.5f}, V={V:.4f})")

            # Mise à jour pour la prochaine itération
            old_reg_param_opt = current_reg_param_opt
            old_tau_param_opt = current_tau_opt
            old_initial_cond_fpe = (m, q, V)

            # Sauvegarde CSV
            if csv_save_enabled:
                try:
                    with open(full_path_se_csv, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([ # Nouvel ordre avec tau
                            alpha_current, f_min_val, current_reg_param_opt, current_tau_opt,
                            current_gen_error, current_estim_error,
                            m, q, V
                        ])
                except IOError as e: print(f"Attention: Impossible d'écrire CSV: {e}")

        except (MinimizationError, ConvergenceError, ValueError, TypeError) as e_opt:
             print(f"ERREUR lors de l'optimisation pour alpha={alpha_current:.3f}: {e_opt}")
             traceback.print_exc()
             all_alphas.append(alpha_current)
             # Stockage NaN
             all_min_obj_values.append(np.nan); all_reg_params_opt.append(np.nan); all_tau_params_opt.append(np.nan);
             all_gen_errors.append(np.nan); all_estim_errors.append(np.nan); all_ms.append(np.nan); all_qs.append(np.nan); all_Vs.append(np.nan);
             print("  -> Poursuite avec les dernières valeurs opt connues.")

        # --- Plot 3D Egen_xs vs Lambda et Tau ---
        if plot_gen_error_vs_lambda and idx % plot_every_n_alpha == 0:
            print(f"  -> Génération du plot 3D Egen_xs vs (Lambda, Tau) (autour de lambda={old_reg_param_opt:.4f}, tau={old_tau_param_opt:.4f})...")

            # Définition des plages pour lambda et tau autour du point précédent
            lambda_plot_min = max(min_reg_param_bound, old_reg_param_opt / lambda_size_plot_gen_error)
            lambda_plot_max = old_reg_param_opt * lambda_size_plot_gen_error
            lambda_range = np.logspace(np.log10(lambda_plot_min), np.log10(lambda_plot_max), lambda_pts_plot_gen_error)

            tau_plot_min = max(min_tau_bound, old_tau_param_opt / tau_size_plot_gen_error)
            tau_plot_max = old_tau_param_opt * tau_size_plot_gen_error
            tau_range = np.logspace(np.log10(tau_plot_min), np.log10(tau_plot_max), tau_pts_plot_gen_error)

            # Création de la grille
            lambda_grid, tau_grid = np.meshgrid(lambda_range, tau_range)
            egen_xs_grid = np.full_like(lambda_grid, np.nan)

            temp_f_kwargs_plot = f_kwargs.copy()
            # Utilise le f_hat_kwargs de l'alpha courant mais on modifiera tau
            temp_f_hat_kwargs_plot_base = current_f_hat_kwargs.copy()
            plot_initial_cond = old_initial_cond_fpe # Condition initiale pour FPE

            print("     Calcul des points de la surface...")
            # Itération sur la grille
            for i in range(lambda_pts_plot_gen_error):
                for j in range(tau_pts_plot_gen_error):
                    lam_plot = lambda_grid[i, j]
                    tau_plot = tau_grid[i, j]

                    temp_f_kwargs_plot["reg_param"] = lam_plot
                    # Ici on utilise la fonction FPE originale avec tau explicite
                    temp_f_hat_kwargs_plot = temp_f_hat_kwargs_plot_base.copy()
                    temp_f_hat_kwargs_plot['tau'] = tau_plot
                    temp_f_hat_kwargs_plot['c'] = c

                    try:
                        m_plot, q_plot, V_plot = fixed_point_finder(
                            f_L2_reg, f_hat_xigamma_mod_Tukey_decorrelated_noise,
                            plot_initial_cond, temp_f_kwargs_plot, temp_f_hat_kwargs_plot,
                            verbose=False
                        )

                        # Calcule l'erreur excédentaire (sans barrières pour la visualisation)
                        excess_gen_error_args_plot = {
                            'm': m_plot, 'q': q_plot, 'V': V_plot, 'delta_in': delta_in,
                            'delta_out': delta_out, 'percentage': percentage, 'beta': beta
                        }
                        current_excess_gen_error = excess_gen_error(**excess_gen_error_args_plot)

                        # Vérifie la stabilité pour éventuellement masquer des points
                        is_stable = True
                        E2_plot = np.nan
                        if V_plot >= barrier_V_threshold: is_stable = False
                        else:
                            try:
                                rs_e2_args_plot = {**excess_gen_error_args_plot, 'tau':tau_plot, 'c':c, 'integration_bound' : integration_bound}
                                E2_plot = RS_E2_xigamma_mod_Tukey_decorrelated_noise(**rs_e2_args_plot)
                                if not np.isfinite(E2_plot) or E2_plot < 0 or alpha_current * (V_plot**2) * E2_plot >= barrier_RS_threshold:
                                    is_stable = False
                            except Exception as e_stab:
                                print(f"RS ne converge pas : {e_stab}") 
                                is_stable = False
                        if not is_stable: print(f"     ! Point instable (lambda={lam_plot:.2e}, tau={tau_plot:.2e}, V={V_plot:.2f}, RS={alpha_current*(V_plot**2)*E2_plot:.2f})")
                        egen_xs_grid[i, j] = current_excess_gen_error

                    except (ConvergenceError, ValueError, TypeError) as e_plot:
                        egen_xs_grid[i, j] = np.nan
                        print(f"     ! Échec FPE plot (lambda={lam_plot:.2e}, tau={tau_plot:.2e})")
                        pass

            # Création du plot 3D
            print("     Création du plot 3D...")
            if not np.all(np.isnan(egen_xs_grid)): # Vérifie s'il y a des données valides
                fig_3d = plt.figure(figsize=(10, 7))
                ax_3d = fig_3d.add_subplot(111, projection='3d')

                log_lambda = np.log10(lambda_grid)
                log_tau = np.log10(tau_grid)
                log_egen_xs = np.log10(egen_xs_grid)

                # Masque les NaNs pour le plot surface
                valid_mask = ~np.isnan(log_egen_xs)
                z_min = np.nanmin(log_egen_xs[valid_mask])
                z_max = np.nanmax(log_egen_xs[valid_mask])
                norm = plt.Normalize(vmin=z_min, vmax=z_max)
                surf = ax_3d.plot_surface(log_lambda, log_tau, log_egen_xs,cmap=cm.viridis,norm=norm)

                # Ajoute une barre de couleur
                fig_3d.colorbar(surf, norm=norm, shrink=0.5, aspect=5, label='log Excess Gen Error')

                # Marque le point optimal
                ax_3d.scatter(np.log10(old_reg_param_opt), np.log10(old_tau_param_opt), log_egen_xs.min(),
                              marker='o', s=100, c='red', label=f'Optimal ($\\lambda, \\tau$)')
                # Marque le point de départ
                guess_labmda_plot = all_reg_params_opt[-2] if len(all_reg_params_opt) > 1 else initial_guess_lambda
                guess_tau_plot = all_tau_params_opt[-2] if len(all_tau_params_opt) > 1 else initial_guess_tau
                ax_3d.scatter(np.log10(guess_labmda_plot), np.log10(guess_tau_plot), log_egen_xs.min(),
                                marker='x', s=100, c='orange', label=f'Guess ($\\lambda_0, \\tau_0$)')

                # Ajoute des lignes de contour
                ax_3d.contour(log_lambda, log_tau, log_egen_xs, zdir='z', offset=z_min, extend3d=True,
                                cmap=cm.coolwarm, linewidths=0.5, alpha=0.5)
                
                # Ajoute des lignes de contour projete sur le plan XY
                ax_3d.contour(log_lambda, log_tau, log_egen_xs, zdir='z', offset=z_min,
                                cmap=cm.coolwarm, linewidths=0.5, alpha=0.5)


                ax_3d.set_xlabel('log10($\\lambda$)')
                ax_3d.set_ylabel('log10($\\tau$)')
                ax_3d.set_zlabel('$\\log E_{gen}^{xs}$')
                ax_3d.set_title(f'Log Excess Gen Error Landscape for $\\alpha = {alpha_current:.3f}$')
                ax_3d.legend()
                # Ajuster l'angle de vue si nécessaire
                # ax_3d.view_init(elev=20., azim=-65)

                # plot_debug_filename = os.path.join(data_folder, f"debug_3D_egen_vs_lambda_tau_alpha_{alpha_current:.3f}.png")
                # try:
                #     fig_3d.savefig(plot_debug_filename)
                #     print(f"     Plot 3D sauvegardé: {plot_debug_filename}")
                # except Exception as e_save:
                #     print(f"     Erreur sauvegarde plot 3D: {e_save}")
                # plt.close(fig_3d) # Ferme la figure
                plt.show()

            else:
                print("     Aucun point valide à plotter pour la surface 3D.")
        # --- Fin Plot 3D ---
    # --- Fin de la boucle ---
    print("\nSweep SE terminé.")

    # Sauvegarde finale Pickle
    if all_alphas:
        data_se = { # Mise à jour des données sauvegardées
            "alphas": np.array(all_alphas),
            "min_objective_value": np.array(all_min_obj_values), # Min(Egen_xs)
            "reg_params_opt": np.array(all_reg_params_opt),    # Optimal lambda
            "tau_params_opt": np.array(all_tau_params_opt),     # Optimal tau
            "gen_error": np.array(all_gen_errors),            # Egen totale à l'optimum
            "estim_error": np.array(all_estim_errors),
            "ms": np.array(all_ms), "qs": np.array(all_qs), "Vs": np.array(all_Vs),
            "c": c,
            "barrier_V_threshold": barrier_V_threshold,
            "barrier_RS_threshold": barrier_RS_threshold,
            # Sauvegarder aussi les paramètres delta_in, etc. peut être utile
            "delta_in": delta_in, "delta_out":delta_out, "percentage":percentage, "beta":beta,
        }
        try:
            with open(full_path_se_pkl, "wb") as f: pickle.dump(data_se, f)
            print(f"Données SE finales sauvegardées dans {full_path_se_pkl}")
        except IOError as e: print(f"Attention: Impossible de sauvegarder Pickle final: {e}")
    else: print("Aucun résultat SE n'a été généré, sauvegarde Pickle annulée.")


# --- Affichage Final ---
if all_alphas:
    print("\nAffichage des paramètres optimaux et Erreurs SE")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    alphas_plot = np.array(all_alphas)
    reg_params_plot = np.array(all_reg_params_opt)
    tau_params_plot = np.array(all_tau_params_opt) # <<< Utilise tau
    estim_errors_plot = np.array(all_estim_errors)
    excess_gen_errors_plot = np.array(all_min_obj_values) # min(Egen_xs)

    # Filtres NaN (combiné pour les plots qui utilisent plusieurs variables)
    valid_indices_params = ~np.isnan(reg_params_plot) & ~np.isnan(tau_params_plot) & ~np.isnan(alphas_plot)
    valid_indices_estim = ~np.isnan(estim_errors_plot) & ~np.isnan(alphas_plot) & valid_indices_params # Assure que params sont valides aussi
    valid_indices_excess = ~np.isnan(excess_gen_errors_plot) & ~np.isnan(alphas_plot) & valid_indices_params

    # Plot Lambda optimal
    axes[0].plot(alphas_plot[valid_indices_params], reg_params_plot[valid_indices_params],
                 label="Optimal $\\lambda$", color="red", marker='o', linestyle="--")
    axes[0].set_ylabel("Optimal $\\lambda$")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", ls='--'); axes[0].legend()

    # Plot Tau optimal
    axes[1].plot(alphas_plot[valid_indices_params], tau_params_plot[valid_indices_params], # <<< Utilise tau
                 label="Optimal $\\tau$", color="green", marker='s', linestyle="--")
    axes[1].set_ylabel("Optimal $\\tau$")
    axes[1].set_yscale("log"); axes[1].grid(True, which="both", ls='--'); axes[1].legend()

    # Plot Erreurs
    ax2b = axes[2].twinx()
    lns = []
    if np.any(valid_indices_estim):
        line1 = axes[2].plot(alphas_plot[valid_indices_estim], estim_errors_plot[valid_indices_estim],
                             label="SE (Estim Error)", color="black", marker='.', linestyle="--")
        lns.extend(line1)
    if np.any(valid_indices_excess):
        line2 = ax2b.plot(alphas_plot[valid_indices_excess], excess_gen_errors_plot[valid_indices_excess],
                             label="SE (Min Excess Gen Error)", color="blue", marker='x', linestyle=":")
        lns.extend(line2)

    axes[2].set_ylabel("Estimation Error", color='black')
    ax2b.set_ylabel("Min Excess Gen Error", color='blue')
    axes[2].set_yscale("log"); ax2b.set_yscale("log")
    axes[2].tick_params(axis='y', labelcolor='black')
    ax2b.tick_params(axis='y', labelcolor='blue')

    labs = [l.get_label() for l in lns]
    axes[2].legend(lns, labs, loc=0)
    axes[2].set_xlabel(r"$\alpha = n/d$"); axes[2].grid(True, which="both", ls='--')

    # Titre mis à jour pour refléter l'optimisation de tau
    fig.suptitle(f"Tukey(c={c}), L2 opt $(\\lambda, \\tau)$, Barriers $V>={barrier_V_threshold:.2f}, RS>={barrier_RS_threshold:.1f}$")
    plt.xscale("log")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste pour le suptitle

    plot_filename = os.path.join(data_folder, f"optimal_lambda_tau_errors_{file_name_base}.png")
    plt.savefig(plot_filename)
    print(f"Plot final sauvegardé dans {plot_filename}")
    plt.show()
else:
    print("\nAucune donnée SE disponible pour l'affichage.")

# --- Section ERM (Commentée) ---
# ...
