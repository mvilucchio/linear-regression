import numpy as np
import os
import pickle
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

# Sweeps - Matéo
from linear_regression.sweeps.alpha_sweeps import sweep_alpha_optimal_lambda_fixed_point
# Fonctions SE
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import f_hat_fast
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
# Résolution Point Fixe (pour le plot Egen vs lambda et dans l'optimisation)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.optimality_finding import find_optimal_reg_param_function
from linear_regression.utils.errors import ConvergenceError, MinimizationError

# Fonctions ERM et Génération de données
from linear_regression.data.generation import data_generation, measure_gen_decorrelated
from linear_regression.erm.erm_solvers import find_coefficients_mod_Tukey
# ---------------------------------------------

integration_bound = 7 # Limite d'intégration pour RS_E2

# --- Définition des Paramètres ---
alpha_min, alpha_max = 50, 10000
n_alpha_pts = 200
# Paramètres du bruit et du modèle
delta_in, delta_out, percentage, beta = 0.1, 1.0, 0.1, 0
# Paramètres de la perte Tukey (c=0 pour Tukey standard)
tau = 1.0
c = 0.0
# Paramètres de l'optimisation et simulation
min_reg_param_bound = 1e-8
# Paramètres des barrières
barrier_V_threshold = 1.24 #5.0 / 4.0 # Seuil pour V
barrier_RS_threshold = 1.0 # Seuil pour RS = alpha * V^2 * E2
barrier_penalty = 1e10
# Nouveaux paramètres pour le plot et sauvegarde
plot_gen_error_vs_lambda = False # Activer/Désactiver les plots intermédiaires
plot_every_n_alpha = 10          # Afficher le plot tous les N alpha
lambda_pts_plot_gen_error = 30
lambda_size_plot_gen_error = 100
csv_save_enabled = True      # Activer/Désactiver la sauvegarde CSV
force_recompute = False       # Mettre à True pour ignorer les fichiers existants

# --- Configuration Fichiers ---
data_folder = "./data/mod_Tukey_decorrelated_noise_c0_lambda_opt_multi_barrier"
os.makedirs(data_folder, exist_ok=True)

file_name_base = f"optimal_lambda_se_tukey_evolved_alpha_min_{alpha_min}_max_{alpha_max}_delta_in_{delta_in}_delta_out_{delta_out}_percentage_{percentage}_beta_{beta}_tau_{tau}_c_{c}_barrierV{barrier_V_threshold:.2f}"
file_name_se_pkl = f"{file_name_base}.pkl"
file_name_se_csv = f"{file_name_base}.csv"
full_path_se_pkl = os.path.join(data_folder, file_name_se_pkl)
full_path_se_csv = os.path.join(data_folder, file_name_se_csv)

# --- Initialisation des listes de résultats ---
all_alphas = []
all_min_obj_values = []
all_reg_params_opt = []
all_gen_errors = []
all_estim_errors = []
all_ms = []
all_qs = []
all_Vs = []
loaded_from_pickle = False
loaded_from_csv = False

# --- Chargement des Données Existantes ---
if not force_recompute:
    # 1. Essayer de charger le Pickle
    try:
        if os.path.exists(full_path_se_pkl):
            print(f"Chargement des données SE complètes depuis {full_path_se_pkl}")
            with open(full_path_se_pkl, "rb") as f:
                data_se = pickle.load(f)
                # Affecter les données chargées aux listes
                all_alphas = list(data_se["alphas"])
                all_min_obj_values = list(data_se["min_objective_value"])
                all_gen_errors = list(data_se["gen_error"])
                all_estim_errors = list(data_se["estim_error"])
                all_reg_params_opt = list(data_se["reg_params_opt"])
                all_ms = list(data_se["ms"])
                all_qs = list(data_se["qs"])
                all_Vs = list(data_se["Vs"])
                # Récupérer les paramètres fixes
                tau = data_se.get("tau", tau)
                c = data_se.get("c", c)
                barrier_V_threshold = data_se.get("barrier_V_threshold", 5./4.)
                barrier_RS_threshold = data_se.get("barrier_RS_threshold", 1.0)
                loaded_from_pickle = True
                print("Chargement Pickle terminé.")
        else:
            print(f"Fichier Pickle {full_path_se_pkl} non trouvé.")

    except Exception as e_pkl:
        print(f"Erreur lors du chargement du fichier Pickle {full_path_se_pkl}: {e_pkl}")
        all_alphas = [] # Réinitialise les listes en cas d'erreur pickle

    # 2. Si le Pickle n'a pas été chargé, essayer le CSV
    if not loaded_from_pickle:
        try:
            if os.path.exists(full_path_se_csv):
                print(f"Chargement des données SE partielles depuis {full_path_se_csv}")
                with open(full_path_se_csv, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader) # Lire l'en-tête
                    # Réinitialiser les listes avant de remplir depuis CSV
                    all_alphas, all_min_obj_values, all_reg_params_opt = [], [], []
                    all_gen_errors, all_estim_errors, all_ms, all_qs, all_Vs = [], [], [], [], []
                    print(f"  En-tête CSV: {header}")
                    for i, row in enumerate(reader):
                        try:
                            # Convertit chaque colonne en float (gère les NaN si présents)
                            all_alphas.append(float(row[0]))
                            all_min_obj_values.append(float(row[1]))
                            all_reg_params_opt.append(float(row[2]))
                            all_gen_errors.append(float(row[3]))
                            all_estim_errors.append(float(row[4]))
                            all_ms.append(float(row[5]))
                            all_qs.append(float(row[6]))
                            all_Vs.append(float(row[7]))
                        except (ValueError, IndexError) as e_row:
                            print(f"  -> Ligne CSV {i+2} invalide ignorée : {row} (Erreur: {e_row})")
                            continue # Ignore les lignes mal formées
                if all_alphas: # Vérifie si au moins une ligne a été lue
                     loaded_from_csv = True
                     print(f"Chargement CSV terminé. {len(all_alphas)} points récupérés.")
                else:
                     print(f"Fichier CSV {full_path_se_csv} est vide ou invalide.")
            else:
                print(f"Fichier CSV {full_path_se_csv} non trouvé.")
        except Exception as e_csv:
            print(f"Erreur lors du chargement du fichier CSV {full_path_se_csv}: {e_csv}")
            # Réinitialise au cas où le chargement partiel aurait échoué
            all_alphas, all_min_obj_values, all_reg_params_opt = [], [], []
            all_gen_errors, all_estim_errors, all_ms, all_qs, all_Vs = [], [], [], [], []
else:
    print("Forçage du recalcul (force_recompute=True). Les fichiers existants seront ignorés/écrasés.")

# --- Calcul Théorique (State Evolution) ---
# Seulement si aucune donnée n'a été chargée depuis Pickle ou CSV
if not loaded_from_pickle and not loaded_from_csv:
    print(f"\nRecalcul SE complet (avec barrières V>={barrier_V_threshold:.2f}, RS>={barrier_RS_threshold:.1f})...")
    init_cond_fpe = (0.9, 0.9, 0.1)
    initial_guess_lambda = alpha_min/20

    # Prépare le fichier CSV s'il est activé (mode 'w' pour écraser/créer)
    if csv_save_enabled:
        try:
            with open(full_path_se_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Écrit l'en-tête
                writer.writerow([
                    "alpha", "min_objective", "reg_param_opt",
                    "gen_error", "estim_error", "m", "q", "V"
                ])
            print(f"Fichier CSV créé/réinitialisé: {full_path_se_csv}")
        except IOError as e:
            print(f"Attention: Impossible d'écrire dans le fichier CSV {full_path_se_csv}: {e}")
            csv_save_enabled = False
    # Prépare les arguments fixes une seule fois
    f_kwargs = {}
    f_hat_kwargs = {
        "delta_in": delta_in, "delta_out": delta_out,
        "percentage": percentage, "beta": beta,
        "tau": tau, "c": c,
        "integration_bound": 7, "epsabs": 1e-12, "epsrel": 1e-8 # Ajuster si besoin
    }
    f_min_common_args = {
         "delta_in": delta_in, "delta_out": delta_out,
         "percentage": percentage, "beta": beta,
         "tau": tau, "c": c,
         "barrier_threshold_V": barrier_V_threshold,
         "barrier_threshold_RS": barrier_RS_threshold,
         "penalty": barrier_penalty, 'integration_bound': integration_bound
    }
    observables = [excess_gen_error, estimation_error, m_overlap, q_overlap, V_overlap]
    observables_args = [
        {"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
        {}, {}, {}, {}
    ]

    # Boucle principale pour le sweep en alpha
    alphas_to_process = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
    old_initial_cond_fpe = init_cond_fpe
    old_reg_param_opt = initial_guess_lambda

    # Réinitialise les listes pour stocker les nouveaux résultats
    all_alphas, all_min_obj_values, all_reg_params_opt = [], [], []
    all_gen_errors, all_estim_errors, all_ms, all_qs, all_Vs = [], [], [], [], []

    for idx, alpha_current in enumerate(alphas_to_process):
        print(f"\n--- Traitement alpha = {alpha_current:.3f} ({idx+1}/{n_alpha_pts}) ---")
        def gen_error_with_multiple_barriers_inner(m, q, V, **kwargs_inner):
            alpha_loop = alpha_current # Utilise alpha de la boucle
            # Combine les args communs avec ceux passés potentiellement par l'optimiseur
            combined_args = {**f_min_common_args, **kwargs_inner}

            barrier_V = combined_args['barrier_threshold_V']
            barrier_RS = combined_args['barrier_threshold_RS']
            penalty_val = combined_args['penalty']

            # 1. Barrière V
            if V >= barrier_V: return penalty_val
            # 2. Barrière RS
            try:
                # Ne passe que les args nécessaires à RS_E2
                rs_e2_needed_args = {k: v for k, v in combined_args.items() if k in ['delta_in', 'delta_out', 'percentage', 'beta', 'tau', 'c', 'integration_bound']}
                E2 = RS_E2_xigamma_mod_Tukey_decorrelated_noise(m, q, V, **rs_e2_needed_args)
                if not np.isfinite(E2): return penalty_val
                RS = alpha_loop * (V**2) * E2
                if RS >= barrier_RS: return penalty_val
            except Exception as e_rs:
                print(f"\t\t  -> Erreur calcul E2/RS: {e_rs}. Pénalité.")
                return penalty_val
            # 3. Calcul gen_error
            gen_err_needed_args = {k: v for k, v in combined_args.items() if k in ['delta_in', 'delta_out', 'percentage', 'beta']}
            return excess_gen_error(m, q, V, **gen_err_needed_args)
        
        current_f_hat_kwargs = f_hat_kwargs.copy()
        current_f_hat_kwargs["alpha"] = alpha_current
        current_f_min_args = f_min_common_args.copy()

        # --- Optimisation Principale pour cet alpha ---
        print(f"  -> Recherche de lambda optimal (démarrage à {old_reg_param_opt:.5f})...")
        try:
            # Utilisation de la fonction d'optimisation corrigée
            f_min_val, current_reg_param_opt, (m, q, V), current_funs_values = \
                find_optimal_reg_param_function(
                    f_func=f_L2_reg,
                    f_hat_func=f_hat_fast,
                    f_kwargs=f_kwargs.copy(), # Passe une copie
                    f_hat_kwargs=current_f_hat_kwargs,
                    initial_guess_reg_param=old_reg_param_opt,
                    initial_cond_fpe=old_initial_cond_fpe,
                    funs=observables,
                    funs_args=observables_args,
                    f_min=gen_error_with_multiple_barriers_inner,
                    f_min_args=current_f_min_args,
                    min_reg_param=min_reg_param_bound
                )
            current_reg_param_opt = current_reg_param_opt[0]

            # Stockage des résultats pour cet alpha
            all_alphas.append(alpha_current)
            all_min_obj_values.append(f_min_val)
            all_reg_params_opt.append(current_reg_param_opt)
            current_gen_error, current_estim_error, current_m, current_q, current_V = current_funs_values
            all_gen_errors.append(current_gen_error)
            all_estim_errors.append(current_estim_error)
            all_ms.append(current_m)
            all_qs.append(current_q)
            all_Vs.append(current_V)

            print(f"  -> Trouvé lambda_opt = {current_reg_param_opt:.5f} (Excess Egen={current_gen_error:.5f}, V={current_V:.4f})")

            # Mise à jour pour la prochaine itération
            old_reg_param_opt = current_reg_param_opt
            old_initial_cond_fpe = (current_m, current_q, current_V)

            # Sauvegarde CSV incrémentale
            if csv_save_enabled:
                try:
                    with open(full_path_se_csv, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            alpha_current, f_min_val, current_reg_param_opt,
                            current_gen_error, current_estim_error,
                            current_m, current_q, current_V
                        ])
                except IOError as e:
                    print(f"Attention: Impossible d'écrire dans le fichier CSV {full_path_se_csv}: {e}")

        except (MinimizationError, ConvergenceError, ValueError, TypeError) as e_opt:
            print(f"ERREUR lors de l'optimisation pour alpha={alpha_current:.3f}: {e_opt}")
            traceback.print_exc() # Imprime la trace pour aider au débogage
            # Stocke NaN pour indiquer l'échec pour cet alpha
            all_alphas.append(alpha_current)
            all_min_obj_values.append(np.nan)
            all_reg_params_opt.append(np.nan)
            all_gen_errors.append(np.nan)
            all_estim_errors.append(np.nan)
            all_ms.append(np.nan)
            all_qs.append(np.nan)
            all_Vs.append(np.nan)
            print("  -> Poursuite avec la dernière valeur lambda_opt connue.")
        
        # --- Plot Egen vs Lambda ---
        if plot_gen_error_vs_lambda and idx % plot_every_n_alpha == 0:
            print(f"  -> Génération du plot Excess Egen vs Lambda (autour de lambda_opti={current_reg_param_opt:.4f})...")
            lambda_plot_min = max(min_reg_param_bound, current_reg_param_opt / lambda_size_plot_gen_error)
            lambda_plot_max = current_reg_param_opt * lambda_size_plot_gen_error
            # Utilisation de linspace en log peut être préférable pour l'échelle log
            lambda_plot_range = np.logspace(np.log10(lambda_plot_min), np.log10(lambda_plot_max), lambda_pts_plot_gen_error)

            gen_errors_for_plot = []
            plot_lambdas = []
            temp_f_kwargs = f_kwargs.copy()
            plot_initial_cond = (current_m, current_q, current_V)

            # Création de la figure pour ce plot spécifique
            fig_debug, ax_debug = plt.subplots()

            for lam_plot in lambda_plot_range:
                temp_f_kwargs["reg_param"] = lam_plot
                try:
                    # Attention: Utiliser une copie des kwargs de f_hat aussi
                    temp_f_hat_kwargs = current_f_hat_kwargs.copy()
                    m_plot, q_plot, V_plot = fixed_point_finder(
                        f_L2_reg, f_hat_fast,
                        plot_initial_cond, temp_f_kwargs, temp_f_hat_kwargs,
                        verbose=False
                    )
                    # Calcule l'erreur de gén sans la barrière pour le plot
                    gen_error_args_plot = {
                        'm': m_plot, 'q': q_plot, 'V': V_plot, 'delta_in': delta_in,
                        'delta_out': delta_out, 'percentage': percentage, 'beta': beta
                    }
                    current_gen_error = excess_gen_error(**gen_error_args_plot)

                    # Vérifie si la solution est stable pour l'afficher différemment
                    is_stable = True
                    if V_plot >= barrier_V_threshold:
                        is_stable = False
                    else:
                        try:
                            rs_e2_args_plot = {**gen_error_args_plot, 'tau':tau, 'c':c, 'integration_bound' : integration_bound}
                            E2_plot = RS_E2_xigamma_mod_Tukey_decorrelated_noise(**rs_e2_args_plot)
                            if not np.isfinite(E2_plot) or alpha_current * (V_plot**2) * E2_plot >= barrier_RS_threshold:
                                is_stable = False
                        except:
                            is_stable = False # Instable si E2 échoue

                    # Ajoute aux listes pour le plot
                    if not is_stable:
                        print(f"     ! Instable pour lambda={lam_plot:.4e} (V={V_plot:.4f}, E2={E2_plot:.4f})")
                    gen_errors_for_plot.append(current_gen_error)
                    plot_lambdas.append(lam_plot)

                except (ConvergenceError, ValueError, TypeError) as e_plot:
                     print(f"     ! Échec FPE pour plot lambda={lam_plot:.4e}: {e_plot}")
                     # Ignore ce point pour le plot si FPE échoue
                     pass

            if plot_lambdas:
                ax_debug.plot(plot_lambdas, gen_errors_for_plot, '.-')
                ax_debug.axvline(old_reg_param_opt, color='r', linestyle='--', label=f'Optimal $\\lambda$ = {old_reg_param_opt:.4f}')
                # Indique le lambda qui sera trouvé par l'optimisation (si elle réussit)
                guess_lambda = all_reg_params_opt[-2] if len(all_reg_params_opt)>1 else initial_guess_lambda
                plt.axvline(guess_lambda, color='g', linestyle=':', label=f'Guess $\\lambda$ = {guess_lambda:.4f}')

                ax_debug.set_xscale('log')
                ax_debug.set_yscale('log')
                ax_debug.set_xlabel('$\\lambda$')
                ax_debug.set_ylabel('$E_{gen}^e$')
                ax_debug.set_title(f'Excess Egen vs $\\lambda$ pour $\\alpha = {alpha_current:.3f}$')
                ax_debug.legend()
                ax_debug.grid(True, which="both", ls='--')
                # plot_debug_filename = os.path.join(data_folder, f"debug_egen_vs_lambda_alpha_{alpha:.3f}.png")
                # fig_debug.savefig(plot_debug_filename)
                # print(f"     Plot sauvegardé: {plot_debug_filename}")
                # plt.close(fig_debug) # Ferme la figure spécifique
                plt.show()
            else:
                print("     Aucun point à plotter pour Excess Egen vs Lambda.")
        # --- Fin Plot Excess Egen vs Lambda ---

    # --- Fin de la boucle ---
    print("\nSweep SE terminé.")

    # Sauvegarde finale Pickle
    if all_alphas: # Seulement si des résultats ont été générés
        data_se = {
            "alphas": np.array(all_alphas),
            "min_objective_value": np.array(all_min_obj_values),
            "gen_error": np.array(all_gen_errors),
            "estim_error": np.array(all_estim_errors),
            "reg_params_opt": np.array(all_reg_params_opt),
            "ms": np.array(all_ms),
            "qs": np.array(all_qs),
            "Vs": np.array(all_Vs),
            "tau": tau, "c": c,
            "barrier_V_threshold": barrier_V_threshold,
            "barrier_RS_threshold": barrier_RS_threshold
        }
        try:
            with open(full_path_se_pkl, "wb") as f:
                pickle.dump(data_se, f)
            print(f"Données SE finales sauvegardées dans {full_path_se_pkl}")
        except IOError as e:
             print(f"Attention: Impossible de sauvegarder le fichier Pickle final {full_path_se_pkl}: {e}")
    else:
        print("Aucun résultat SE n'a été généré, sauvegarde Pickle annulée.")

# --- Affichage Final ---
if all_alphas: # S'il y a des données (chargées ou calculées)
    print("\nAffichage (SE)")
    plt.figure(figsize=(8, 6))
    # Filtre les NaN potentiels avant de plotter
    alphas_plot = np.array(all_alphas)
    estim_errors_plot = np.array(all_estim_errors)
    gen_errors_plot = np.array(all_gen_errors)
    ms_plot = np.array(all_ms)
    qs_plot = np.array(all_qs)
    Vs_plot = np.array(all_Vs)
    reg_param_opt_plot = np.array(all_reg_params_opt)
    valid_indices = ~np.isnan(gen_errors_plot) & ~np.isnan(alphas_plot) & ~np.isnan(estim_errors_plot) & ~np.isnan(ms_plot) & ~np.isnan(qs_plot) & ~np.isnan(Vs_plot) & ~np.isnan(reg_param_opt_plot)

    if np.any(valid_indices):
        # plt.plot(alphas_plot[valid_indices], np.abs(estim_errors_plot[valid_indices]-(percentage*(1-beta))**2),
        #         label="SE (|Estim Error - $(\\epsilon(\\beta-1))^2$|)", color="black", marker='.', linestyle="--")
        # plt.plot(alphas_plot[valid_indices], gen_errors_plot[valid_indices],
        #         label="SE (Excess Gen Error)", color="blue", marker='.', linestyle="--")
        # plt.plot(alphas_plot[valid_indices], np.abs(ms_plot[valid_indices]- (1+percentage*(beta-1))),
        #             label="SE (|m - $(1+\\epsilon(\\beta-1))$|)", color="red", marker='.', linestyle="--")
        # plt.plot(alphas_plot[valid_indices], np.abs(1-ms_plot[valid_indices]**2/qs_plot[valid_indices]),
        #             label="SE ($|1-\\eta|$)", color="green", marker='.', linestyle="--")
        # plt.plot(alphas_plot[valid_indices], Vs_plot[valid_indices],
        #             label="SE (V)", color="purple", marker='.', linestyle="--")
        plt.plot(alphas_plot[valid_indices], alphas_plot[valid_indices]/ reg_param_opt_plot[valid_indices],
                label="SE (alpha / lambda_opt)", color="black", marker='.', linestyle="--")
        print(f"Plotting {np.sum(valid_indices)} points valides.")
    else:
        print("Aucune donnée SE valide à plotter.")

    plt.xscale("log")
    plt.xlabel(r"$\alpha = n/d$")
    plt.yscale("log")
    plt.ylabel("Errors")
    plt.title(f"Tukey($\\tau={tau}, c={c}$), L2 opt $\\lambda$, Barriers $V >= {barrier_V_threshold:.2f}, RS>={barrier_RS_threshold:.1f}$")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plot_filename = os.path.join(data_folder, f"gen_error_tukey_xigamma_{alpha_min}_{alpha_max}_multi_barrier_final.png")
    plt.savefig(plot_filename)
    print(f"Plot final sauvegardé dans {plot_filename}")
    plt.show()
else:
    print("\nAucune donnée SE disponible (ni calculée, ni chargée) pour l'affichage.")


# --- Section ERM ---

