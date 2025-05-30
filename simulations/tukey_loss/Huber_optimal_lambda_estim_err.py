import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from linear_regression.sweeps.alpha_sweeps import sweep_alpha_optimal_lambda_fixed_point
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.regression.Huber_loss import f_hat_Huber_decorrelated_noise
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import f_hat_fast
from linear_regression.aux_functions.observables_state_evolution import (
    gen_error,
    estimation_error,
    m_overlap,
    q_overlap,
    V_overlap,
    excess_gen_error
)
# Résolution Point Fixe (pour le plot Egen vs lambda)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.utils.errors import ConvergenceError # Pour gérer les erreurs FPE

# --- Définition de la fonction objectif avec barrière ---
def estim_error_with_barrier(m, q, V, barrier_threshold=5./4., penalty=1e10, **kwargs):
    """
    Wrapper pour gen_error ajoutant une pénalité si V dépasse un seuil.
    """
    if V >= barrier_threshold:
        # print(f"\t\t  -> V={V:.4f} >= {barrier_threshold}. Application de la pénalité.") # Décommenter pour voir quand la barrière est atteinte
        return penalty
    else:
        # Appelle la fonction gen_error originale avec les arguments restants
        return estimation_error(m, q, V, **kwargs)
# -----------------------------------------------------

# --- Définition des Paramètres ---
alpha_min, alpha_max = 50, 10000
n_alpha_pts = 200
# Paramètres du bruit et du modèle
delta_in, delta_out, percentage, beta = 1.0, 1.0, 0.1, 0.0
# Paramètres de la perte Tukey (c=0 pour Tukey standard)
tau = 1.0
c = 0.0
# Paramètres de l'optimisation et simulation
min_reg_param_bound = 1e-8 # Borne inférieure pour lambda
barrier_V_threshold = 1.24 #5.0 / 4.0 # Seuil pour V
barrier_penalty = 1e10 # Pénalité si V >= seuil

# --- Configuration Fichiers ---
data_folder = "./data/Huber_lambda_opt_barrier_estim_err"
os.makedirs(data_folder, exist_ok=True)
file_name_se = f"optimal_lambda_se_huber_alpha_min_{alpha_min}_max_{alpha_max}_n_alpha_pts_{n_alpha_pts}_delta_in_{delta_in}_delta_out_{delta_out}_percentage_{percentage}_beta_{beta}_tau_{tau}_c_{c}.pkl"
full_path_se = os.path.join(data_folder, file_name_se)

# --- Calcul Théorique (State Evolution) ---
if not os.path.exists(full_path_se):
    print(f"Fichier SE {full_path_se} non trouvé. Lancement du calcul SE (avec barrière V>={barrier_V_threshold:.2f})...")
    # Condition initiale pour m, q, V (à ajuster si besoin)
    init_cond_fpe = (0.8, 0.9, 0.1) # Important d'avoir V < 5/4 initialement
    # Estimation initiale pour lambda (sera optimisé)
    initial_guess_lambda = alpha_min/20

    # Dictionnaires d'arguments pour les fonctions SE
    f_kwargs = {}
    f_hat_kwargs = {
        "delta_in": delta_in, "delta_out": delta_out,
        "percentage": percentage, "beta": beta,
        "a": tau, #"c": c,
        #"integration_bound": 7, "epsabs": 1e-12, "epsrel": 1e-8
    }
    # Arguments pour la fonction à minimiser (gen_error) et son wrapper
    f_min_args_barrier = {
         "delta_in": delta_in, "delta_out": delta_out,
         "percentage": percentage, "beta": beta,
         "barrier_threshold": barrier_V_threshold,
         "penalty": barrier_penalty
    }
    # Observables à suivre
    observables = [excess_gen_error, estimation_error, m_overlap, q_overlap, V_overlap]
    observables_args = [
        {"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta}, # Args pour gen_error original
        {}, {}, {}, {}
    ]

    try:
        # Appel avec le wrapper gen_error_with_barrier comme f_min
        (
            alphas_se,
            min_obj_values_se,   # Valeur min de f_min
            reg_params_opt_se,   # Lambda optimal
            funs_values_se,      # Valeurs des observables
        ) = sweep_alpha_optimal_lambda_fixed_point(
            f_func=f_L2_reg,
            f_hat_func=f_hat_Huber_decorrelated_noise,
            alpha_min=alpha_min, alpha_max=alpha_max, n_alpha_pts=n_alpha_pts,
            inital_guess_lambda=initial_guess_lambda,
            f_kwargs=f_kwargs, f_hat_kwargs=f_hat_kwargs,
            initial_cond_fpe=init_cond_fpe,
            funs=observables, funs_args=observables_args,
            f_min=estim_error_with_barrier,
            f_min_args=f_min_args_barrier,
            min_reg_param=min_reg_param_bound,
            decreasing=False
        )
        print("Sweep SE terminé avec succès.")

        # Extraction des overlaps et erreurs réelles (pas la pénalité)
        gen_errors_se, estim_errors_se, ms_se, qs_se, Vs_se = funs_values_se

        # Sauvegarde des résultats
        data_se = {
            "alphas": alphas_se,
            "min_objective_value": min_obj_values_se, # Ce qui a été minimisé (peut être la pénalité)
            "gen_error": gen_errors_se, # L'erreur de géné réelle au point optimal
            "estim_error": estim_errors_se,
            "reg_params_opt": reg_params_opt_se,
            "ms": ms_se, "qs": qs_se, "Vs": Vs_se,
            "tau": tau, "c": c,
            "barrier_V_threshold": barrier_V_threshold,
            "barrier_penalty": barrier_penalty
        }

        with open(full_path_se, "wb") as f:
            pickle.dump(data_se, f)
        print(f"Données SE sauvegardées dans {full_path_se}")

    except Exception as e:
        print(f"ERREUR lors du sweep SE: {e}")
        exit()

else:
     print(f"Chargement des données SE depuis {full_path_se}")
     with open(full_path_se, "rb") as f:
        data_se = pickle.load(f)
        alphas_se = data_se["alphas"]
        ms_se = data_se["ms"]
        qs_se = data_se["qs"]
        Vs_se = data_se["Vs"]
        reg_params_opt_se = data_se["reg_params_opt"]
        estim_errors_se = data_se["estim_error"]
        tau = data_se["tau"]
        c = data_se["c"]
        barrier_V_threshold = data_se.get("barrier_V_threshold", None)
        barrier_penalty = data_se.get("barrier_penalty", None)

# --- Affichage rapide des résultats SE ---
# print("Quelques résultats SE (alpha, lambda_opt, m, q, V, estim_err):")
# indices_to_show = np.linspace(0, len(alphas_se) - 1, 5, dtype=int)
# for i in indices_to_show:
#     print(f"  {alphas_se[i]:.2f}, {reg_params_opt_se[i]:.4f}, {ms_se[i]:.3f}, {qs_se[i]:.3f}, {Vs_se[i]:.3f}, {estim_errors_se[i]:.4f}")

# # --- Simulations Numériques (ERM) ---
# print("\nLancement des simulations ERM...")

# reps = 5 # Nombre de répétitions pour moyennage
# ds = [300] # Dimensions à tester

# plt.figure(figsize=(8, 6))
# # Plot théorie (erreur d'estimation)
# plt.plot(alphas_se, estim_errors_se, label="SE (Estim Error)", color="black", linestyle="--")

# for d in ds:
#     print(f"  Simulation pour d = {d}")
#     # Adapte les alphas pour éviter les trop grandes valeurs si ERM est lent
#     alphas_erm = np.logspace(np.log10(alpha_min), np.log10(min(alpha_max, 100)), n_alpha_pts // 2) # Moins de points pour ERM
#     len_alphas_erm = len(alphas_erm)
    
#     # Interpole lambda optimal pour les alphas de ERM
#     reg_params_opt_erm = np.interp(alphas_erm, alphas_se, reg_params_opt_se)

#     ms_erm = np.empty((len_alphas_erm, 2))
#     qs_erm = np.empty((len_alphas_erm, 2))
#     estim_error_erm = np.empty((len_alphas_erm, 2))
#     # gen_error_erm = np.empty((len_alphas_erm, 2)) # Décommenter si calculé

#     for i, alpha in enumerate(tqdm(alphas_erm, desc=f"ERM d={d}")):
#         n = int(np.around(alpha * d))
#         if n == 0: continue # Évite n=0 pour alpha très petit

#         m_list, q_list, estim_error_list = [], [], [] #, gen_error_list = [], [], [], []

#         for rep in range(reps):
#             xs, ys, xs_gen, ys_gen, wstar = data_generation(
#                 measure_fun=measure_gen_decorrelated,
#                 n_features=d,
#                 n_samples=n,
#                 n_generalization=1000, # Taille de l'ensemble de test
#                 measure_fun_args=(delta_in, delta_out, percentage, beta),
#             )

#             # Initialisation informée pour ERM
#             w_init = ms_se[i] * wstar + np.sqrt(max(0, qs_se[i] - ms_se[i]**2)) * np.random.randn(d)

#             try:
#                 # Appel du solveur ERM avec lambda optimal et tau, c fixes
#                 w = find_coefficients_mod_Tukey(
#                     ys=ys,
#                     xs=xs,
#                     reg_param=reg_params_opt_erm[i], # Lambda optimal pour cet alpha
#                     tau=tau,                         # Tau fixe
#                     c=c,                             # c fixe (ici 0.0)
#                     p=2, # OK pour c=0 (non utilisé) ou c!=0 queue quadratique
#                     initial_w=w_init                 # Initialisation informée
#                 )

#                 m, q = np.dot(w, wstar) / d, np.sum(w**2) / d
#                 m_list.append(m)
#                 q_list.append(q)
#                 estim_error_list.append(np.sum((w - wstar)**2) / d)
#                 # gen_error_list.append(np.mean((xs_gen @ w / np.sqrt(d) - ys_gen)**2)) # Calcul erreur géné empirique

#             except Exception as e_erm: # Capturer spécifiquement les erreurs d'optimisation
#                 print(f"\nAttention: Erreur ERM pour alpha={alpha:.2f}, rep={rep+1}: {e_erm}")
#                 # Ajouter NaN ou ignorer cette répétition
#                 m_list.append(np.nan)
#                 q_list.append(np.nan)
#                 estim_error_list.append(np.nan)
#                 # gen_error_list.append(np.nan)


#         # Calcul des moyennes et écarts-types en ignorant les NaN
#         ms_erm[i] = np.nanmean(m_list), np.nanstd(m_list)
#         qs_erm[i] = np.nanmean(q_list), np.nanstd(q_list)
#         estim_error_erm[i] = np.nanmean(estim_error_list), np.nanstd(estim_error_list)
#         # gen_error_erm[i] = np.nanmean(gen_error_list), np.nanstd(gen_error_list)

#     # Plot ERM (erreur d'estimation)
#     plt.errorbar(
#         alphas_erm, estim_error_erm[:, 0], yerr=estim_error_erm[:, 1],
#         fmt="o", markersize=4, label=f"ERM (Estim Error) d={d}", ls="", capsize=3
#     )

#     # Sauvegarde optionnelle des données ERM
#     # file_name_erm = f"optimal_erm_tukey_xigamma_{alpha_min}_{max(alphas_erm):.0f}_{len_alphas_erm}_{d}_{delta_in}_{delta_out}_{percentage}_{beta}_{tau}_{c}.pkl"
#     # full_path_erm = os.path.join(data_folder, file_name_erm)
#     # erm_data = { ... }
#     # with open(full_path_erm, "wb") as f: pickle.dump(erm_data, f)

# plt.xscale("log")
# plt.xlabel(r"$\alpha = n/d$")
# plt.yscale("log")
# plt.ylabel("Estimation Error ($1 - 2m + q$)")
# plt.title(f"Tukey ($\\tau={tau}, c={c}$), L2 optimal $\\lambda$, Noise($\\delta_{{in}}={delta_in}, \\delta_{{out}}={delta_out}, \\epsilon={percentage}$)")
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# # Sauvegarde du plot
# plot_filename = os.path.join(data_folder, f"estim_error_tukey_xigamma_{alpha_min}_{alpha_max}.png")
# plt.savefig(plot_filename)
# print(f"Plot sauvegardé dans {plot_filename}")
# plt.show()
