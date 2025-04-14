import numpy as np
import os
import time
from tqdm import tqdm
import warnings
import pickle

# --- Importation des fonctions ---
try:
    from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
    from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
    from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import f_hat_mod_Tukey_decorrelated_noise
    from linear_regression.utils.errors import ConvergenceError
    from linear_regression.fixed_point_equations import TOL_FPE, MAX_ITER_FPE, BLEND_FPE
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    exit()
except Exception as e:
    print(f"Une autre erreur est survenue lors de l'importation : {e}")
    exit()

# --- Paramètres de la Simulation ---

NOM_LOSS = "Tukey_mod"

# Paramètres physiques fixes
ALPHA = 10.0
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0.001

# Paramètres de la grille
N_REG_PARAM_PTS = 10
N_TAU_PTS = 10

# Gamme pour reg_param (lambda)
REG_PARAM_MIN = 10.0
REG_PARAM_MAX = 11.0
USE_REG_LOGSPACE = False

# Gamme pour tau
TAU_MIN = 1.0
TAU_MAX = 2.0
USE_TAU_LOGSPACE = False

# Options pour la fonction f_hat
# INTEGRATION_BOUND = 15.0
# INTEGRATION_EPSABS = 1e-12
# INTEGRATION_EPSREL = 1e-10

# Options pour le solveur de point fixe
# FPE_ABS_TOL = 1e-7
# FPE_MAX_ITER = 10000
# FPE_BLEND = 0.85

# Condition initiale pour le premier point du balayage
DEFAULT_INITIAL_COND = (0.9, 0.8, 0.06)

# Configuration sauvegarde
DATA_FOLDER = "./data/phase_diagrams_tukey"
FILE_NAME_BASE = f"phase_diagram_{NOM_LOSS}_alpha_{ALPHA:.1f}_deltas_{DELTA_IN}_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
FILE_PATH_CSV = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".csv")
FILE_PATH_PKL = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".pkl")

# Créer le dossier si nécessaire
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# --- Génération des grilles et Initialisation ---
if USE_REG_LOGSPACE:
    reg_params_grid = np.logspace(np.log10(REG_PARAM_MIN), np.log10(REG_PARAM_MAX), N_REG_PARAM_PTS)
else:
    reg_params_grid = np.linspace(REG_PARAM_MIN, REG_PARAM_MAX, N_REG_PARAM_PTS)

if USE_TAU_LOGSPACE:
    taus_grid = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU_PTS)
else:
    taus_grid = np.linspace(TAU_MIN, TAU_MAX, N_TAU_PTS)

# Inverser le parcours
reg_params_iter = reg_params_grid[::-1]
taus_iter = taus_grid[::-1]

# Initialisation des matrices pour stockage final
MQV_results = np.full((N_REG_PARAM_PTS, N_TAU_PTS, 6), np.nan)
time_results = np.full((N_REG_PARAM_PTS, N_TAU_PTS), np.nan)

# Écriture de l'en-tête du fichier CSV
write_header = not os.path.exists(FILE_PATH_CSV) or os.path.getsize(FILE_PATH_CSV) == 0
with open(FILE_PATH_CSV, "a") as f: # Mode 'a' pour ajouter
    if write_header:
        f.write("reg_param,tau,m,q,V,m_hat,q_hat,V_hat,time_sec\n")

# --- Boucle de Calcul Principale ---
print(f"Calcul du diagramme de phase pour alpha = {ALPHA}...")

current_initial_cond = tuple(DEFAULT_INITIAL_COND)
last_valid_cond_prev_col = tuple(DEFAULT_INITIAL_COND)

# Boucle externe sur reg_param (index i, decroissant)
for i_iter, reg_param in enumerate(tqdm(reg_params_iter, desc="Progression Lambda", position=0)):
    idx_reg = N_REG_PARAM_PTS - 1 - i_iter # Index pour le stockage dans l'ordre croissant

    current_initial_cond = last_valid_cond_prev_col
    last_valid_cond_this_col = None

    # Boucle interne sur tau (index j, decroissant)
    for j_iter, tau in enumerate(tqdm(taus_iter, desc=f"  Tau (Lambda={reg_param:.2e})", position=1, leave=False)):
        idx_tau = N_TAU_PTS - 1 - j_iter # Index pour le stockage

        # Mise à jour des kwargs (copies locales)
        local_f_kwargs = {"reg_param": reg_param}
        local_f_hat_kwargs = {
            "alpha": ALPHA, "delta_in": DELTA_IN, "delta_out": DELTA_OUT,
            "percentage": PERCENTAGE, "beta": BETA, "tau": tau, "c": C_TUKEY,
            # "integration_bound": INTEGRATION_BOUND,
            # "integration_epsabs": INTEGRATION_EPSABS,
            # "integration_epsrel": INTEGRATION_EPSREL
        }

        point_start_time = time.time()
        try:
            # Appel du solveur
            m, q, V = fixed_point_finder(
                f_func=f_L2_reg,
                f_hat_func=f_hat_mod_Tukey_decorrelated_noise,
                initial_condition=current_initial_cond,
                f_kwargs=local_f_kwargs,
                f_hat_kwargs=local_f_hat_kwargs,
                #abs_tol=FPE_ABS_TOL,
                #max_iter=FPE_MAX_ITER,
                #update_function=lambda new, old, b: tuple(b * n + (1 - b) * o for n, o in zip(new, old)),
                #args_update_function=(FPE_BLEND,),
                #min_iter=FPE_MIN_ITER,
                verbose=False
            )

            # Recalculer les chapeaux
            m_hat, q_hat, V_hat = f_hat_mod_Tukey_decorrelated_noise(m, q, V, **local_f_hat_kwargs)

            point_end_time = time.time()
            point_duration = point_end_time - point_start_time

            # Vérifier la finitude avant stockage/écriture
            result_tuple = (m, q, V, m_hat, q_hat, V_hat)
            if np.all(np.isfinite(result_tuple)):
                # Stockage dans la matrice numpy
                MQV_results[idx_reg, idx_tau, :] = result_tuple
                time_results[idx_reg, idx_tau] = point_duration

                # Écriture immédiate dans le CSV
                with open(FILE_PATH_CSV, "a") as f:
                    f.write(f"{reg_param},{tau},{m},{q},{V},{m_hat},{q_hat},{V_hat},{point_duration}\n")

                # Mise à jour pour le prochain point
                current_initial_cond = (m, q, V)
                if last_valid_cond_this_col is None:
                    last_valid_cond_this_col = current_initial_cond
            else:
                 # Laisse NaN dans la matrice, n'écrit pas dans CSV
                print(f"\nWarn: Résultat non fini pour reg={reg_param:.3e}, tau={tau:.3f}.")
                # Garde l'ancienne current_initial_cond

        except (ConvergenceError, ValueError, FloatingPointError, OverflowError) as e:
             point_end_time = time.time()
             point_duration = point_end_time - point_start_time
             print(f"\nWarn: Échec FPE pour reg={reg_param:.3e}, tau={tau:.3f} après {point_duration:.2f}s. Erreur: {type(e).__name__}")
             # Laisse NaN dans la matrice et n'écrit rien dans CSV
             # current_initial_cond n'est pas mis à jour

    # Mise à jour de la condition initiale pour la prochaine colonne lambda
    if last_valid_cond_this_col is not None:
        last_valid_cond_prev_col = last_valid_cond_this_col
    # else: garde l'ancienne valeur (qui était celle de la colonne d'avant)

# --- Sauvegarde Finale Optionnelle avec Pickle ---
final_results_dict = {
    "description": f"Phase diagram results for {NOM_LOSS} loss with L2 regularization",
    "ALPHA": ALPHA,
    "DELTA_IN": DELTA_IN,
    "DELTA_OUT": DELTA_OUT,
    "PERCENTAGE": PERCENTAGE,
    "BETA": BETA,
    "C_TUKEY": C_TUKEY,
    "reg_param_min": REG_PARAM_MIN,
    "reg_param_max": REG_PARAM_MAX,
    "n_reg_param_pts": N_REG_PARAM_PTS,
    "use_reg_logspace": USE_REG_LOGSPACE,
    "tau_min": TAU_MIN,
    "tau_max": TAU_MAX,
    "n_tau_pts": N_TAU_PTS,
    "use_tau_logspace": USE_TAU_LOGSPACE,
    "reg_params": reg_params_grid, # Grille ordonnée
    "taus": taus_grid,           # Grille ordonnée
    "MQV_results": MQV_results,  # Tableau 3D [idx_reg, idx_tau, params]
    "time_results": time_results # Tableau 2D [idx_reg, idx_tau]
}

print(f"Sauvegarde finale des résultats complets dans {FILE_PATH_PKL}...")
try:
    with open(FILE_PATH_PKL, "wb") as f:
        pickle.dump(final_results_dict, f)
    print("Sauvegarde finale terminée.")
except Exception as e:
    print(f"Erreur lors de la sauvegarde finale du fichier pickle : {e}")

print("Script de calcul terminé.")
