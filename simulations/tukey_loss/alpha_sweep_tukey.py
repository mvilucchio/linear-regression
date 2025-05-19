import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import warnings


from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (f_hat_mod_Tukey_decorrelated_noise, f_hat_xigamma_mod_Tukey_decorrelated_noise, f_hat_fast)
from linear_regression.aux_functions.stability_functions import (RS_E2_mod_Tukey_decorrelated_noise, RS_E2_xigamma_mod_Tukey_decorrelated_noise)
from linear_regression.aux_functions.moreau_proximals import DƔ_proximal_L2
from linear_regression.aux_functions.misc import excess_gen_error
from linear_regression.utils.errors import ConvergenceError
from linear_regression.fixed_point_equations import TOL_FPE, MAX_ITER_FPE, BLEND_FPE

# --- Paramètres de la Simulation ---

# Contrôle des calculs coûteux
CALCULATE_RS = False

# Paramètres physiques fixes
NOM_LOSS = "Tukey_fast"
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0 #attention au nombre de décimales

# Hyperparamètres fixés pour le balayage en alpha
REG_PARAM = 2.0
TAU = 1.0
print(f"Hyperparamètres fixes : lambda={REG_PARAM:.2f}, tau={TAU:.2f}")

# Plage pour alpha
ALPHA_MIN =0.5
ALPHA_MAX = 100000
N_ALPHA_PTS = 10000

# Options d'intégration (utilisées pour RS si CALCULATE_RS=True)
INTEGRATION_BOUND = 5
INTEGRATION_EPSABS = 1e-7
INTEGRATION_EPSREL = 1e-4
DEFAULT_N_STD = 4 # Nombre d'écarts-types pour l'intégration en w

# Options pour le solveur de point fixe
FPE_ABS_TOL = 1e-8
#FPE_MAX_ITER = 10000
#FPE_BLEND = 0.85
#FPE_MIN_ITER = 50

# Condition initiale pour le premier alpha
initial_cond_fpe = (9.91940240e-01,9.84089626e-01,2.01330592e-03)

# Configuration sauvegarde
DATA_FOLDER = "./data/alpha_sweeps_tukey" # Dossier dédié
FILE_NAME_BASE = f"alpha_sweep_{NOM_LOSS}_alpha_min_{ALPHA_MIN:.1f}_alpha_max_{ALPHA_MAX:.1f}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_delta_in_{DELTA_IN}_delta_out_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
FILE_PATH_CSV = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".csv")
FILE_PATH_PKL = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".pkl") # Pour sauvegarde finale

# Créer le dossier si nécessaire
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# --- Initialisation et Préparation CSV ---
alphas = np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), N_ALPHA_PTS)

# Tableaux pour stocker tous les résultats (pour la sauvegarde pickle finale)
ms_results = np.full(N_ALPHA_PTS, np.nan)
qs_results = np.full(N_ALPHA_PTS, np.nan)
Vs_results = np.full(N_ALPHA_PTS, np.nan)
m_hat_results = np.full(N_ALPHA_PTS, np.nan)
q_hat_results = np.full(N_ALPHA_PTS, np.nan)
V_hat_results = np.full(N_ALPHA_PTS, np.nan)
gen_error_results = np.full(N_ALPHA_PTS, np.nan)
rs_values_results = np.full(N_ALPHA_PTS, np.nan)
time_results = np.full(N_ALPHA_PTS, np.nan)

# Écriture de l'en-tête du fichier CSV
HEADER_CSV = "alpha,m,q,V,m_hat,q_hat,V_hat,gen_error,rs_value,time_sec\n"
try:
    with open(FILE_PATH_CSV, "w") as f: # Mode 'w' pour écraser ou créer
        f.write(HEADER_CSV)
except IOError as e:
    print(f"Erreur: Impossible d'écrire l'en-tête dans {FILE_PATH_CSV}. Erreur: {e}")
    exit()

# --- Balayage en Alpha avec Sauvegarde Incrémentale ---
print(f"Calcul des points fixes pour alpha de {ALPHA_MIN:.1e} à {ALPHA_MAX:.1e}...")
current_initial_cond = tuple(initial_cond_fpe)

f_kwargs = {"reg_param": REG_PARAM} # Lambda est fixe

for idx, alpha in enumerate(tqdm(alphas, desc="Balayage Alpha")):
    f_hat_kwargs = { # Alpha change, mais aussi d'autres paramètres fixes
        "alpha": alpha,
        "delta_in": DELTA_IN, "delta_out": DELTA_OUT,
        "percentage": PERCENTAGE, "beta": BETA, "tau": TAU, "c": C_TUKEY,
        "integration_bound": DEFAULT_N_STD,
        #"integration_epsabs": INTEGRATION_EPSABS,
        #"integration_epsrel": INTEGRATION_EPSREL
    }

    m, q, V, m_hat, q_hat, V_hat, gen_err, rs_value = (np.nan,) * 8 # Initialiser à NaN
    point_duration = np.nan
    success = False

    point_start_time = time.time()
    try:
        # Recherche du point fixe
        m, q, V = fixed_point_finder(
            f_func=f_L2_reg,
            f_hat_func=f_hat_fast, #f_hat_xigamma_mod_Tukey_decorrelated_noise,
            initial_condition=current_initial_cond,
            f_kwargs=f_kwargs,
            f_hat_kwargs=f_hat_kwargs,
            abs_tol=FPE_ABS_TOL,
            #max_iter=FPE_MAX_ITER,
            #min_iter=FPE_MIN_ITER,
            #update_function=lambda new, old, b: tuple(b * n + (1 - b) * o for n, o in zip(new, old)),
            #args_update_function=(FPE_BLEND,),
            verbose=False
        )

        # Si convergence, calculer les autres quantités
        if np.all(np.isfinite([m, q, V])):
            # Calculer les chapeaux
            m_hat, q_hat, V_hat= f_hat_fast(m, q, V, **f_hat_kwargs) #f_hat_xigamma_mod_Tukey_decorrelated_noise(m, q, V, **f_hat_kwargs)

            # Calculer l'erreur de généralisation
            gen_err = excess_gen_error(m, q, V, DELTA_IN, DELTA_OUT, PERCENTAGE, BETA) # Ou autre mesure

            # Calculer RS (si demandé)
            rs_value = np.nan # Initialiser
            if CALCULATE_RS and np.isfinite(V_hat):
                 try:
                    integral_rs = RS_E2_xigamma_mod_Tukey_decorrelated_noise(
                        m, q, V, DELTA_IN, DELTA_OUT, PERCENTAGE, BETA,
                        TAU, C_TUKEY,
                        INTEGRATION_BOUND, INTEGRATION_EPSABS, INTEGRATION_EPSREL
                    )
                    if not np.isnan(integral_rs):
                        dprox_sq = DƔ_proximal_L2(0.0, V_hat, REG_PARAM)**2
                        rs_value = alpha * dprox_sq * integral_rs
                 except Exception as e_rs:
                     print(f"\nWarn: Erreur calcul RS pour alpha={alpha:.2e}: {e_rs}")
                     rs_value = np.nan

            # Si tout est fini, marquer comme succès et mettre à jour l'init cond
            if np.all(np.isfinite([m, q, V, m_hat, q_hat, V_hat, gen_err])): # RS peut être NaN
                 success = True
                 current_initial_cond = (m, q, V) # Warm start pour le prochain alpha

    except (ConvergenceError, ValueError, FloatingPointError, OverflowError) as e:
         print(f"\nWarn: Échec FPE pour alpha={alpha:.2e}. Erreur: {type(e).__name__}")
    except Exception as e:
         print(f"\nErreur inattendue pour alpha={alpha:.2e}: {e}")

    point_end_time = time.time()
    point_duration = point_end_time - point_start_time

    # --- Sauvegarde Incrémentale CSV ---
    if success:
        # Stocker dans les tableaux numpy pour la sauvegarde finale
        ms_results[idx] = m
        qs_results[idx] = q
        Vs_results[idx] = V
        m_hat_results[idx] = m_hat
        q_hat_results[idx] = q_hat
        V_hat_results[idx] = V_hat
        gen_error_results[idx] = gen_err
        rs_values_results[idx] = rs_value # Sera NaN si non calculé ou erreur
        time_results[idx] = point_duration

        # Écrire dans le fichier CSV
        try:
            with open(FILE_PATH_CSV, "a") as f:
                # Formatter les NaN pour le CSV si nécessaire (ex: chaîne vide)
                rs_str = f"{rs_value:.8e}" if np.isfinite(rs_value) else ""
                f.write(f"{alpha:.8e},{m:.8e},{q:.8e},{V:.8e},{m_hat:.8e},{q_hat:.8e},{V_hat:.8e},{gen_err:.8e},{rs_str},{point_duration:.4f}\n")
                f.flush() # Forcer l'écriture sur le disque
        except IOError as e:
            print(f"Erreur: Impossible d'écrire dans {FILE_PATH_CSV}. Erreur: {e}")
            # Peut-être arrêter ou juste continuer sans sauvegarde CSV

    # else: les valeurs restent NaN dans les tableaux numpy

# --- Sauvegarde Finale Optionnelle avec Pickle ---
final_results_dict = {
    # Métadonnées
    "description": f"Alpha sweep results for {NOM_LOSS} loss with L2 regularization",
    "ALPHA_MIN": ALPHA_MIN, "ALPHA_MAX": ALPHA_MAX, "N_ALPHA_PTS": N_ALPHA_PTS,
    "DELTA_IN": DELTA_IN, "DELTA_OUT": DELTA_OUT, "PERCENTAGE": PERCENTAGE,
    "BETA": BETA, "C_TUKEY": C_TUKEY, "REG_PARAM": REG_PARAM, "TAU": TAU,
    "CALCULATED_RS": CALCULATE_RS, "integration_bound": DEFAULT_N_STD,
    # Données
    "alphas": alphas,
    "ms": ms_results,
    "qs": qs_results,
    "Vs": Vs_results,
    "m_hats": m_hat_results,
    "q_hats": q_hat_results,
    "V_hats": V_hat_results,
    "gen_error": gen_error_results,
    "rs_values": rs_values_results,
    "times_sec": time_results
}

print(f"Sauvegarde finale des résultats complets dans {FILE_PATH_PKL}...")
try:
    with open(FILE_PATH_PKL, "wb") as f:
        pickle.dump(final_results_dict, f)
    print("Sauvegarde finale terminée.")
except Exception as e:
    print(f"Erreur lors de la sauvegarde finale du fichier pickle : {e}")

print("Script de calcul alpha sweep terminé.")
