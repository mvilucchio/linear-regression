import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import warnings
import os

# --- Importation des fonctions ---
try:
    from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
    from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
    from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (
        f_hat_mod_Tukey_decorrelated_noise, RS_alpha_E2_mod_Tukey_decorrelated_noise
    )
    from linear_regression.aux_functions.moreau_proximals import DƔ_proximal_L2
    from linear_regression.utils.errors import ConvergenceError
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    exit()
except Exception as e:
    print(f"Une autre erreur est survenue lors de l'importation : {e}")
    exit()

# --- Paramètres ---

NOM_LOSS = "Tukey_mod"

# Paramètres fixes
ALPHA = 10.0
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0.001

# Paramètres de la grille
N_REG_PARAM_PTS = 100
N_TAU_PTS = 100

# Gamme pour reg_param (lambda)
REG_PARAM_MIN = 0.01
REG_PARAM_MAX = 2.0
USE_REG_LOGSPACE = True

# Gamme pour tau
TAU_MIN = 0.4
TAU_MAX = .85
USE_TAU_LOGSPACE = True

# Options d'intégration
INTEGRATION_BOUND = 15.0
INTEGRATION_EPSABS = 1e-12
INTEGRATION_EPSREL = 1e-10

file_name = f"phase_diagram_{NOM_LOSS}_alpha_{ALPHA:.1f}_deltas_{DELTA_IN}_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}.csv"
data_folder = "./data/"

# save the data in the datafolder in a CSV file
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

file_path = os.path.join(data_folder, file_name)

with open(file_path, "w") as f:
    f.write("reg_param,tau,m,q,V,m_hat,q_hat,V_hat\n") # Header for CSV file

# Condition initiale par défaut (utilisée seulement pour le tout premier point)
DEFAULT_INITIAL_COND = (0.006895984123128191, 1.94187081845284e-04, 0.40232727668693236)

# --- Mise en place de la grille ---
if USE_REG_LOGSPACE:
    # Générons en ordre croissant puis on boucle sur l'index inversé.
    reg_params = np.logspace(np.log10(REG_PARAM_MIN), np.log10(REG_PARAM_MAX), N_REG_PARAM_PTS)
else:
    reg_params = np.linspace(REG_PARAM_MIN, REG_PARAM_MAX, N_REG_PARAM_PTS)

if USE_TAU_LOGSPACE:
    taus = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU_PTS)
else:
    taus = np.linspace(TAU_MIN, TAU_MAX, N_TAU_PTS)

# Stockage des résultats (tuples m, q, V)
# Indexation : [index_reg_param, index_tau, (m=0, q=1, V=2, m_hat=3, q_hat=4, V_hat=5)]
MQV_results = np.full((N_REG_PARAM_PTS, N_TAU_PTS, 6), np.nan)

# --- Boucle de Calcul Adaptée ---
print(f"Calcul du diagramme de phase pour alpha = {ALPHA} (balayage adapté)...")
print(f"Grille : {N_REG_PARAM_PTS} points reg_param [{reg_params[0]:.2e}, {reg_params[-1]:.2e}] (parcouru de max à min)")
print(f"        {N_TAU_PTS} points tau [{taus[0]:.2f}, {taus[-1]:.2f}] (parcouru de max à min)")

""" # Boucle externe sur tau (index j, décroissant) - avec tqdm en position 0
# Utilisation de `position=0` pour la barre externe principale
outer_loop_indices = range(N_TAU_PTS - 1, -1, -1)
for j in tqdm(outer_loop_indices, desc="Progression Tau", position=0):
    tau = taus[j]

    # Initialisation pour le premier reg_param (le plus grand) de cette colonne tau
    if j < N_TAU_PTS - 1:
        last_m, last_q, last_V, last_m_hat, last_q_hat, last_V_hat = MQV_results[N_REG_PARAM_PTS - 1, j + 1, :]
        if np.isfinite(last_m):
            current_initial_cond = (last_m, last_q, last_V)
        else:
            print(f"\nWarning: Init pour tau={tau:.3f} (j={j}) depuis tau précédent échoué. Utilise défaut.")
            current_initial_cond = DEFAULT_INITIAL_COND
    else:
        current_initial_cond = DEFAULT_INITIAL_COND

    # Créer l'itérable pour la boucle interne
    inner_loop_indices = range(N_REG_PARAM_PTS - 1, -1, -1)

    # Envelopper l'itérable avec tqdm pour la barre interne
    # Utilisation de `position=1` pour la placer sous la barre externe
    # Utilisation de `leave=False` pour qu'elle s'efface après chaque boucle externe
    inner_pbar = tqdm(
        inner_loop_indices,
        desc=f"  RegParam (tau={tau:.2f})", # Description indiquant le tau courant 
        position=1,                         # Position sous la barre externe
        leave=False                         # Effacer la barre interne quand terminée
    )

    # Boucle interne sur reg_param (index i, décroissant) en utilisant l'itérateur tqdm
    for i in inner_pbar:
        reg_param = reg_params[i]
        #print(f"reg_param : {reg_param}") """
# Boucle externe sur reg_param (index i, decroissant) - avec tqdm en position 0
# Utilisation de `position=0` pour la barre externe principale
outer_loop_indices = range(N_REG_PARAM_PTS - 1, -1, -1)
for i in tqdm(outer_loop_indices, desc="Progression reg_param", position=0):
    reg_param = reg_params[i]

    # Initialisation pour le premier tau (le plus grand) de cette colonne reg_param
    if i < N_REG_PARAM_PTS - 1:
        last_m, last_q, last_V, last_m_hat, last_q_hat, last_V_hat = MQV_results[i+1, N_TAU_PTS-1, :]
        if np.isfinite(last_m):
            current_initial_cond = (last_m, last_q, last_V)
        else:
            print(f"\nWarning: Init pour reg_param={reg_param:.3f} (i={i}) depuis tau précédent échoué. Utilise défaut.")
            current_initial_cond = DEFAULT_INITIAL_COND
    else:
        current_initial_cond = DEFAULT_INITIAL_COND

    # Créer l'itérable pour la boucle interne
    inner_loop_indices = range(N_TAU_PTS - 1, -1, -1)

    # Envelopper l'itérable avec tqdm pour la barre interne
    # Utilisation de `position=1` pour la placer sous la barre externe
    # Utilisation de `leave=False` pour qu'elle s'efface après chaque boucle externe
    inner_pbar = tqdm(
        inner_loop_indices,
        desc=f"  Tau (reg_param={reg_param:.2f})", # Description indiquant le reg_param courant 
        position=1,                         # Position sous la barre externe
        leave=False                         # Effacer la barre interne quand terminée
    )

    # Boucle interne sur reg_param (index i, décroissant) en utilisant l'itérateur tqdm
    for j in inner_pbar:
        tau = taus[j]
        #print(f"reg_param : {reg_param}")

        f_kwargs = {"reg_param": reg_param}
        f_hat_kwargs = {
            "alpha": ALPHA, "delta_in": DELTA_IN, "delta_out": DELTA_OUT,
            "percentage": PERCENTAGE, "beta": BETA, "tau": tau, "c": C_TUKEY,
            "integration_bound": INTEGRATION_BOUND,
            "integration_epsabs": INTEGRATION_EPSABS, "integration_epsrel": INTEGRATION_EPSREL
        }

        try:
            m, q, V = fixed_point_finder(
                f_func=f_L2_reg,
                f_hat_func=f_hat_mod_Tukey_decorrelated_noise,
                initial_condition=current_initial_cond,
                f_kwargs=f_kwargs,
                f_hat_kwargs=f_hat_kwargs,
                verbose=False
            )

            m_hat, q_hat, V_hat = f_hat_mod_Tukey_decorrelated_noise(
                m, q, V, **f_hat_kwargs
            )

            if np.isfinite(m) and np.isfinite(q) and np.isfinite(V) and np.isfinite(m_hat) and np.isfinite(q_hat) and np.isfinite(V_hat):
                MQV_results[i, j, :] = (m, q, V, m_hat, q_hat, V_hat)
                current_initial_cond = (m, q, V)
            else:
                pass # Garde l'ancienne current_initial_cond

        except (ConvergenceError, FloatingPointError) as e:
            print(f"\nAttention: {type(e).__name__} pour reg={reg_param:.3e}, tau={tau:.3f}. Point ignoré (NaN).")
        except Exception as e:
            print(f"\nErreur inattendue pour reg={reg_param:.3e}, tau={tau:.3f}: {e}.")
            pass # Garde l'ancienne current_initial_cond

        # save the results in a new line in the datafile
        with open(file_path, "a") as f:
            f.write(f"{reg_param},{tau},{m},{q},{V},{m_hat},{q_hat},{V_hat}\n")

# --- Calcul de la condition RS ---
print("Calcul de la condition RS...")
RS_values = np.full((N_REG_PARAM_PTS, N_TAU_PTS), np.nan)
for i in range(N_REG_PARAM_PTS):
    for j in range(N_TAU_PTS):
        m, q, V, m_hat, q_hat, V_hat = MQV_results[i, j, :]
        if np.isfinite(m) and np.isfinite(q) and np.isfinite(V) and np.isfinite(V_hat):
            #print(f"Calcul de RS pour reg_param={reg_params[i]:.3e}, tau={taus[j]:.3f}")
            try:
                RS_values[i, j] = RS_alpha_E2_mod_Tukey_decorrelated_noise(
                    m,
                    q,
                    V,
                    ALPHA,
                    DELTA_IN,
                    DELTA_OUT,
                    PERCENTAGE,
                    BETA,
                    taus[j],
                    C_TUKEY,
                    INTEGRATION_BOUND,
                    INTEGRATION_EPSABS, 
                    INTEGRATION_EPSREL
                    ) * DƔ_proximal_L2(0, V_hat, reg_params[i])**2
            except Exception as e:
                print(f"\nErreur inattendue pour RS: {e}.")
                pass

""" # --- Extraction de V et Visualisation ---
print("Extraction des résultats V et RS et génération du graphique...")

# Extraire seulement V pour le tracé
V_results = MQV_results[:, :, 2]

# Filtrer les RuntimeWarnings concernant les NaN lors du tracé
warnings.filterwarnings("ignore", category=RuntimeWarning)

fig, ax = plt.subplots(figsize=(9, 7))
cmap = plt.cm.viridis

# Création de la grille de coordonnées pour le tracé
T, R = np.meshgrid(taus, reg_params) # Taus sur X, Reg_params sur Y

# Gestion des valeurs V infinies ou très grandes pour une meilleure échelle de couleurs
finite_V = V_results[np.isfinite(V_results)]
vmin = np.min(finite_V) if finite_V.size > 0 else 0

# Plafonner à un percentile élevé pour la Vmax pour éviter que quelques points extrêmes écrasent l'échelle
vmax = np.percentile(finite_V, 99) if finite_V.size > 0 else 1

try:
    pcm = ax.pcolormesh(T, R, V_results, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
except ValueError as e:
     print(f"Erreur lors du tracé pcolormesh : {e}")
     pcm = None

if pcm:
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Paramètre d\'ordre V')

# Ajout de la ligne de contour pour V
try:
    if not np.all(np.isnan(V_results)):
        contour_plot = ax.contour(T, R, V_results, colors='red', linestyles='--', linewidths=2)

        ax.clabel(contour_plot, inline=True, fontsize=8, fmt='V=%1.2f', colors='red')
except ValueError as e:
    print(f"Impossible de tracer la ligne de contour de V : {e}")
except Exception as e:
     print(f"Erreur inattendue lors du tracé du contour : {e}")

# Ajout de la ligne de contour pour RS
try:
    if not np.all(np.isnan(RS_values)):
        contour_RS = ax.contour(T, R, RS_values, colors='blue', linestyles='--', linewidths=2)
        ax.clabel(contour_RS, inline=True, fontsize=8, fmt='RS=%1.2f', colors='blue')
except ValueError as e:
    print(f"Impossible de tracer la ligne de contour de RS : {e}")
except Exception as e:
    print(f"Erreur inattendue lors du tracé du contour de RS : {e}")

if USE_TAU_LOGSPACE:
    ax.set_xscale('log')
if USE_REG_LOGSPACE:
    ax.set_yscale('log')

ax.set_xlabel(r'$\tau$ (Seuil de Tukey)')
ax.set_ylabel(r'$\lambda$ (Paramètre de régularisation)')
ax.set_title(rf'Diagramme de phase pour V (Tukey modifiée quad, $\alpha$ = {ALPHA:.2f})')
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()
 """
print("Script terminé.")
