import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import warnings

# --- Helpers de Plotting (du template) ---
IMG_DIRECTORY = "./imgs/alpha_sweeps_tukey" # Répertoire spécifique

def save_plot(fig, name, formats=["pdf", "png"], directory=IMG_DIRECTORY, date=False):
    """Sauvegarde la figure dans les formats spécifiés."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    for fmt in formats:
        full_path = os.path.join(directory, f"{name}.{fmt}")
        try:
            fig.savefig(full_path, format=fmt, bbox_inches='tight', dpi=300)
            print(f"Plot sauvegardé : {full_path}")
        except Exception as e:
            print(f"Erreur sauvegarde plot {full_path}: {e}")

def set_size(width, fraction=1, subplots=(1, 1)):
    """Définit la taille de la figure pour LaTeX."""
    if width == "thesis": width_pt = 426.79135
    elif width == "beamer": width_pt = 307.28987
    else: width_pt = width
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

# --- Configuration du Plot ---
LOAD_FROM_PKL = True # Privilégier PKL
SAVE_PLOT = True
STYLE_FILE = "./plotting/latex_ready.mplstyle" # Ajustez si nécessaire
FIG_WIDTH = 469.75502 # Largeur LaTeX standard
IMG_FORMATS = ["pdf", "png"]

# --- Paramètres de la Simulation (pour retrouver le fichier) ---
# Doivent correspondre à ceux du script de calcul !
NOM_LOSS_tukey = "Tukey_evolved"
NOM_LOSS_huber = "Huber"
ALPHA_MIN = 50
ALPHA_MAX = 10000
N_ALPHA_PTS = 200
DELTA_IN = 1.0
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0
REG_PARAM = 0.1
TAUS = [1.0,2.0]

# --- Chargement des Données ---
DATA_FOLDER_fixed_reg = "./data/alpha_sweeps_tukey"
DATA_FOLDER_Tukey_opti_reg = "./data/Tukey_evolved_lambda_opt_barrier_estim_err" #"./data/Tukey_evolved_lambda_opt_barrier" # 
DATA_FOLDER_Huber_opti_reg = "./data/Huber_lambda_opt_barrier_estim_err" #"./data/Huber_lambda_opt_barrier" #
#FILE_NAME_BASE = f"alpha_sweep_{NOM_LOSS}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_cin_{DELTA_IN}_cout_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"

FILE_NAME_BASE_Tukey_fixed_reg = [None]*len(TAUS)
FILE_NAME_BASE_Tukey_opti_reg = [None]*len(TAUS)
FILE_NAME_BASE_Huber_fixed_reg = [None]*len(TAUS)
FILE_NAME_BASE_Huber_opti_reg = [None]*len(TAUS)
FILE_PATH_PKL_Tukey_fixed_reg = [None]*len(TAUS)
FILE_PATH_PKL_Huber_fixed_reg = [None]*len(TAUS)
FILE_PATH_PKL_Tukey_opti_reg = [None]*len(TAUS)
FILE_PATH_PKL_Huber_opti_reg = [None]*len(TAUS)

for (i,TAU) in enumerate(TAUS) :
    FILE_NAME_BASE_Tukey_fixed_reg[i] = f"alpha_sweep_{NOM_LOSS_tukey}_alpha_min_{ALPHA_MIN:.1f}_alpha_max_{ALPHA_MAX:.1f}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_delta_in_{DELTA_IN}_delta_out_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
    FILE_NAME_BASE_Tukey_opti_reg[i] = f"optimal_lambda_se_tukey_evolved_alpha_min_{ALPHA_MIN}_max_{ALPHA_MAX}_n_alpha_pts_{N_ALPHA_PTS}_delta_in_{DELTA_IN}_delta_out_{DELTA_OUT}_percentage_{PERCENTAGE}_beta_{BETA}_tau_{TAU}_c_{C_TUKEY:.1f}"
    FILE_NAME_BASE_Huber_fixed_reg[i] = f"alpha_sweep_{NOM_LOSS_huber}_alpha_min_{ALPHA_MIN:.1f}_alpha_max_{ALPHA_MAX:.1f}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_delta_in_{DELTA_IN}_delta_out_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
    FILE_NAME_BASE_Huber_opti_reg[i] = f"optimal_lambda_se_huber_alpha_min_{ALPHA_MIN}_max_{ALPHA_MAX}_n_alpha_pts_{N_ALPHA_PTS}_delta_in_{DELTA_IN}_delta_out_{DELTA_OUT}_percentage_{PERCENTAGE}_beta_{BETA}_tau_{TAU}_c_{C_TUKEY:.1f}"
    FILE_PATH_PKL_Tukey_fixed_reg[i] = os.path.join(DATA_FOLDER_fixed_reg, FILE_NAME_BASE_Tukey_fixed_reg[i] + ".pkl")
    FILE_PATH_PKL_Huber_fixed_reg[i] = os.path.join(DATA_FOLDER_fixed_reg, FILE_NAME_BASE_Huber_fixed_reg[i] + ".pkl")
    FILE_PATH_PKL_Tukey_opti_reg[i] = os.path.join(DATA_FOLDER_Tukey_opti_reg, FILE_NAME_BASE_Tukey_opti_reg[i] + ".pkl")
    FILE_PATH_PKL_Huber_opti_reg[i] = os.path.join(DATA_FOLDER_Huber_opti_reg, FILE_NAME_BASE_Huber_opti_reg[i] + ".pkl")

data_loaded_Tukey_fixed_reg = False
data_loaded_Huber_fixed_reg = False
data_loaded_Tukey_opti_reg = False
data_loaded_Huber_opti_reg = False
data_loaded = False
results_dict = None
results_dict_Tukey_fixed_reg = [None]*len(TAUS)
results_dict_Huber_fixed_reg = [None]*len(TAUS)
results_dict_Tukey_opti_reg = [None]*len(TAUS)
results_dict_Huber_opti_reg = [None]*len(TAUS)

if LOAD_FROM_PKL : #and os.path.exists(FILE_PATH_PKL_Tukey_opti_reg) and os.path.exists(FILE_PATH_PKL_Huber_opti_reg) and os.path.exists(FILE_PATH_PKL_Tukey_fixed_reg) and os.path.exists(FILE_PATH_PKL_Huber_fixed_reg):
    print(f"Chargement depuis {FILE_PATH_PKL_Tukey_fixed_reg}, {FILE_PATH_PKL_Huber_fixed_reg}, {FILE_PATH_PKL_Tukey_opti_reg} et {FILE_PATH_PKL_Huber_opti_reg}...")
    try:
        for i,TAU in enumerate(TAUS) :
            with open(FILE_PATH_PKL_Tukey_fixed_reg[i], "rb") as f:
                results_dict_Tukey_fixed_reg[i] = pickle.load(f)
            print(f"Chargement PKL Tukey fixed réussi pour tau={TAU}.")
            data_loaded_Tukey_fixed_reg = True
            with open(FILE_PATH_PKL_Huber_fixed_reg[i], "rb") as f:
                results_dict_Huber_fixed_reg[i] = pickle.load(f)
            print(f"Chargement PKL Huber fixed réussi pour tau={TAU}.")
            data_loaded_Huber_fixed_reg = True
            with open(FILE_PATH_PKL_Tukey_opti_reg[i], "rb") as f:
                results_dict_Tukey_opti_reg[i] = pickle.load(f)
            print(f"Chargement PKL Tukey opti réussi pour tau={TAU}.")
            data_loaded_Tukey_opti_reg = True
            with open(FILE_PATH_PKL_Huber_opti_reg[i], "rb") as f:
                results_dict_Huber_opti_reg[i] = pickle.load(f)
            print(f"Chargement PKL Huber opti réussi pour tau={TAU}.")
            data_loaded_Huber_opti_reg = True
        print("Tous les chargements PKL réussis.")
        data_loaded = True
    except Exception as e:
        print(f"Erreur lors du chargement PKL : {e}.")

if not data_loaded:
    print("Erreur : Impossible de charger les données.")
    exit()

# --- Extraction des données du dictionnaire ---
alphas = results_dict_Tukey_fixed_reg[0]['alphas']

ms_Tukey_fixed_reg = [results_dict_Tukey_fixed_reg[i]['ms'] for i in range(len(TAUS))]
qs_Tukey_fixed_reg = [results_dict_Tukey_fixed_reg[i]['qs'] for i in range(len(TAUS))]
Vs_Tukey_fixed_reg = [results_dict_Tukey_fixed_reg[i]['Vs'] for i in range(len(TAUS))]
ms_Huber_fixed_reg = [results_dict_Huber_fixed_reg[i]['ms'] for i in range(len(TAUS))]
qs_Huber_fixed_reg = [results_dict_Huber_fixed_reg[i]['qs'] for i in range(len(TAUS))]
Vs_Huber_fixed_reg = [results_dict_Huber_fixed_reg[i]['Vs'] for i in range(len(TAUS))]
gen_error_Tukey_fixed_reg = [results_dict_Tukey_fixed_reg[i]['gen_error'] for i in range(len(TAUS))]
gen_error_Huber_fixed_reg = [results_dict_Huber_fixed_reg[i]['gen_error'] for i in range(len(TAUS))]
rs_values_Tukey_fixed_reg = [results_dict_Tukey_fixed_reg[i].get('rs_values', np.full_like(alphas, np.nan)) for i in range(len(TAUS))] # Utilise .get pour compatibilité
rs_values_Huber_fixed_reg = [results_dict_Huber_fixed_reg[i].get('rs_values', np.full_like(alphas, np.nan)) for i in range(len(TAUS))] # Utilise .get pour compatibilité
estim_error_Tukey_fixed_reg = [1-2*ms_Tukey_fixed_reg[i]+qs_Tukey_fixed_reg[i] for i in range(len(TAUS))]
estim_error_Huber_fixed_reg = [1-2*ms_Huber_fixed_reg[i]+qs_Huber_fixed_reg[i] for i in range(len(TAUS))]


ms_Tukey_opti_reg = [ results_dict_Tukey_opti_reg[i]['ms'] for i in range(len(TAUS))]
qs_Tukey_opti_reg = [ results_dict_Tukey_opti_reg[i]['qs'] for i in range(len(TAUS))]
Vs_Tukey_opti_reg = [ results_dict_Tukey_opti_reg[i]['Vs'] for i in range(len(TAUS))]
ms_Huber_opti_reg = [ results_dict_Huber_opti_reg[i]['ms'] for i in range(len(TAUS))]
qs_Huber_opti_reg = [ results_dict_Huber_opti_reg[i]['qs'] for i in range(len(TAUS))]
Vs_Huber_opti_reg = [ results_dict_Huber_opti_reg[i]['Vs'] for i in range(len(TAUS))]
gen_error_Tukey_opti_reg = [ results_dict_Tukey_opti_reg[i]['gen_error'] for i in range(len(TAUS))]
gen_error_Huber_opti_reg = [ results_dict_Huber_opti_reg[i]['gen_error'] for i in range(len(TAUS))]
rs_values_Tukey_opti_reg = [ results_dict_Tukey_opti_reg[i].get('rs_values', np.full_like(alphas, np.nan)) for i in range(len(TAUS))] # Utilise .get pour compatibilité
rs_values_Huber_opti_reg = [ results_dict_Huber_opti_reg[i].get('rs_values', np.full_like(alphas, np.nan)) for i in range(len(TAUS))] # Utilise .get pour compatibilité
estim_error_Tukey_opti_reg = [1-2*ms_Tukey_opti_reg[i]+qs_Tukey_opti_reg[i] for i in range(len(TAUS))]
estim_error_Huber_opti_reg = [1-2*ms_Huber_opti_reg[i]+qs_Huber_opti_reg[i] for i in range(len(TAUS))]

# Création des données 1-m^2/q

# m2_q = np.full_like(alphas, np.nan)
# for i in range(len(alphas)):
#     if ms[i] > 0 and qs[i] > 0:
#         m2_q[i] = np.abs(1-(ms[i]**2) / qs[i])

# one_ms = np.full_like(alphas, np.nan)
# for i in range(len(alphas)):
#     one_ms[i] = 1-ms[i]

# one_qs = np.full_like(alphas, np.nan)
# for i in range(len(alphas)):
#     one_qs[i] = 1-qs[i]

# estim_err = np.full_like(alphas, np.nan)
# for i in range(len(alphas)):
#     estim_err[i]= 1+qs[i] - 2*ms[i]

# gen_error_in = (1-PERCENTAGE)*((1+PERCENTAGE*(BETA-1))**2 +qs-2*ms) #Excess : (1 - PERCENTAGE) * (delta_in + 1 - 2 * ms + qs) - (1 - PERCENTAGE) * (delta_in + 1 - 2 * ms_BO + qs_BO) avec - (1 + PERCENTAGE * (BETA -1))**2 * q_b = - 2 * ms_BO + qs_BO)
# gen_error_out = PERCENTAGE*((1+PERCENTAGE*(BETA-1))**2 + qs-2*BETA*ms) #Excess : PERCENTAGE * (delta_out + BETA**2 - 2 * ms + qs) - PERCENTAGE * (delta_out + BETA**2 - 2 * ms_BO + qs_BO) avec - (1 + PERCENTAGE * (BETA -1))**2 * q_b = - 2 * ms_BO + qs_BO)
# gen_error_actual = gen_error+(1 - PERCENTAGE) * PERCENTAGE**2 * (1 - BETA) ** 2 + PERCENTAGE * (
#         1 - PERCENTAGE
#     ) ** 2 * (BETA - 1) ** 2

# --- Préparation du Plot ---
#if os.path.exists(STYLE_FILE):
#    plt.style.use(STYLE_FILE)

fig_width_in, fig_height_in = set_size(FIG_WIDTH, fraction=0.9)
fig, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))

# --- Tracé des Données (Axe Y Principal) ---

#crée une matrice de len(TAUS) lignes et 2 colonnes pour les couleurs des courbes de sorte que chaque courbe ait une couleur différente pour tau

color_array = [f"C{i}" for i in range(len(TAUS))]  # Couleurs pour différencier les courbes
line_style_array = ['-', '-.', '--', ':']  # Styles de ligne pour différencier les courbes

for (i,TAU) in enumerate(TAUS) :
    # ax1.plot(alphas, gen_error_Tukey_fixed_reg[i], linestyle='-', markersize=3, color=color_array[i], label=f'$E_{{gen}}^e$ (Tukey, tau={TAU})')
    # ax1.plot(alphas, gen_error_Huber_fixed_reg[i], linestyle='-.', markersize=3, color=color_array[i], label=f'$E_{{gen}}^e$ (Huber, tau={TAU})')
    # ax1.plot(alphas, gen_error_Tukey_opti_reg[i], linestyle='--', markersize=3, color=color_array[i], label=f'$E_{{gen}}^e$ (Tukey optimal, tau={TAU})')
    # ax1.plot(alphas, gen_error_Huber_opti_reg[i], linestyle=':', markersize=3, color=color_array[i], label=f'$E_{{gen}}^e$ (Huber optimal, tau={TAU})')
    ax1.plot(alphas, estim_error_Tukey_fixed_reg[i], linestyle='-', markersize=3, color=color_array[i], label=f'$E_{{estim}}^e$ (Tukey, tau={TAU})')
    ax1.plot(alphas, estim_error_Huber_fixed_reg[i], linestyle='-.', markersize=3, color=color_array[i], label=f'$E_{{estim}}^e$ (Huber, tau={TAU})')
    ax1.plot(alphas, estim_error_Tukey_opti_reg[i], linestyle='--', markersize=3, color=color_array[i], label=f'$E_{{estim}}^e$ (Tukey optimal, tau={TAU})')
    ax1.plot(alphas, estim_error_Huber_opti_reg[i], linestyle=':', markersize=3, color=color_array[i], label=f'$E_{{estim}}^e$ (Huber optimal, tau={TAU})')

# ax1.plot(alphas, one_ms, marker='.', linestyle='-', markersize=3, color='tab:green', label='$1-m$')
# ax1.plot(alphas, one_qs, marker='.', linestyle='-', markersize=3, color='tab:red', label='$1-q$')
# ax1.plot(alphas, Vs, marker='.', linestyle='-', markersize=3, color='tab:purple', label='$V$')
# ax1.plot(alphas, m2_q, marker='.', linestyle='-', markersize=3, color='tab:cyan', label='$1-m^2/q$')
# ax1.plot(alphas[720:], estim_err[720:], marker='.', linestyle='-', markersize=3, color='tab:orange', label='$E_estim$')
# ax1.plot(alphas, gen_error_in, marker='.', linestyle='-', markersize=3, color='tab:gray', label='$E_{gen,in}$')
# ax1.plot(alphas, gen_error_out, marker='.', linestyle='-', markersize=3, color='tab:brown', label='$E_{gen,out}$')
#ax1.plot(alphas, gen_error_actual, marker='.', linestyle='-', markersize=3, color='tab:olive', label='$E_{gen}$')
#ax1.plot(alphas, gen_error - gen_error_in-gen_error_out, marker='.', linestyle='-', markersize=3, color='tab:blue', label='difference')

# Configuration de l'axe Y principal
ax1.set_xlabel(r'$\alpha = n/d$')
ax1.set_ylabel(r'Error')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax1.tick_params(axis='y', labelcolor='black')
# Définir les limites si nécessaire (souvent utile en log-log)
# ax1.set_ylim(1.75e-5, 2.5e-5)
# ax1.set_xlim(min(alphas[720:]), max(alphas[720:]))


# # --- Tracé de RS (Axe Y Secondaire) ---
# ax2 = ax1.twinx() # Crée un deuxième axe Y partageant le même axe X
# color_rs = 'tab:orange'
# # Tracer seulement les points RS valides
# valid_rs_indices = ~np.isnan(rs_values)
# if np.any(valid_rs_indices):
#     ax2.plot(alphas[valid_rs_indices], rs_values[valid_rs_indices], marker='x', linestyle='--', markersize=4, color=color_rs, label='RS Condition')
#     ax2.set_ylabel('RS Condition', color=color_rs)
#     ax2.tick_params(axis='y', labelcolor=color_rs)
#     ax2.set_ylim(0, max(1.1, np.nanmax(rs_values[valid_rs_indices])*1.1) if np.nanmax(rs_values[valid_rs_indices]) > 0 else 1.1) # Ajuste l'échelle RS
#     # Ligne horizontale à RS=1
#     ax2.axhline(1.0, color=color_rs, linestyle=':', linewidth=1.0, alpha=0.7)
# else:
#     print("Aucune valeur RS valide à tracer.")
#     ax2.set_yticks([]) # Cacher les ticks si pas de données

# --- Finalisation (Légendes, Titre, Sauvegarde) ---
# Combine les légendes des deux axes
lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
#ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
ax1.legend(lines1, labels1, loc='best', fontsize=8)

title = rf'Balayage $\alpha$ (Tukey VS Huber), $\lambda$ fixed={REG_PARAM:.2f}, $\epsilon$={PERCENTAGE:.1f}, $\beta$={BETA:.1f}, c={C_TUKEY:.3f}, delta_in={DELTA_IN}, delta_out={DELTA_OUT}'
plt.title(title, fontsize=10)

plt.tight_layout()

if SAVE_PLOT:
    plot_filename_base = f"Estim_error_AlphaSweep_Tukey_VS_Huber_lambda_{REG_PARAM:.1f}_DeltaInOut_{DELTA_IN}_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
    save_plot(fig, plot_filename_base, formats=IMG_FORMATS, directory=IMG_DIRECTORY)

plt.show()
print("Script de tracé terminé.")
