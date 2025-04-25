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
NOM_LOSS = "Tukey_mod_xigamma_c0"
ALPHA_MIN = 1000
ALPHA_MAX = 1000000
N_ALPHA_PTS = 1000
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0
REG_PARAM = 2.0
TAU = 1.0

# --- Chargement des Données ---
DATA_FOLDER = "./data/alpha_sweeps_tukey" # Doit correspondre
#FILE_NAME_BASE = f"alpha_sweep_{NOM_LOSS}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_cin_{DELTA_IN}_cout_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
FILE_NAME_BASE = f"alpha_sweep_{NOM_LOSS}_alpha_min_{ALPHA_MIN:.1f}_alpha_max_{ALPHA_MAX:.1f}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_delta_in_{DELTA_IN}_delta_out_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
FILE_PATH_PKL = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".pkl")
FILE_PATH_CSV = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".csv") # Fallback

data_loaded = False
results_dict = None

if LOAD_FROM_PKL and os.path.exists(FILE_PATH_PKL):
    print(f"Chargement depuis {FILE_PATH_PKL}...")
    try:
        with open(FILE_PATH_PKL, "rb") as f:
            results_dict = pickle.load(f)
        print("Chargement PKL réussi.")
        data_loaded = True
    except Exception as e:
        print(f"Erreur lors du chargement PKL : {e}. Tentative avec CSV si disponible.")

# Fallback CSV (lecture basique, à adapter si besoin)
if not data_loaded and os.path.exists(FILE_PATH_CSV):
     print(f"Chargement et reconstruction depuis {FILE_PATH_CSV}...")
     try:
         # Attention: cette lecture suppose que le CSV contient les colonnes dans cet ordre précis
         # et qu'il n'y a pas de lignes manquantes. La reconstruction est fragile.
         data_csv = np.genfromtxt(FILE_PATH_CSV, delimiter=',', skip_header=1)
         results_dict = {
             "alphas": data_csv[:, 0],
             "ms": data_csv[:, 1],
             "qs": data_csv[:, 2],
             "Vs": data_csv[:, 3],
             "m_hats": data_csv[:, 4],
             "q_hats": data_csv[:, 5],
             "V_hats": data_csv[:, 6],
             "gen_error": data_csv[:, 7],
             "rs_values": data_csv[:, 8],
             "times_sec": data_csv[:, 9],
             # Les métadonnées ne sont pas dans le CSV, on les remet ici
             "ALPHA_MIN": ALPHA_MIN, "ALPHA_MAX": ALPHA_MAX, "N_ALPHA_PTS": len(data_csv[:, 0]),
             "DELTA_IN": DELTA_IN, "DELTA_OUT": DELTA_OUT, "PERCENTAGE": PERCENTAGE,
             "BETA": BETA, "C_TUKEY": C_TUKEY, "REG_PARAM": REG_PARAM, "TAU": TAU,
             "CALCULATED_RS": True # On suppose qu'elle est calculée si la colonne existe
         }
         print("Chargement et reconstruction CSV réussis.")
         data_loaded = True
     except Exception as e:
        print(f"Erreur lors du chargement/traitement CSV : {e}")


if not data_loaded:
    print("Erreur : Impossible de charger les données.")
    exit()

# --- Extraction des données du dictionnaire ---
alphas = results_dict['alphas']
ms = results_dict['ms']
qs = results_dict['qs']
Vs = results_dict['Vs']
gen_error = results_dict['gen_error']
rs_values = results_dict.get('rs_values', np.full_like(alphas, np.nan)) # Utilise .get pour compatibilité

# Création des données 1-m^2/q

m2_q = np.full_like(alphas, np.nan)
for i in range(len(alphas)):
    if ms[i] > 0 and qs[i] > 0:
        m2_q[i] = np.abs(1-(ms[i]**2) / qs[i])

one_ms = np.full_like(alphas, np.nan)
for i in range(len(alphas)):
    one_ms[i] = 1-ms[i]

one_qs = np.full_like(alphas, np.nan)
for i in range(len(alphas)):
    one_qs[i] = 1-qs[i]

estim_err = np.full_like(alphas, np.nan)
for i in range(len(alphas)):
    estim_err[i]= 1+qs[i] - 2*ms[i]

# --- Préparation du Plot ---
#if os.path.exists(STYLE_FILE):
#    plt.style.use(STYLE_FILE)

fig_width_in, fig_height_in = set_size(FIG_WIDTH, fraction=0.9)
fig, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))

# --- Tracé des Données (Axe Y Principal) ---
ax1.plot(alphas, gen_error, marker='.', linestyle='-', markersize=3, color='tab:blue', label='$E_{gen}$')
ax1.plot(alphas, one_ms, marker='.', linestyle='-', markersize=3, color='tab:green', label='$1-m$')
ax1.plot(alphas, one_qs, marker='.', linestyle='-', markersize=3, color='tab:red', label='$1-q$')
ax1.plot(alphas, Vs, marker='.', linestyle='-', markersize=3, color='tab:purple', label='$V$')
ax1.plot(alphas, m2_q, marker='.', linestyle='-', markersize=3, color='tab:cyan', label='$1-m^2/q$')
ax1.plot(alphas, estim_err, marker='.', linestyle='-', markersize=3, color='tab:orange', label='$E_estim$')

# Configuration de l'axe Y principal
ax1.set_xlabel(r'$\alpha = n/d$')
ax1.set_ylabel(r'Valeurs des Overlaps / $E_{gen}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax1.tick_params(axis='y', labelcolor='black')
# Définir les limites si nécessaire (souvent utile en log-log)
# ax1.set_ylim(1e-3, 1e2)
ax1.set_xlim(min(alphas), max(alphas))


# --- Tracé de RS (Axe Y Secondaire) ---
ax2 = ax1.twinx() # Crée un deuxième axe Y partageant le même axe X
color_rs = 'tab:orange'
# Tracer seulement les points RS valides
valid_rs_indices = ~np.isnan(rs_values)
if np.any(valid_rs_indices):
    ax2.plot(alphas[valid_rs_indices], rs_values[valid_rs_indices], marker='x', linestyle='--', markersize=4, color=color_rs, label='RS Condition')
    ax2.set_ylabel('RS Condition', color=color_rs)
    ax2.tick_params(axis='y', labelcolor=color_rs)
    ax2.set_ylim(0, max(1.1, np.nanmax(rs_values[valid_rs_indices])*1.1) if np.nanmax(rs_values[valid_rs_indices]) > 0 else 1.1) # Ajuste l'échelle RS
    # Ligne horizontale à RS=1
    ax2.axhline(1.0, color=color_rs, linestyle=':', linewidth=1.0, alpha=0.7)
else:
    print("Aucune valeur RS valide à tracer.")
    ax2.set_yticks([]) # Cacher les ticks si pas de données

# --- Finalisation (Légendes, Titre, Sauvegarde) ---
# Combine les légendes des deux axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

title = rf'Balayage $\alpha$ ({NOM_LOSS}, $\lambda$={REG_PARAM:.2f}, $\tau$={TAU:.2f}, $\epsilon$={PERCENTAGE:.1f}, $\beta$={BETA:.1f}, c={C_TUKEY:.3f})'
plt.title(title, fontsize=10)

plt.tight_layout()

if SAVE_PLOT:
    plot_filename_base = f"AlphaSweep_{NOM_LOSS}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_DeltaInOut_{DELTA_IN}_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
    save_plot(fig, plot_filename_base, formats=IMG_FORMATS, directory=IMG_DIRECTORY)

plt.show()
print("Script de tracé terminé.")
