import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import warnings

# --- Helpers de Plotting (du template) ---
IMG_DIRECTORY = "./imgs/alpha_sweeps_tukey"  # Répertoire spécifique


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
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


# --- Configuration du Plot ---
LOAD_FROM_PKL = True  # Privilégier PKL
SAVE_PLOT = True
STYLE_FILE = "./plotting/latex_ready.mplstyle"  # Ajustez si nécessaire
FIG_WIDTH = 469.75502  # Largeur LaTeX standard
IMG_FORMATS = ["pdf", "png"]

# --- Paramètres de la Simulation (pour retrouver le fichier) ---
# Doivent correspondre à ceux du script de calcul !
NOM_LOSS = "Tukey_mod_xigamma_c0"
ALPHA_MIN = 0.5
ALPHA_MAX = 1000
N_ALPHA_PTS = 100
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0
REG_PARAM = 2.0
TAU = 1.0

# --- Chargement des Données ---
DATA_FOLDER_SE = "./data/alpha_sweeps_tukey"  # Doit correspondre
# FILE_NAME_BASE = f"alpha_sweep_{NOM_LOSS}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_cin_{DELTA_IN}_cout_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
FILE_NAME_BASE = f"alpha_sweep_{NOM_LOSS}_alpha_min_{ALPHA_MIN:.1f}_alpha_max_{ALPHA_MAX:.1f}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_delta_in_{DELTA_IN}_delta_out_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
FILE_PATH_PKL = os.path.join(DATA_FOLDER_SE, FILE_NAME_BASE + ".pkl")
FILE_PATH_CSV = os.path.join(
    DATA_FOLDER_SE, FILE_NAME_BASE + ".csv")  # Fallback

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
        print(
            f"Erreur lors du chargement PKL : {e}. Tentative avec CSV si disponible.")

print(f"Chargement depuis {FILE_PATH_CSV}...")

# Fallback CSV (lecture basique, à adapter si besoin)
if not data_loaded and os.path.exists(FILE_PATH_CSV):
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
            "CALCULATED_RS": True  # On suppose qu'elle est calculée si la colonne existe
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

# Création des données

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
    estim_err[i] = 1+qs[i] - 2*ms[i]

angle = np.arccos(np.clip(ms / np.sqrt(qs), -1, 1)) / np.pi

# --- Préparation du Plot ---
# if os.path.exists(STYLE_FILE):
#    plt.style.use(STYLE_FILE)

fig_width_in, fig_height_in = set_size(FIG_WIDTH, fraction=0.9)
fig, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))

# --- Tracé des Données (Axe Y Principal) ---
ax1.plot(alphas, gen_error, marker='.', linestyle='-',
         markersize=3, color='tab:blue', label='$E_{gen}$')
ax1.plot(alphas, estim_err, marker='.', linestyle='-',
         markersize=3, color='tab:orange', label='$E_estim$')
ax1.plot(alphas, angle, marker='.', linestyle='-',
         markersize=3, color='tab:green', label=r'$\theta$')

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

# --- ERM Donnees ---

reps = 10
ALPHA_MIN = .5
ALPHA_MAX = 300
N_ALPHA_PTS = 20
d = 500
c_tukey_ERM = 0.0

DATA_FOLDER_ERM = "./data/mod_Tukey_decorrelated_noise"  # Répertoire spécifique
FILE_NAME_ERM = f"ERM_mod_Tukey_{TAU:.2f}_{c_tukey_ERM:.2e}_alpha_sweep_{ALPHA_MIN:.2f}_{ALPHA_MAX:.3f}_{N_ALPHA_PTS:d}_reps_{reps:d}_d_{d:d}_decorrelated_noise_{DELTA_IN:.2f}_{DELTA_OUT:.2f}_{PERCENTAGE:.2f}_{BETA:.2f}.pkl"

data = np.loadtxt(
    os.path.join(DATA_FOLDER_ERM, FILE_NAME_ERM),
    delimiter=",",
    skiprows=1,
)

alphas = data[:, 0]
ms_means, ms_stds = data[:, 1], data[:, 2]
qs_means, q_stds = data[:, 3], data[:, 4]
estim_errors_means, estim_errors_stds = data[:, 5], data[:, 6]
gen_errors_means, gen_errors_stds = data[:, 7], data[:, 8]

ax1.errorbar(
    alphas,
    gen_errors_means,
    yerr=gen_errors_stds,
    label=r"$E_{gen}$ (ERM)",
    fmt="o-",
    color='tab:blue',
    linestyle='None',
)

ax1.errorbar(
    alphas,
    estim_errors_means,
    yerr=estim_errors_stds,
    label=r"$E_{estim}$ (ERM)",
    fmt="o-",
    color='tab:orange',
    linestyle='None',
)


# --- Finalisation (Légendes, Titre, Sauvegarde) ---

ax1.legend(loc='best', fontsize=8)

title = rf'Balayage $\alpha$ ({NOM_LOSS}, $\lambda$={REG_PARAM:.2f}, $\tau$={TAU:.2f}, $\epsilon$={PERCENTAGE:.1f}, $\beta$={BETA:.1f}, c={C_TUKEY:.3f})'
plt.title(title, fontsize=10)

plt.tight_layout()

if SAVE_PLOT:
    plot_filename_base = f"AlphaSweep_{NOM_LOSS}_lambda_{REG_PARAM:.1f}_tau_{TAU:.1f}_DeltaInOut_{DELTA_IN}_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
    save_plot(fig, plot_filename_base,
              formats=IMG_FORMATS, directory=IMG_DIRECTORY)

print("Script de tracé terminé.")
plt.show()

