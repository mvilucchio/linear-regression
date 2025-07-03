import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.lines as mlines

from linear_regression.aux_functions.moreau_proximals import DƔ_proximal_L2

# --- Helpers de Plotting --- Matéo
IMG_DIRECTORY = "./imgs/phase_diagrams"

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
PLOT_M = True
PLOT_Q = True
PLOT_V = True
PLOT_RS = True
PLOT_TIME = True
HEATMAP_VARIABLE = None # Choix: 'V', 'RS', 'TIME', 'M', 'Q', None

LOAD_FROM_PKL = True
SAVE_PLOT = True
STYLE_FILE = "./plotting/latex_ready.mplstyle"
FIG_WIDTH = 469.75502 # Largeur standard LaTeX
IMG_FORMATS = ["pdf", "png"]

# --- Paramètres de la Simulation (pour retrouver le fichier) ---
NOM_LOSS = "Tukey_mod_xigamma_c0"
ALPHA = 10.0
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0.0
# Options d'intégration (utilisées seulement pour le calcul de RS ici)
INTEGRATION_BOUND = 7.0
INTEGRATION_EPSABS = 1e-8
INTEGRATION_EPSREL = 1e-5

# --- Chargement des Données ---
DATA_FOLDER = "./data/phase_diagrams_tukey"
FILE_NAME_BASE = f"phase_diagram_{NOM_LOSS}_alpha_{ALPHA:.1f}_deltas_{DELTA_IN}_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}"
FILE_PATH_PKL = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".pkl")
FILE_PATH_CSV = os.path.join(DATA_FOLDER, FILE_NAME_BASE + ".csv")

data_loaded = False
results_dict = None
MQV_results = None
reg_params = None
taus = None
time_results = None
USE_REG_LOGSPACE = False
USE_TAU_LOGSPACE = False
N_REG_PARAM_PTS = 0
N_TAU_PTS = 0

if LOAD_FROM_PKL and os.path.exists(FILE_PATH_PKL):
    print(f"Chargement depuis {FILE_PATH_PKL}...")
    try:
        with open(FILE_PATH_PKL, "rb") as f:
            results_dict = pickle.load(f)
        reg_params = results_dict["reg_params"]
        taus = results_dict["taus"]
        MQV_results = results_dict["MQV_results"]
        time_results = results_dict.get("time_results", None)
        USE_REG_LOGSPACE = results_dict.get("use_reg_logspace", True)
        USE_TAU_LOGSPACE = results_dict.get("use_tau_logspace", True)
        if time_results is None: PLOT_TIME = False
        N_REG_PARAM_PTS = len(reg_params)
        N_TAU_PTS = len(taus)
        assert MQV_results.shape == (N_REG_PARAM_PTS, N_TAU_PTS, 6), "Shape mismatch PKL"
        print("Chargement PKL réussi.")
        data_loaded = True
    except Exception as e:
        print(f"Erreur lors du chargement PKL : {e}. Tentative avec CSV si disponible.")

if not data_loaded and os.path.exists(FILE_PATH_CSV):
    print(f"Chargement et reconstruction depuis {FILE_PATH_CSV}...")
    try:
        df = pd.read_csv(FILE_PATH_CSV)
        if df.empty: raise ValueError("CSV vide.")
        reg_params = sorted(df['reg_param'].unique())
        taus = sorted(df['tau'].unique())
        N_REG_PARAM_PTS = len(reg_params)
        N_TAU_PTS = len(taus)
        MQV_results = np.full((N_REG_PARAM_PTS, N_TAU_PTS, 6), np.nan)
        time_results = np.full((N_REG_PARAM_PTS, N_TAU_PTS), np.nan)
        reg_map = {val: i for i, val in enumerate(reg_params)}
        tau_map = {val: j for j, val in enumerate(taus)}
        for index, row in df.iterrows():
             i = reg_map.get(row['reg_param'])
             j = tau_map.get(row['tau'])
             if i is not None and j is not None:
                 MQV_results[i, j, :] = row[2:8].values
                 if 'time_sec' in row: time_results[i, j] = row['time_sec']
        print("Chargement et reconstruction CSV réussis.")
        data_loaded = True
    except Exception as e:
        print(f"Erreur lors du chargement/traitement CSV : {e}")

if not data_loaded:
    print("Erreur : Impossible de charger les données.")
    exit()

# --- Calcul de la Condition RS (si demandé) ---
if PLOT_RS:
    print("Calcul de la condition RS...")
    RS_values = np.full((N_REG_PARAM_PTS, N_TAU_PTS), np.nan)
    for i in tqdm(range(N_REG_PARAM_PTS), desc="Calcul RS (RegParam)"):
        for j in range(N_TAU_PTS):
            m, q, V, m_hat, q_hat, V_hat = MQV_results[i, j, :]
            current_reg_param = reg_params[i]
            current_tau = taus[j]
            if np.all(np.isfinite([m, q, V, V_hat])):
                try:
                    integral_rs = 0 # RS_E2_xigamma_mod_Tukey_decorrelated_noise(
                    #     m, q, V, DELTA_IN, DELTA_OUT, PERCENTAGE, BETA,
                    #     current_tau, C_TUKEY,
                    #     INTEGRATION_BOUND, INTEGRATION_EPSABS, INTEGRATION_EPSREL
                    # ) This is outdated
                    dprox_val_sq = DƔ_proximal_L2(0.0, V_hat, current_reg_param)**2
                    if not np.isnan(integral_rs):
                        RS_values[i, j] = ALPHA * dprox_val_sq * integral_rs
                except Exception as e:
                    RS_values[i, j] = np.nan
    print("Calcul RS terminé.")
else:
    RS_values = None

# --- Préparation et Création du Plot ---
if os.path.exists(STYLE_FILE): plt.style.use(STYLE_FILE)

fig_width_in, fig_height_in = set_size(FIG_WIDTH, fraction=0.9) # Ajuster fraction si besoin
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

T, R = np.meshgrid(taus, reg_params)
cmap_V = plt.cm.viridis
cmap_RS = plt.cm.plasma
cmap_Time = plt.cm.magma

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

# --- Tracé des Données ---
handles_for_legend = [] # Pour stocker les proxys de légende

# 1. Heatmap (si demandé)
plotted_heatmap = False
if HEATMAP_VARIABLE == 'V' and PLOT_V:
    data_to_plot = MQV_results[:, :, 2]
    cmap_heatmap = plt.cm.viridis
    label_heatmap = 'V'
elif HEATMAP_VARIABLE == 'RS' and PLOT_RS and RS_values is not None:
    data_to_plot = RS_values
    cmap_heatmap = plt.cm.plasma
    label_heatmap = 'RS Condition'
elif HEATMAP_VARIABLE == 'TIME' and PLOT_TIME and time_results is not None:
    data_to_plot = time_results
    cmap_heatmap = plt.cm.magma
    label_heatmap = 'Temps (s)'
elif HEATMAP_VARIABLE == 'M' and PLOT_M:
    data_to_plot = MQV_results[:, :, 0]
    cmap_heatmap = plt.cm.viridis
    label_heatmap = 'm'
elif HEATMAP_VARIABLE == 'Q' and PLOT_Q:
    data_to_plot = MQV_results[:, :, 1]
    cmap_heatmap = plt.cm.viridis
    label_heatmap = 'q'
else:
    data_to_plot = None

if data_to_plot is not None:
    finite_data = data_to_plot[np.isfinite(data_to_plot)]
    if finite_data.size > 0:
        vmin_h = np.nanmin(finite_data)
        vmax_h = np.nanpercentile(finite_data, 99.5) # Plafonner
        pcm = ax.pcolormesh(T, R, data_to_plot, cmap=cmap_heatmap, shading='gouraud', vmin=vmin_h, vmax=vmax_h, rasterized=True)
        cbar = fig.colorbar(pcm, ax=ax, label=label_heatmap, pad=0.02)
        plotted_heatmap = True
    else:
        print(f"Aucune donnée valide pour le heatmap de {HEATMAP_VARIABLE}")

# 2. Contours
contour_colors = {
    'V': 'red',
    'RS': 'blue',
    'TIME': 'lime',
    'M': 'grey',
    'Q': 'cyan'
}
contour_linestyles = {
    'V': '--',
    'RS': '-',
    'TIME': ':',
    'M': '-.',
    'Q': '-'
}
contour_levels = { # Niveaux par défaut, peuvent être ajustés
    'V': 6,
    'RS': [0.5, 0.75, 1.0,1.25,1.5],
    'TIME': 5,
    'M': 5,
    'Q': 5
}

# Fonction pour tracer les contours et créer un proxy pour la légende
def plot_contour_with_legend(ax, data, var_name, levels, color, linestyle, linewidth=1.0):
    handle = None
    finite_data = data[np.isfinite(data)]
    if finite_data.size > 0:
        try:
            contour = ax.contour(T, R, data, levels=levels, colors=color, linestyles=linestyle, linewidths=linewidth)
            ax.clabel(contour, inline=True, fontsize=7, fmt=f'{var_name}=%.1f')
            # Créer un proxy pour la légende
            handle = mlines.Line2D([], [], color=color, linestyle=linestyle, linewidth=linewidth, label=f'{var_name} Contours')
        except Exception as e:
            print(f"Erreur contour {var_name}: {e}")
    else:
        print(f"Aucune donnée valide pour les contours de {var_name}")
    return handle

if PLOT_V and HEATMAP_VARIABLE != 'V':
    handle = plot_contour_with_legend(ax, MQV_results[:, :, 2], 'V', contour_levels['V'], contour_colors['V'], contour_linestyles['V'], linewidth=0.8)
    if handle: handles_for_legend.append(handle)

if PLOT_M and HEATMAP_VARIABLE != 'M':
    handle = plot_contour_with_legend(ax, MQV_results[:, :, 0], 'm', contour_levels['M'], contour_colors['M'], contour_linestyles['M'])
    if handle: handles_for_legend.append(handle)

if PLOT_Q and HEATMAP_VARIABLE != 'Q':
    handle = plot_contour_with_legend(ax, MQV_results[:, :, 1], 'q', contour_levels['Q'], contour_colors['Q'], contour_linestyles['Q'])
    if handle: handles_for_legend.append(handle)

if PLOT_RS and RS_values is not None and HEATMAP_VARIABLE != 'RS':
    handle = plot_contour_with_legend(ax, RS_values, 'RS', contour_levels['RS'], contour_colors['RS'], contour_linestyles['RS'], linewidth=1.5)
    if handle: handles_for_legend.append(handle)

if PLOT_TIME and time_results is not None and HEATMAP_VARIABLE != 'TIME':
    handle = plot_contour_with_legend(ax, time_results, 'Time', contour_levels['TIME'], contour_colors['TIME'], contour_linestyles['TIME'])
    if handle: handles_for_legend.append(handle)

# --- Finalisation du Plot ---
if USE_TAU_LOGSPACE: ax.set_xscale('log')
if USE_REG_LOGSPACE: ax.set_yscale('log')

ax.set_xlabel(r'$\tau$ (Tukey Threshold)')
ax.set_ylabel(r'$\lambda$ (L2 Regularization)')
title = rf'Phase Diagram ({NOM_LOSS}, $\alpha$={ALPHA:.1f}, $\Delta_{{in}}$={DELTA_IN:.1f}, $\Delta_{{out}}$={DELTA_OUT:.1f}, $\epsilon$={PERCENTAGE:.1f}, $\beta$={BETA:.1f}, c={C_TUKEY:.3f})'
ax.set_title(title, fontsize=9)
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)

# Créer la légende à partir des proxys stockés
if handles_for_legend:
    ax.legend(handles=handles_for_legend, loc='best', fontsize=8)

plt.tight_layout(pad=0.5)

# --- Sauvegarde ---
if SAVE_PLOT:
    plot_filename_base = f"PhaseDiag_{NOM_LOSS}_alpha_{ALPHA:.1f}_DeltaInOut_{DELTA_IN}_{DELTA_OUT}_eps_{PERCENTAGE}_beta_{BETA}_c_{C_TUKEY}_heatmap-{HEATMAP_VARIABLE or 'None'}"
    save_plot(fig, plot_filename_base, formats=IMG_FORMATS, directory=IMG_DIRECTORY)

plt.show()
print("Script de tracé terminé.")
