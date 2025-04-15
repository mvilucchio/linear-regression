import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import sys

try:
    from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (
        m_int_mod_Tukey_decorrelated_noise,
        q_int_mod_Tukey_decorrelated_noise,
        V_int_mod_Tukey_decorrelated_noise,
        BIG_NUMBER
    )
    from linear_regression.aux_functions.stability_functions import RS_int_mod_Tukey_decorrelated_noise
    print(f"BIG_NUMBER importé du module: {BIG_NUMBER}")
except Exception as e:
    print(f"Une autre erreur est survenue lors de l'importation : {e}")
    exit()


# --- Paramètres pour la visualisation ---
m_vis = 0.4
q_vis = 0.7
V_vis = 0.8
lambda_vis = 10.0
tau_vis = 2.0
BIG_NUMBER = 15

# Paramètres physiques
DELTA_IN = 0.8
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0.001

# Paramètres de la grille de visualisation
GRID_POINTS = 100
PLOT_RANGE_EXTENSION = 1.2 # Garder une petite extension pour voir les bords

# --- Création de la grille ---
xi_min, xi_max = -PLOT_RANGE_EXTENSION * BIG_NUMBER, PLOT_RANGE_EXTENSION * BIG_NUMBER
y_min, y_max = -PLOT_RANGE_EXTENSION * BIG_NUMBER, PLOT_RANGE_EXTENSION * BIG_NUMBER

xi_grid = np.linspace(xi_min, xi_max, GRID_POINTS)
y_grid = np.linspace(y_min, y_max, GRID_POINTS)
XI, Y = np.meshgrid(xi_grid, y_grid)

# --- Évaluation des intégrandes ---
args = (q_vis, m_vis, V_vis, DELTA_IN, DELTA_OUT, PERCENTAGE, BETA, tau_vis, C_TUKEY)
Z_m = np.zeros_like(XI)
Z_q = np.zeros_like(XI)
Z_V = np.zeros_like(XI)
Z_RS = np.zeros_like(XI)

print("Évaluation des intégrandes sur la grille...")
for i in range(GRID_POINTS):
    for j in range(GRID_POINTS):
        xi_val = XI[i, j]
        y_val = Y[i, j]
        try:
            Z_m[i, j] = m_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_m[i, j] = np.nan
        try:
            Z_q[i, j] = q_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_q[i, j] = np.nan
        try:
            Z_V[i, j] = V_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_V[i, j] = np.nan
        try:
            Z_RS[i, j] = RS_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_RS[i, j] = np.nan

Z_m = np.nan_to_num(Z_m)
Z_q = np.nan_to_num(Z_q)
Z_V = np.nan_to_num(Z_V)
Z_RS = np.nan_to_num(Z_RS)
print("Évaluation terminée.")

# --- Création des Plots 3D ---
print("Création des plots 3D...")
fig = plt.figure(figsize=(20, 7))

integrands = [Z_m, Z_q, Z_V, Z_RS]
titles = [
    r'Intégrande pour $\hat{m}$',
    r'Intégrande pour $\hat{q}$',
    r'Intégrande pour $\hat{V}$',
    r'Intégrande pour $RS$'
]
cmaps = [cm.viridis, cm.plasma, cm.cividis, cm.inferno]

for i, (Z, title, cmap) in enumerate(zip(integrands, titles, cmaps)):
    ax = fig.add_subplot(1, 4, i + 1, projection='3d')

    # Tracer la surface
    # rstride/cstride contrôlent la densité du maillage affiché (plus petit = plus dense)
    surf = ax.plot_surface(XI, Y, Z, cmap=cmap, linewidth=0, antialiased=False, rstride=1, cstride=1)

    # Ajouter une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    # Ajouter des contours projetés sur le plan z=min(Z) pour aider à la visualisation
    z_min = np.nanmin(Z) if np.any(np.isfinite(Z)) else 0
    try:
         if np.any(np.isfinite(Z)) and np.nanstd(Z) > 1e-9:
              cset = ax.contour(XI, Y, Z, zdir='z', offset=z_min, cmap=cmap, linewidths=0.5, alpha=0.5)
         else:
             print(f"Contours non tracés pour '{title}' (données non finies ou constantes)")
    except Exception as e:
        print(f"Erreur lors du tracé des contours projetés pour '{title}': {e}")


    ax.set_title(title)
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel('Valeur Intégrande')

    # Limiter les axes x et y aux bornes d'intégration pour se concentrer sur la zone pertinente
    ax.set_xlim(-BIG_NUMBER, BIG_NUMBER)
    ax.set_ylim(-BIG_NUMBER, BIG_NUMBER)

    # Ajuster l'angle de vue
    # ax.view_init(elev=20., azim=-65)

fig.suptitle(
    f'Visualisation 3D des Intégrandes (Tukey Mod.) pour m={m_vis:.2f}, q={q_vis:.2f}, V={V_vis:.2f}, τ={tau_vis:.2f}, c={C_TUKEY:.3f}',
    fontsize=14
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print("Plots créés. Affichage...")
plt.show()
