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
        m_star_int_xigamma_Tukey,
        q_star_int_xigamma_Tukey,
        V_star_int_xigamma_Tukey,
        BIG_NUMBER,
    )
    from linear_regression.aux_functions.stability_functions import RS_int_mod_Tukey_decorrelated_noise
except Exception as e:
    print(f"Une autre erreur est survenue lors de l'importation : {e}")
    exit()


# --- Paramètres pour la visualisation ---
alpha, m_vis, q_vis, V_vis = 1.00000000e+03,9.91467985e-01,9.83152819e-01,2.01229546e-03
#m_vis = 0.4
#q_vis = 0.7
#V_vis = 0.8
lambda_vis = 2.0
tau_vis = 1.0
BIG_NUMBER = 15

# Paramètres physiques
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0.001

# Paramètres de la grille de visualisation
GRID_POINTS = 1+2*50
PLOT_RANGE_EXTENSION = 1.2 # Garder une petite extension pour voir les bords

# --- Création de la grille ---
xi_min, xi_max = -PLOT_RANGE_EXTENSION * BIG_NUMBER, PLOT_RANGE_EXTENSION * BIG_NUMBER
y_min, y_max = -PLOT_RANGE_EXTENSION * BIG_NUMBER, PLOT_RANGE_EXTENSION * BIG_NUMBER
dy = (y_max - y_min) / (GRID_POINTS - 1)
tolerance = dy/2.0 # Tolérance pour la distance entre la droite et le point (xi, y)
line_delta = tau_vis*2 # value of |y_sqrt(q)*xi|

X_grid = np.linspace(xi_min, xi_max, GRID_POINTS)
Y_grid = np.linspace(y_min, y_max, GRID_POINTS)
XI, Y = np.meshgrid(X_grid, Y_grid)

# Variables pour stocker les max aux frontières
max_abs_m_boundary = 0.0
max_abs_q_boundary = 0.0
max_abs_V_boundary = 0.0

# --- Évaluation des intégrandes ---
args = (q_vis, m_vis, V_vis, DELTA_IN, DELTA_OUT, PERCENTAGE, BETA, tau_vis, C_TUKEY)
Z_m = np.zeros_like(XI) # Remplacer par u_axis si nécessaire
Z_q = np.zeros_like(XI) # Remplacer par u_axis si nécessaire
Z_V = np.zeros_like(XI) # Remplacer par u_axis si nécessaire
Z_RS = np.zeros_like(XI) # Remplacer par u_axis si nécessaire

max_abs = [0.0, 0.0, 0.0, 0.0]

print("Évaluation des intégrandes sur la grille...")
for i in range(GRID_POINTS):
    for j in range(GRID_POINTS):
        xi_val = XI[i, j]
        y_val = Y[i, j]
        try:
            Z_m[i, j] = m_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_m[i, j] = np.nan
        if not np.isnan(Z_m[i, j]):
            max_abs[0] = max(max_abs[0], abs(Z_m[i, j]))
        try:
            Z_q[i, j] = q_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_q[i, j] = np.nan
        if not np.isnan(Z_m[i, j]):
            max_abs[1] = max(max_abs[1], abs(Z_q[i, j]))
        try:
            Z_V[i, j] = -V_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_V[i, j] = np.nan
        if not np.isnan(Z_m[i, j]):
            max_abs[2] = max(max_abs[2], abs(Z_V[i, j]))
        try:
            Z_RS[i, j] = RS_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        except Exception: Z_RS[i, j] = np.nan
        if not np.isnan(Z_m[i, j]):
            max_abs[3] = max(max_abs[3], abs(Z_RS[i, j]))

        dist_plus = abs(y_val - (np.sqrt(q_vis) * xi_val + line_delta))
        dist_minus = abs(y_val - (np.sqrt(q_vis)* xi_val - line_delta))
            
        if dist_plus < tolerance or dist_minus < tolerance:
            if not np.isnan(Z_m[i,j]):
                max_abs_m_boundary = max(max_abs_m_boundary, abs(Z_m[i, j]))
            if not np.isnan(Z_q[i,j]):
                max_abs_q_boundary = max(max_abs_q_boundary, abs(Z_q[i, j]))
            if not np.isnan(Z_V[i,j]):
                max_abs_V_boundary = max(max_abs_V_boundary, abs(Z_V[i, j])) 

Z_m = np.nan_to_num(Z_m)
Z_q = np.nan_to_num(Z_q)
Z_V = np.nan_to_num(Z_V)
Z_RS = np.nan_to_num(Z_RS)
print("Évaluation terminée.")

# --- Calcul de l'écart maximal de symétrie ---
max_diff_m = 0.0
max_diff_q = 0.0
max_diff_V = 0.0
max_diff_RS = 0.0

print("Calcul de l'écart maximal de symétrie...")

# Puisque GRID_POINTS est impair, l'indice du centre est GRID_POINTS // 2
# L'indice symétrique de i est GRID_POINTS - 1 - i
for i in range(GRID_POINTS):
    for j in range(GRID_POINTS):
        # Indices du point symétrique (-xi, -y)
        i_sym = GRID_POINTS - 1 - i
        j_sym = GRID_POINTS - 1 - j
        
        # Calculer la différence absolue
        diff_m = abs(Z_m[i, j] - Z_m[i_sym, j_sym])
        diff_q = abs(Z_q[i, j] - Z_q[i_sym, j_sym])
        diff_V = abs(Z_V[i, j] - Z_V[i_sym, j_sym])
        diff_RS = abs(Z_RS[i, j] - Z_RS[i_sym, j_sym])
        
        # Mettre à jour les maxima
        max_diff_m = max(max_diff_m, diff_m)
        max_diff_q = max(max_diff_q, diff_q)
        max_diff_V = max(max_diff_V, diff_V)
        max_diff_RS = max(max_diff_RS, diff_RS)

print("Calcul de symétrie terminé.")

# --- Affichage des résultats des calculs ---
print("\n--- Résultats des Analyses ---")
print(f"Valeur absolue maximale relative de l'intégrande m proche de y = sqrt(q)*xi +/- {line_delta} : {max_abs_m_boundary/max_abs[0]:.4e}")
print(f"Valeur absolue maximale relative de l'intégrande q proche de y = sqrt(q)*xi +/- {line_delta} : {max_abs_q_boundary/max_abs[1]:.4e}")
print(f"Valeur absolue maximale relative de l'intégrande V proche de y = sqrt(q)*xi +/- {line_delta} : {max_abs_V_boundary/max_abs[2]:.4e}")

print(f"\nÉcart maximal relatif entre Z(xi, y) et Z(-xi, -y) pour m : {max_diff_m/max_abs[0]:.4e}")
print(f"Écart maximal relatif entre Z(xi, y) et Z(-xi, -y) pour q : {max_diff_q/max_abs[1]:.4e}")
print(f"Écart maximal relatif entre Z(xi, y) et Z(-xi, -y) pour V : {max_diff_V/max_abs[2]:.4e}")
print(f"Écart maximal relatif entre Z(xi, y) et Z(-xi, -y) pour RS: {max_diff_RS/max_abs[3]:.4e}")
print("-----------------------------\n")

# --- Création des Plots 3D ---
print("Création des plots 3D...")
fig = plt.figure(figsize=(22, 8))

integrands = [Z_m, Z_q, Z_V, Z_RS]
titles = [
    r'Intégrande pour $\hat{m}$',
    r'Intégrande pour $\hat{q}$',
    r'Intégrande pour $\hat{V}$',
    r'Intégrande pour $RS$'
]
cmaps = [cm.viridis, cm.plasma, cm.cividis, cm.inferno]

xi_line = np.linspace(xi_min, xi_max, 200)
y_line_plus = np.sqrt(q_vis) * xi_line + tau_vis
y_line_minus = np.sqrt(q_vis) * xi_line - tau_vis
mask_plus = (y_line_plus >= -BIG_NUMBER) & (y_line_plus <= BIG_NUMBER)
mask_minus = (y_line_minus >= -BIG_NUMBER) & (y_line_minus <= BIG_NUMBER)

for i, (Z, title, cmap) in enumerate(zip(integrands, titles, cmaps)):
    ax = fig.add_subplot(1, 4, i + 1, projection='3d')

    # Tracer la surface
    # rstride/cstride contrôlent la densité du maillage affiché (plus petit = plus dense)
    surf = ax.plot_surface(XI, Y, Z, cmap=cmap, linewidth=0, antialiased=False, rstride=1, cstride=1) # changer les axes

    # Ajouter une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    # Ajouter des contours projetés sur le plan z=min(Z) pour aider à la visualisation
    z_max = np.nanmax(Z) if np.any(np.isfinite(Z)) else 0
    z_min = np.nanmin(Z) if np.any(np.isfinite(Z)) else 0
    try:
        if np.any(np.isfinite(Z)) and np.nanstd(Z) > 1e-9:
            cset = ax.contour(XI, Y, Z, zdir='z', offset=z_max, extend3d=True, cmap=cmap, linewidths=0.5)
        else:
            print(f"Contours non tracés pour '{title}' (données non finies ou constantes)")
    except Exception as e:
        print(f"Erreur lors du tracé des contours projetés pour '{title}': {e}")

    #afficher des plans verticaux pour des valeurs de z de z_min à z_max)
    ax.plot(xi_line[mask_plus], y_line_plus[mask_plus], z_max/3 , color='black', linewidth=1.5, label=r'$y = \sqrt{q} \xi + \tau$')
    ax.plot(xi_line[mask_minus], y_line_minus[mask_minus], z_max/3, color='black', linewidth=1.5, label=r'$y = \sqrt{q} \xi - \tau$')
        
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
    f'Visualisation 3D des Intégrandes inliers (Tukey Mod.) pour τ={tau_vis:.2f}, c={C_TUKEY:.3f}, λ={lambda_vis:.2f}, δ_in={DELTA_IN:.2f}, δ_out={DELTA_OUT:.2f}, β={BETA:.2f}, alpha={alpha:.2f}',
    fontsize=11
) #changer in / out
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print("Plots créés. Affichage...")
plt.show()
