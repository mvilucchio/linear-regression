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
        m_star_int_uv_Tukey,
        q_star_int_uv_Tukey,
        V_star_int_uv_Tukey,
        BIG_NUMBER,
        m_int_uv_Tukey,
        q_int_uv_Tukey,
        V_int_uv_Tukey,
        V_prime,
        m_star_int_wv_Tukey,
        q_star_int_wv_Tukey,
        V_star_int_wv_Tukey,
        mu,
        m_star_int_xigamma_Tukey,
        q_star_int_xigamma_Tukey,
        V_star_int_xigamma_Tukey,
    )
    from linear_regression.aux_functions.stability_functions import RS_int_mod_Tukey_decorrelated_noise
    print(f"BIG_NUMBER importé du module: {BIG_NUMBER}")
except Exception as e:
    print(f"Une autre erreur est survenue lors de l'importation : {e}")
    exit()


# --- Paramètres pour la visualisation ---
alpha, m_vis, q_vis, V_vis = 3.86170730e+00,5.79394816e-01,3.51861875e-01,3.53164220e-01
#m_vis = 0.4
#q_vis = 0.7
#V_vis = 0.8
lambda_vis = 2.0
tau_vis = 1.0
BIG_NUMBER = 15
DEFAULT_N_STD = 4 # Nombre d'écarts-types pour l'intégration en w

# Paramètres physiques
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1
BETA = 0.0
C_TUKEY = 0.001

#var_in, var_out = V_prime(q_vis,m_vis,V_vis,DELTA_IN,DELTA_OUT,PERCENTAGE,BETA,tau_vis)
mu_in, mu_out,J_beta_in,J_beta_out,delta_prime_in,delta_prime_out = mu(q_vis,m_vis,V_vis,DELTA_IN,DELTA_OUT,PERCENTAGE,BETA,tau_vis)
var_mu_in = np.sqrt(mu_in)**(-1)
var_mu_out = np.sqrt(mu_out)**(-1)
center_in=J_beta_in*tau_vis/(delta_prime_in*mu_in)
center_out=J_beta_out*tau_vis/(delta_prime_out*mu_out)

print(f"var_mu_in: {var_mu_in:.2f}, var_mu_out: {var_mu_out:.2f}, J_beta_in: {J_beta_in:.2f}, J_beta_out: {J_beta_out:.2f}, delta_prime_in: {delta_prime_in:.2f}, delta_prime_out: {delta_prime_out:.2f}, center_in: {center_in:.2f}, center_out: {center_out:.2f}")

# Paramètres de la grille de visualisation
GRID_POINTS = 100
PLOT_RANGE_EXTENSION = 1.2 # Garder une petite extension pour voir les bords

# --- Création de la grille ---
#xi_min, xi_max = -PLOT_RANGE_EXTENSION * BIG_NUMBER, PLOT_RANGE_EXTENSION * BIG_NUMBER
#y_min, y_max = -PLOT_RANGE_EXTENSION * BIG_NUMBER, PLOT_RANGE_EXTENSION * BIG_NUMBER

#xi_grid = np.linspace(xi_min, xi_max, GRID_POINTS)
#y_grid = np.linspace(y_min, y_max, GRID_POINTS)
#XI, Y = np.meshgrid(xi_grid, y_grid)

#v_min, v_max = -tau_vis/10 * PLOT_RANGE_EXTENSION, tau_vis * PLOT_RANGE_EXTENSION
#u_min_in, u_max_in = -DEFAULT_N_STD*var_in * PLOT_RANGE_EXTENSION, (tau_vis +DEFAULT_N_STD*var_in) * PLOT_RANGE_EXTENSION
#u_min_out, u_max_out = -DEFAULT_N_STD*var_out * PLOT_RANGE_EXTENSION, (tau_vis+DEFAULT_N_STD*var_out) * PLOT_RANGE_EXTENSION
#w_min_in, w_max_in = -DEFAULT_N_STD*var_in * PLOT_RANGE_EXTENSION, DEFAULT_N_STD*var_in * PLOT_RANGE_EXTENSION
#w_min_out, w_max_out = -DEFAULT_N_STD*var_out * PLOT_RANGE_EXTENSION, DEFAULT_N_STD*var_out * PLOT_RANGE_EXTENSION

xi_min_axis_in, xi_max_axis_in = -DEFAULT_N_STD*var_mu_in+min(0,center_in), DEFAULT_N_STD*var_mu_in+max(0,center_in)
xi_min_axis_out, xi_max_axis_out = - DEFAULT_N_STD*var_mu_out+min(0,center_out),DEFAULT_N_STD*var_mu_out+max(0,center_out)
xi_min_in, xi_max_in = PLOT_RANGE_EXTENSION * xi_min_axis_in, PLOT_RANGE_EXTENSION * xi_max_axis_in
xi_min_out, xi_max_out = PLOT_RANGE_EXTENSION * xi_min_axis_out, PLOT_RANGE_EXTENSION * xi_max_axis_out
gamma_min, gamma_max = -PLOT_RANGE_EXTENSION * delta_prime_in/10, PLOT_RANGE_EXTENSION * (DEFAULT_N_STD*np.sqrt(delta_prime_in*mu_in)) # changer in / out

#v_grid = np.linspace(v_min, v_max, GRID_POINTS)
#u_grid = np.linspace(u_min_in, u_max_in, GRID_POINTS)
#w_grid_in = np.linspace(w_min_in, w_max_in, GRID_POINTS)
#w_grid_out = np.linspace(w_min_out, w_max_out, GRID_POINTS)
#w_axis, v_axis = np.meshgrid(w_grid_in,v_grid) # changer in / out

xi_grid_in = np.linspace(xi_min_in, xi_max_in, GRID_POINTS)
xi_grid_out = np.linspace(xi_min_out, xi_max_out, GRID_POINTS)
gamma_grid = np.linspace(gamma_min, gamma_max, GRID_POINTS)
xi_axis, gamma_axis = np.meshgrid(xi_grid_in, gamma_grid) # changer in / out

print(f"Grille créée avec {GRID_POINTS} points dans chaque direction. Plage de xi: [{xi_min_in:.2f}, {xi_max_in:.2f}] (in), [{xi_min_out:.2f}, {xi_max_out:.2f}] (out), Plage de gamma: [{gamma_min:.2f}, {gamma_max:.2f}]")

# --- Évaluation des intégrandes ---
args = (q_vis, m_vis, V_vis, DELTA_IN, DELTA_OUT, PERCENTAGE, BETA, tau_vis, C_TUKEY)
args_in = (q_vis, m_vis, V_vis, DELTA_IN, 1, tau_vis, C_TUKEY)
args_out = (q_vis, m_vis, V_vis, DELTA_OUT, BETA, tau_vis, C_TUKEY)
Z_m = np.zeros_like(xi_axis) # Remplacer par u_axis si nécessaire
Z_q = np.zeros_like(xi_axis) # Remplacer par u_axis si nécessaire
Z_V = np.zeros_like(xi_axis) # Remplacer par u_axis si nécessaire
Z_RS = np.zeros_like(xi_axis) # Remplacer par u_axis si nécessaire

print("Évaluation des intégrandes sur la grille...")
for i in range(GRID_POINTS):
    for j in range(GRID_POINTS):
        #xi_val = XI[i, j]
        #y_val = Y[i, j]
        gamma_val = gamma_axis[i, j]
        xi_val = xi_axis[i, j] # Remplacer par u_axis si nécessaire
        #print(f"Évaluation des intégrandes pour xi={xi_val:.2f}, gamma={gamma_val:.2f}...")
        try:
            #Z_m[i, j] = m_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
            Z_m[i, j] = 2*m_star_int_xigamma_Tukey(xi_val, gamma_val, *args_in) # changer in / out
        except Exception: Z_m[i, j] = np.nan
        try:
            #Z_q[i, j] = q_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
            Z_q[i, j] = 2*q_star_int_xigamma_Tukey(xi_val, gamma_val, *args_in) # changer in / out
        except Exception: Z_q[i, j] = np.nan
        try:
            #Z_V[i, j] = V_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
            Z_V[i, j] = -2*V_star_int_xigamma_Tukey(xi_val, gamma_val, *args_in) # changer in / out
        except Exception: Z_V[i, j] = np.nan
        #try:
            #Z_RS[i, j] = alpha*RS_int_mod_Tukey_decorrelated_noise(xi_val, y_val, *args)
        #except Exception: Z_RS[i, j] = np.nan

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
    surf = ax.plot_surface(xi_axis, gamma_axis, Z, cmap=cmap, linewidth=0, antialiased=False, rstride=1, cstride=1) # changer les axes

    # Ajouter une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    # Ajouter des contours projetés sur le plan z=min(Z) pour aider à la visualisation
    z_min = np.nanmin(Z) if np.any(np.isfinite(Z)) else 0
    try:
         if np.any(np.isfinite(Z)) and np.nanstd(Z) > 1e-9:
              cset = ax.contour(xi_axis , gamma_axis, Z, zdir='z', offset=z_min, cmap=cmap, linewidths=0.5, alpha=0.5) # changer les axes
         else:
             print(f"Contours non tracés pour '{title}' (données non finies ou constantes)")
    except Exception as e:
        print(f"Erreur lors du tracé des contours projetés pour '{title}': {e}")


    ax.set_title(title)
    #ax.set_xlabel(r'$\xi$')
    ax.set_xlabel(r'$\xi$')
    #ax.set_ylabel(r'$y$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel('Valeur Intégrande')

    # Limiter les axes x et y aux bornes d'intégration pour se concentrer sur la zone pertinente
    #ax.set_xlim(-BIG_NUMBER, BIG_NUMBER)
    ax.set_xlim(xi_min_axis_in, xi_max_axis_in) # changer in /out
    #ax.set_ylim(-BIG_NUMBER, BIG_NUMBER)
    ax.set_ylim(0, DEFAULT_N_STD*np.sqrt(delta_prime_in*mu_in )) # changer in /out

    # Ajuster l'angle de vue
    # ax.view_init(elev=20., azim=-65)

fig.suptitle(
    f'Visualisation 3D des Intégrandes inliers (Tukey Mod.) pour τ={tau_vis:.2f}, c={C_TUKEY:.3f}, λ={lambda_vis:.2f}, δ_in={DELTA_IN:.2f}, δ_out={DELTA_OUT:.2f}, β={BETA:.2f}, alpha={alpha:.2f}',
    fontsize=11
) #changer in / out
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print("Plots créés. Affichage...")
plt.show()
