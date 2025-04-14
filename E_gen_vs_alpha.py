import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (
        RS_alpha_E2_mod_Tukey_decorrelated_noise
    )
from linear_regression.aux_functions.moreau_proximals import DƔ_proximal_L2


# --- Importation des fonctions ---
try:
    from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
    from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
    from linear_regression.fixed_point_equations.regression.mod_Tukey_loss import (
        f_hat_mod_Tukey_decorrelated_noise,
    )
    from linear_regression.utils.errors import ConvergenceError
except Exception as e:
    print(f"Une autre erreur est survenue lors de l'importation : {e}")
    exit()

# --- Paramètres ---

# Paramètres fixes du modèle et de la perte
DELTA_IN = 0.1
DELTA_OUT = 1.0
PERCENTAGE = 0.1 # Epsilon
BETA = 0.0
C_TUKEY = 0.01

# Hyperparamètres fixés pour le balayage en alpha
REG_PARAM = 1.0 # Exemple de lambda fixe (à ajuster si besoin)
TAU = 1.0       # Exemple de tau fixe (à ajuster si besoin)
print(f"Utilisation des hyperparamètres fixes : reg_param={REG_PARAM:.1f}, tau={TAU:.1f}")

# Plage pour alpha
ALPHA_MIN = 1e0
ALPHA_MAX = 1e3
N_ALPHA_PTS = 100 # Nombre de points pour alpha (log-espacés)

# Options d'intégration (valeurs relâchées pour la vitesse)
INTEGRATION_BOUND = 10.0
INTEGRATION_EPSABS = 1e-10
INTEGRATION_EPSREL = 1e-7
print(f"Options Intégration: bound={INTEGRATION_BOUND}, epsabs={INTEGRATION_EPSABS:.1e}, epsrel={INTEGRATION_EPSREL:.1e}")

# Condition initiale pour le premier alpha (peut nécessiter ajustement)
initial_cond_fpe = (0.8, 0.7, 0.06)

# --- Balayage en Alpha ---
alphas = np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), N_ALPHA_PTS)

MQV_results = np.zeros((N_ALPHA_PTS, 6)) # Pour stocker les résultats (m, q, V, m_hat, q_hat, V_hats)

# Arguments fixes pour f_L2_reg
f_kwargs = {"reg_param": REG_PARAM}

print(f"Calcul des points fixes pour alpha de {ALPHA_MIN:.1e} à {ALPHA_MAX:.1e}...")
current_initial_cond = initial_cond_fpe # Initialisation pour le premier alpha

for idx, alpha in enumerate(tqdm(alphas, desc="Balayage Alpha")):
    # Mise à jour des arguments pour f_hat (alpha change)
    f_hat_kwargs = {
        "alpha": alpha,
        "delta_in": DELTA_IN, "delta_out": DELTA_OUT,
        "percentage": PERCENTAGE, "beta": BETA, "tau": TAU, "c": C_TUKEY,
        "integration_bound": INTEGRATION_BOUND,
        "integration_epsabs": INTEGRATION_EPSABS,
        "integration_epsrel": INTEGRATION_EPSREL
    }

    try:
        # Recherche du point fixe en utilisant la condition initiale adaptée
        m, q, V = fixed_point_finder(
            f_func=f_L2_reg,
            f_hat_func=f_hat_mod_Tukey_decorrelated_noise,
            initial_condition=current_initial_cond,
            f_kwargs=f_kwargs,
            f_hat_kwargs=f_hat_kwargs,
            verbose=False
        )
        m_hat, q_hat, V_hats = f_hat_mod_Tukey_decorrelated_noise(
            m, q, V,
            alpha=alpha,
            delta_in=DELTA_IN, delta_out=DELTA_OUT,
            percentage=PERCENTAGE, beta=BETA, tau=TAU, c=C_TUKEY,
            integration_bound=INTEGRATION_BOUND,
            integration_epsabs=INTEGRATION_EPSABS,
            integration_epsrel=INTEGRATION_EPSREL
        )
        print(f"Résultat pour alpha={alpha:.2e}: m={m}, q={q}, V={V}") # Debug

        # Si la convergence réussit et le résultat est valide
        if np.isfinite(m) and np.isfinite(q) and np.isfinite(V) and np.isfinite(m_hat) and np.isfinite(q_hat) and np.isfinite(V_hats):
            MQV_results[idx] = (m, q, V, m_hat, q_hat, V_hats)

            # Utiliser le résultat comme condition initiale pour le prochain alpha
            current_initial_cond = (m, q, V)
            
        else:
            # Laisser NaN, ne pas mettre à jour l'initialisation
            print(f"\nWarning: Résultat non fini pour alpha={alpha:.2e}. Garde init précédente.") # Debug

    # Si la convergence échoue ou autre erreur numérique
    except (ConvergenceError, FloatingPointError) as e:
        print(f"\nAttention: {type(e).__name__} pour alpha={alpha:.2e}. Point ignoré (NaN). Garde init précédente.") # Debug

    except Exception as e:
        print(f"\nErreur inattendue pour alpha={alpha:.2e}: {e}. Garde init précédente.")


# --- Calcul de la Fonction de Généralisation ---
print("Calcul de la fonction de généralisation...")

# Pré-calculer le facteur constant
prefactor = 1.0 + PERCENTAGE * (BETA - 1.0)
#prefactor = 1.0

# Calculer GenError seulement pour les points où m et q sont valides
valid_indices = ~np.isnan(MQV_results[:, 0]) & ~np.isnan(MQV_results[:, 1])
ms = MQV_results[valid_indices, 0]
qs = MQV_results[valid_indices, 1]
gen_error = np.full(N_ALPHA_PTS, np.nan) # Initialiser avec NaN
gen_error[valid_indices] = prefactor**2 + qs - 2 * ms * prefactor

# Vérifier combien de points ont été calculés avec succès
num_valid = np.sum(valid_indices)
print(f"Calcul terminé. {num_valid}/{N_ALPHA_PTS} points valides obtenus.")

if num_valid == 0:
    print("Aucun point valide à tracer. Vérifiez les paramètres ou la plage alpha. Arrêt.")
    exit()

# Calcul de RS
print("Calcul de la condition RS...")
RS_values = np.full(N_ALPHA_PTS, np.nan)

for j in tqdm(range(N_ALPHA_PTS), desc="Calcul de RS", unit="alpha",leave=False):
        m, q, V, m_hat, q_hat, V_hat = MQV_results[j, :]
        if np.isfinite(m) and np.isfinite(q) and np.isfinite(V) and np.isfinite(V_hat):
            #print(f"Calcul de RS pour reg_param={reg_params[i]:.3e}, tau={taus[j]:.3f}")
            try:
                RS_values[j] = RS_alpha_E2_mod_Tukey_decorrelated_noise(
                    m,
                    q,
                    V,
                    alphas[j],
                    DELTA_IN,
                    DELTA_OUT,
                    PERCENTAGE,
                    BETA,
                    TAU,
                    C_TUKEY,
                    INTEGRATION_BOUND,
                    INTEGRATION_EPSABS, 
                    INTEGRATION_EPSREL
                    ) * DƔ_proximal_L2(0, V_hat, REG_PARAM)**2
            except Exception as e:
                print(f"\nErreur inattendue pour RS: {e}.")
                pass

# --- Visualisation Log-Log ---
print("Génération du graphique log-log...")
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignorer les avertissements log(NaN)

fig, ax = plt.subplots(figsize=(8, 6))

# Tracer seulement les points valides pour gen_error
ax.plot(alphas[valid_indices], gen_error[valid_indices], marker='o', linestyle='-', markersize=4, label='GenError Calculée')
ax.plot(alphas[valid_indices], MQV_results[valid_indices,0], marker='o', linestyle='-', markersize=4, label='m')
ax.plot(alphas[valid_indices], MQV_results[valid_indices,1], marker='o', linestyle='-', markersize=4, label='q')
ax.plot(alphas[valid_indices], MQV_results[valid_indices,2], marker='o', linestyle='-', markersize=4, label='V')

# Formatage du graphique
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\alpha = n/d$')
ax.set_ylabel(r'GenError = $(1+\epsilon(\beta-1))^2 + q - 2m(1+\epsilon(\beta-1))$')
ax.set_title(f'Fonction de Généralisation et RS vs Alpha (log-log)\n($\lambda$={REG_PARAM:.1f}, $\\tau$={TAU:.1f}, $\\epsilon$={PERCENTAGE:.1f}, $\\beta$={BETA:.1f})') # Utilisation de lambda et tau dans le titre
ax.grid(True, which='both', linestyle=':')
ax.legend(loc='upper left')

# Ajout de l'axe pour RS de 0 à 1
ax2 = ax.twinx()
ax2.set_yscale('linear')
ax2.set_ylim(0, 1)
ax2.set_ylabel(r'RS')
ax2.plot(alphas[valid_indices], RS_values[valid_indices], marker='x', linestyle='--', markersize=4, color='orange', label='RS Calculée')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("gen_error_vs_alpha.png", dpi=300)
print("Graphique enregistré sous 'gen_error_vs_alpha.png'.")
#plt.show()

print("Script terminé.")
