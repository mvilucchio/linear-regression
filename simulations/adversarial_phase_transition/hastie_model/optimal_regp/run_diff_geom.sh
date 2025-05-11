#!/bin/bash

# alpha one
# alpha_min, alpha_max, n_alphas, gamma, pstar_t, reg_p, metric_type_chosen, pstar_g

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 0.5 1.0 2.0 misclass 1.0 & 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 0.5 1.0 2.0 misclass 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.0 1.0 2.0 misclass 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.0 1.0 2.0 misclass 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.5 1.0 2.0 misclass 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.5 1.0 2.0 misclass 2.0 &

wait 

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 0.5 1.0 2.0 adv 1.0 & 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 0.5 1.0 2.0 adv 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.0 1.0 2.0 adv 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.0 1.0 2.0 adv 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.5 1.0 2.0 adv 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.5 1.0 2.0 adv 2.0 &

wait 

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 0.5 1.0 2.0 bound 1.0 & 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 0.5 1.0 2.0 bound 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.0 1.0 2.0 bound 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.0 1.0 2.0 bound 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.5 1.0 2.0 bound 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp_diff_geom.py 0.2 2.5 50 1.5 1.0 2.0 bound 2.0 &

wait 

# gamma one
# gamma_min, gamma_max, n_gammas, alpha, pstar_t, reg_p, metric_type_chosen, pstar_g

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 0.5 1.0 2.0 misclass 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 0.5 1.0 2.0 misclass 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.0 1.0 2.0 misclass 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.0 1.0 2.0 misclass 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.5 1.0 2.0 misclass 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.5 1.0 2.0 misclass 2.0 &

wait

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 0.5 1.0 2.0 adv 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 0.5 1.0 2.0 adv 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.0 1.0 2.0 adv 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.0 1.0 2.0 adv 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.5 1.0 2.0 adv 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.5 1.0 2.0 adv 2.0 &

wait

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 0.5 1.0 2.0 bound 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 0.5 1.0 2.0 bound 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.0 1.0 2.0 bound 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.0 1.0 2.0 bound 2.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.5 1.0 2.0 bound 1.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp_diff_geom.py 0.5 2.0 50 1.5 1.0 2.0 bound 2.0 &

wait
