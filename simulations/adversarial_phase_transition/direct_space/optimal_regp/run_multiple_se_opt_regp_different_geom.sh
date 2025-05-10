#!/bin/bash

# alpha_min, alpha_max, n_alphas, pstar_t, reg_p, metric_type_chosen, pstar_g

echo "Running SE with optimal regp"

# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 2.0 2.0 misclass 2.0 & #  > se_miscalss2.log 2>se_miscalss2.err
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 2.0 2.0 bound 2.0 & # bound > se_bound2.log 2>se_bound2.err
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 2.0 2.0 adv 2.0 & #  > se_adv2.log 2>se_adv2.err

# wait

# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 2.0 2.0 misclass 1.0 & #  > se_miscalss2.log 2>se_miscalss2.err
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 2.0 2.0 bound 1.0 & # bound > se_bound2.log 2>se_bound2.err
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 2.0 2.0 adv 1.0 & #  > se_adv2.log 2>se_adv2.err

# wait

# alpha_min_se, alpha_max_se, n_alphas_se, d, delta, pstar_t, reg_p, metric_name_chosen, pstar_g

echo "Running ERM with optimal regp"

python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 500 0.0 2.0 2.0 misclass 2.0 & #> erm_miscalss2.log 2>erm_miscalss2.err 
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 500 0.0 2.0 2.0 bound 2.0 & # > erm_bound2.log 2>erm_bound2.err 
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 500 0.0 2.0 2.0 adv 2.0 & # > erm_adv2.log 2>erm_adv2.err 

wait

python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 500 0.0 2.0 2.0 misclass 1.0 & #> erm_miscalss2.log 2>erm_miscalss2.err 
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 500 0.0 2.0 2.0 bound 1.0 & # > erm_bound2.log 2>erm_bound2.err 
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp_different_geom.py 0.25 3.0 20 500 0.0 2.0 2.0 adv 1.0 & # > erm_adv2.log 2>erm_adv2.err 

wait
