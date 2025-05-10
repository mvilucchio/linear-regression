#!/bin/bash

echo "Running SE with optimal regp"

# alpha_min, alpha_max, n_alphas, pstar, reg_p, metric_type_chosen

python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 2.5 30 1.0 1.0 misclass & # > miscalss1.log 2>miscalss1.err
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 2.5 30 1.0 1.0 bound & # > bound1.log 2>bound1.err
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 2.5 30 1.0 1.0 adv & # > adv1.log 2>adv1.err

# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 20 2.0 2.0 misclass > se_miscalss2.log 2>se_miscalss2.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 20 2.0 2.0 bound > se_bound2.log 2>se_bound2.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 20 2.0 2.0 adv > se_adv2.log 2>se_adv2.err &

wait

# alpha_min_se, alpha_max_se, n_alphas_se, d, delta, pstar, reg_p, metric_name_chosen

# echo "Running ERM with optimal regp"

# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.25 3.0 20 500 0.0 2.0 2.0 misclass & #> erm_miscalss2.log 2>erm_miscalss2.err 
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.25 3.0 20 500 0.0 2.0 2.0 bound & # > erm_bound2.log 2>erm_bound2.err 
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.25 3.0 20 500 0.0 2.0 2.0 adv # > erm_adv2.log 2>erm_adv2.err 

# wait
