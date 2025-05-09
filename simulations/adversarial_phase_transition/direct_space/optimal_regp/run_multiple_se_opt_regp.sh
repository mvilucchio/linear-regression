#!/bin/bash

# alpha_min, alpha_max, n_alphas, pstar, reg_p, metric_type_chosen


# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 1.0 1.0 misclass > miscalss1.log 2>miscalss1.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 1.0 1.0 bound > bound1.log 2>bound1.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 1.0 1.0 adv > adv1.log 2>adv1.err &

# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 2.0 2.0 misclass > miscalss2.log 2>miscalss2.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 2.0 2.0 bound > bound2.log 2>bound2.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 2.0 2.0 adv > adv2.log 2>adv2.err &

# alpha_min_se, alpha_max_se, n_alphas_se, d, delta, pstar, reg_p, metric_name_chosen

python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.25 3.0 50 500 0.0 2.0 2.0 misclass > erm_miscalss2.log 2>erm_miscalss2.err &
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.25 3.0 50 500 0.0 2.0 2.0 bound > erm_bound2.log 2>erm_bound2.err &
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.25 3.0 50 500 0.0 2.0 2.0 adv > erm_adv2.log 2>erm_adv2.err &

wait
