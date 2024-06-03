#!/bin/bash
#SBATCH --chdir=/home/vilucchi/projects/linear-regression
#SBATCH --job-name=any_attack_ERM
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --output=output_any_attack_perturbation.out
#SBATCH --error=error_any_attack_perturbation.err
#SBATCH --time=12:00:00
#SBATCH --qos=serial

module load gcc openmpi python/3.10.4
source /home/vilucchi/projects/linear-regression/venv/bin/activate

srun --mpi=pmi2 python3 ./simulations/any_attack_perturbation/cluster_ERM_data_adv_different_reg.py

deactivate
