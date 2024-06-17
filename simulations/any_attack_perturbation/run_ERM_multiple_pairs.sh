#!/bin/bash
#SBATCH --chdir=/home/vilucchi/projects/linear-regression
#SBATCH --job-name=ERM_reg_eps
#SBATCH --nodes=1
#SBATCH --ntasks=27
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --output=simulations/any_attack_perturbation/output_multiple_pairs.out
#SBATCH --error=simulations/any_attack_perturbation/error_multiple_pairs.err
#SBATCH --time=02:00:00
#SBATCH --qos=serial

module load gcc openmpi python/3.10.4
source /home/vilucchi/projects/linear-regression/venv/bin/activate

srun --mpi=pmi2 python3 ./simulations/any_attack_perturbation/cluster_ERM_multiple_pairs.py

deactivate
