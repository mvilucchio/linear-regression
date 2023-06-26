#!/bin/bash
#SBATCH --chdir=/home/vilucchi/github_proj/linear_regression
#SBATCH --job-name=amp_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --output=amp_parallel.out
#SBATCH --error=amp_parallel.err
#SBATCH --time=24:00:00

module load gcc openmpi python/3.10.4
source /home/vilucchi/venvs/my-venv/bin/activate

srun --mpi=pmi2 python3 ./simulations/cluster_simulations/MPI_just_simulations_amp_L1.py

deactivate
