#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00          # Max walltime
#SBATCH --out=train.%j.out      # Output file name
#SBATCH --error=train.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# Load the python module
module load gcc/9.3.0 arrow/8
module load python/3.10
module load scipy-stack

# Create the env
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -r ~/scratch/latent-trees/requirements.txt

# Run Python Script
cd ~/scratch/latent-trees/src

python3 train.py
