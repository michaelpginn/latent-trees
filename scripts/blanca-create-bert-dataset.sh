#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=24:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=create_dataset.%j.out      # Output file name
#SBATCH --error=create_dataset.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
conda activate AutoIGT

export HF_DATASETS_CACHE=$SLURM_SCRATCH

# Run Python Script
cd /projects/migi8081/latent-trees

python3 create_pretrain_corpus.py
