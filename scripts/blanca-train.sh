#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=12:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=train.%j.out      # Output file name
#SBATCH --error=train.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
conda activate AutoIGT

# Run Python Script
cd /projects/migi8081/latent-trees/src

#python3 train.py --dataset 'ID' --train_epochs 1000
#python3 train.py --dataset 'ID' --pretrained --train_epochs 1000
#python3 train.py --dataset 'GEN' --train_epochs 1000
#python3 train.py --dataset 'GEN' --pretrained --train_epochs 1000
python3 train.py --dataset 'GENX' --train_epochs 1000
python3 train.py --dataset 'GENX' --pretrained --train_epochs 1000

python3 train.py --dataset 'ID' --train_epochs 1000 --use_tree_bert
#python3 train.py --dataset 'ID' --pretrained --train_epochs 1000  --use_tree_bert
python3 train.py --dataset 'GEN' --train_epochs 1000  --use_tree_bert
#python3 train.py --dataset 'GEN' --pretrained --train_epochs 1000  --use_tree_bert
python3 train.py --dataset 'GENX' --train_epochs 1000  --use_tree_bert
