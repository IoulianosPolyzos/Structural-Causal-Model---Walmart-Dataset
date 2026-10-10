#!/bin/bash

#SBATCH --job-name=dataset      # task_name
#SBATCH --partition=yoda             #
#SBATCH --nodes=1                       #  1 node
#SBATCH --ntasks=1                      # 1 task
#SBATCH --cpus-per-task=4               # 4 cores CPU
#SBATCH --mem=16G                       # 16GB RAM
#SBATCH --time=05:00:00                 # 4 hours
#SBATCH --output=logs/output_%j.txt     #
#SBATCH --error=logs/error_%j.txt       #

# Δημιουργία φακέλου για τα logs αν δεν υπάρχει
mkdir -p logs

source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate dowhy


echo "Starting dataset preprocessing at: $(date)"

python /home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/main.py


echo "Finished experiment at: $(date)"
