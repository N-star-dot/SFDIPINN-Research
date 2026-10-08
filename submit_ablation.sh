#!/bin/bash -l
#$ -l h_rt=12:00:00         # Request 12 hours of runtime
#$ -pe omp 4                # Request 4 CPU cores
#$ -N sfdi_ablation         # Name of the job
#$ -j y                     # Merge standard error and output into one file

# Load modules and activate environment
module load python3/3.12
cd /usr4/ugrad/nghia/SFDI/SFDIPINN-Research/
source .venv/bin/activate

# Run the MLP first to ensure weights are generated
python scripts/00b_train_forward_mlp.py

# Run the ablation script
python scripts/04_ablation.py
