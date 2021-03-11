#!/bin/bash
#SBATCH -J td3-2-po-gs-2D  
#SBATCH -n 20
#SBATCH --mem-per-cpu=1100   
#SBATCH -t 10:00:00    
#SBATCH -e /work/scratch/bj19ihup/test.err.%j
#SBATCH -o /work/scratch/bj19ihup/test.out.%j
#SBATCH --mail-user=ramzi.ourari@tu-darmstadt.de
#SBATCH --mail-type=ALL
# -------------------------------
source anaconda3/etc/profile.d/conda.sh
conda activate swarm
cd /home/bj19ihup/swarm
srun --exclusive python train.py --config-file swarm-experiments/td3-2-po-gs-2D.yaml

