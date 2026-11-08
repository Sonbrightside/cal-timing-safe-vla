#!/bin/bash
#SBATCH -J smolvla_prof
#SBATCH -p public
#SBATCH -q htc
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/prof_%j.out


conda activate /scratch/hson17/envs/lerobot-libero
cd /scratch/hson17/cal_project
source env.sh

echo "node=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu --format=csv

for t in 0 1 2 3 4 5 6 7 8 9; do
    echo "===== task $t ====="
    python profile_smolvla.py profile_${SLURM_JOB_ID}/task${t} ${t}
done