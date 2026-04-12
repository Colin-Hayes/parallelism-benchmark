#!/bin/bash
# run_pp_bench.slurm
#SBATCH --job-name=pp_bench
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdes-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=208G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -o /work/hdd/bdes/$USER/benchmark/logs/pp_%j.out
#SBATCH -e /work/hdd/bdes/$USER/benchmark/logs/pp_%j.err

module load pytorch-conda
source ~/benchmark_env/bin/activate

RESULTS=/work/hdd/bdes/$USER/benchmark/results
mkdir -p $RESULTS /work/hdd/bdes/$USER/benchmark/logs

python /u/$USER/benchmark/pp_bench.py \
    --output $RESULTS/results_pp_${SLURM_JOB_ID}.json