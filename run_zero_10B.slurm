#!/bin/bash
#SBATCH --job-name=zero_10B
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdes-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -o /projects/bdes/%u/benchmark/logs/zero_10B_%j.out
#SBATCH -e /projects/bdes/%u/benchmark/logs/zero_10B_%j.err

module load pytorch-conda
source ~/benchmark_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

RESULTS=/projects/bdes/$USER/benchmark/results
mkdir -p $RESULTS /projects/bdes/$USER/benchmark/logs

python /u/$USER/benchmark/zero_bench.py \
    --output $RESULTS/results_zero_10B_${SLURM_JOB_ID}.json \
    --model_size 10B \
    --nproc_per_node 4

cp $RESULTS/results_zero_10B_${SLURM_JOB_ID}.json \
   /u/$USER/benchmark/results_zero_10B_${SLURM_JOB_ID}.json 2>/dev/null || true
