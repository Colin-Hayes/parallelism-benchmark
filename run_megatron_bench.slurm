#!/bin/bash
# run_megatron_bench.slurm
#SBATCH --job-name=megatron_bench
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdes-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -o /u/$USER/benchmark/logs/megatron_%j.out
#SBATCH -e /u/$USER/benchmark/logs/megatron_%j.err

module load pytorch-conda
source ~/benchmark_env/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30

RESULTS=/u/$USER/benchmark/results
mkdir -p $RESULTS /u/$USER/benchmark/logs

nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv \
    | tee $RESULTS/gpu_info_${SLURM_JOB_ID}.txt

# megatron_bench.py is a plain Python orchestrator — it spawns one torchrun
# subprocess per config so NCCL crashes are isolated and don't kill the job.
python /u/$USER/benchmark/megatron_bench.py \
    --output $RESULTS/results_megatron_${SLURM_JOB_ID}.json \
    --nproc_per_node 4
