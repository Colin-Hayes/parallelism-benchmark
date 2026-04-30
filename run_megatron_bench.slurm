#!/bin/bash
# run_megatron_bench.slurm
#SBATCH --job-name=megatron_bench
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdes-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -o /work/hdd/bdes/$USER/benchmark/logs/megatron_%j.out
#SBATCH -e /work/hdd/bdes/$USER/benchmark/logs/megatron_%j.err

module load pytorch-conda
source ~/benchmark_env/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1

RESULTS=/work/hdd/bdes/$USER/benchmark/results
mkdir -p $RESULTS /work/hdd/bdes/$USER/benchmark/logs

nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv \
    | tee $RESULTS/gpu_info_${SLURM_JOB_ID}.txt

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29511 \
    /u/$USER/benchmark/megatron_bench.py \
    --output $RESULTS/results_megatron_${SLURM_JOB_ID}.json
