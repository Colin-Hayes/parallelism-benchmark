#!/bin/bash
# run_zero_profile.slurm
#SBATCH --job-name=zero_profile
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdes-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -o /projects/bdes/$USER/benchmark/logs/zero_profile_%j.out
#SBATCH -e /projects/bdes/$USER/benchmark/logs/zero_profile_%j.err

module load pytorch-conda
source ~/benchmark_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

mkdir -p /projects/bdes/$USER/benchmark/logs

echo "========================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "========================================================"

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29512 \
    /u/$USER/benchmark/profile_zero_memory.py \
    --model_size 6.7B

echo ""
echo "End time : $(date)"
echo "Exit code: $?"
