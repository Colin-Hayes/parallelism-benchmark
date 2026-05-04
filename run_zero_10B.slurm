#!/bin/bash
# run_zero_10B.slurm
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
#SBATCH -o /projects/bdes/$USER/benchmark/logs/zero_10B_%j.out
#SBATCH -e /projects/bdes/$USER/benchmark/logs/zero_10B_%j.err

module load pytorch-conda
source ~/benchmark_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

BASE=/projects/bdes/$USER/benchmark
RESULTS=$BASE/results
LOGS=$BASE/logs
mkdir -p $RESULTS $LOGS

echo "========================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "========================================================"

echo ""
echo "--- Software versions ---"
python  --version
python  -c "import torch;      print('torch      ', torch.__version__)"
python  -c "import deepspeed;  print('deepspeed  ', deepspeed.__version__)"
nvcc    --version 2>/dev/null | grep release || echo "nvcc not on PATH"
echo ""

nvidia-smi --query-gpu=index,name,memory.total \
    --format=csv | tee $RESULTS/gpu_info_${SLURM_JOB_ID}.txt
echo ""

nvidia-smi dmon --select ugmt --delay 10 \
    > $LOGS/gpu_dmon_${SLURM_JOB_ID}.txt &
DMON_PID=$!

python /u/$USER/benchmark/zero_bench.py \
    --output $RESULTS/results_zero_10B_${SLURM_JOB_ID}.json \
    --model_size 10B \
    --nproc_per_node 4

STATUS=$?

cp $RESULTS/results_zero_10B_${SLURM_JOB_ID}.json \
   /u/$USER/benchmark/results_zero_10B_${SLURM_JOB_ID}.json 2>/dev/null || true

kill $DMON_PID 2>/dev/null

echo ""
echo "End time  : $(date)"
echo "Exit code : $STATUS"
exit $STATUS
