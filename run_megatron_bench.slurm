#!/bin/bash
# run_megatron_bench.slurm
#SBATCH --job-name=megatron_bench
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdes-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -o /projects/bdes/$USER/benchmark/logs/megatron_%j.out
#SBATCH -e /projects/bdes/$USER/benchmark/logs/megatron_%j.err

module load pytorch-conda
source ~/benchmark_env/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

BASE=/projects/bdes/$USER/benchmark
RESULTS=$BASE/results
LOGS=$BASE/logs
mkdir -p $RESULTS $LOGS

# ── Job metadata ──────────────────────────────────────────────────────────────
echo "========================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "========================================================"

# ── Software versions ─────────────────────────────────────────────────────────
echo ""
echo "--- Software versions ---"
python  --version
python  -c "import torch;        print('torch        ', torch.__version__)"
python  -c "import megatron.core; import importlib.metadata; print('megatron-core', importlib.metadata.version('megatron-core'))" 2>/dev/null || echo "megatron-core version unknown"
nvcc    --version 2>/dev/null | grep release || echo "nvcc not on PATH"
echo ""

# ── GPU inventory ─────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap \
    --format=csv | tee $RESULTS/gpu_info_${SLURM_JOB_ID}.txt
echo ""

# ── Continuous GPU monitor (runs in background, sampled every 10 s) ───────────
nvidia-smi dmon \
    --select ugmt \
    --delay 10 \
    > $LOGS/gpu_dmon_${SLURM_JOB_ID}.txt &
DMON_PID=$!

# ── Benchmark ────────────────────────────────────────────────────────────────
# megatron_bench.py is a plain Python orchestrator — it spawns one torchrun
# subprocess per config so NCCL crashes are isolated and don't kill the job.
python /u/$USER/benchmark/megatron_bench.py \
    --output $RESULTS/results_megatron_${SLURM_JOB_ID}.json \
    --nproc_per_node 4

STATUS=$?

# ── Copy results to home directory ────────────────────────────────────────────
cp $RESULTS/results_megatron_${SLURM_JOB_ID}.json /u/$USER/benchmark/results_megatron_${SLURM_JOB_ID}.json 2>/dev/null || true

# ── Teardown ──────────────────────────────────────────────────────────────────
kill $DMON_PID 2>/dev/null

echo ""
echo "--- Job finished ---"
echo "End time  : $(date)"
echo "Exit code : $STATUS"
exit $STATUS
