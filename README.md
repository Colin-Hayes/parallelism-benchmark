# Parallelism Benchmark

Compares ZeRO (DeepSpeed Stage 0 and Stage 3) and Megatron TP+PP across GPT-2-style model sizes on 4× NVIDIA GPUs.

| Strategy | Description |
|---|---|
| ZeRO-0 | Standard DDP — full model + optimizer states on each GPU |
| ZeRO-3 | Full sharding of weights, gradients, and optimizer states across GPUs |
| Megatron TP=4, PP=1 | Tensor parallel across all 4 GPUs, no pipeline |
| Megatron TP=2, PP=2 | 2 TP ranks per stage, 2 pipeline stages (1F1B schedule) |

Model sizes: 125M, 1.3B, 2.7B, 6.7B, 10B · Batch size: 4/GPU · Sequence length: 512

Each config runs in its own `torchrun` subprocess so an OOM or NCCL crash in one config does not kill the benchmark — it is recorded as `status: "OOM"` or `"crash"` and the next config runs in a fresh process.

---

## Setup

```bash
# Clone into home directory on Delta
git clone https://github.com/Colin-Hayes/parallelism-benchmark.git ~/benchmark

# Create environment
module load pytorch-conda
python -m venv ~/benchmark_env --system-site-packages
source ~/benchmark_env/bin/activate
pip install deepspeed transformers megatron-core

# Create output directories in project storage
mkdir -p /projects/bdes/$USER/benchmark/results
mkdir -p /projects/bdes/$USER/benchmark/logs
```

---

## Running

### Full benchmark (all model sizes)

```bash
sbatch ~/benchmark/run_zero_bench.slurm
sbatch ~/benchmark/run_megatron_bench.slurm
```

### 10B only

```bash
sbatch ~/benchmark/run_zero_10B.slurm
sbatch ~/benchmark/run_megatron_10B.slurm
```

Results are written to `/projects/bdes/$USER/benchmark/results/` and copied to `~/benchmark/`.

---

## Interactive debugging

```bash
srun --account=bdes-delta-gpu \
     --partition=gpuA40x4-interactive \
     --nodes=1 --gpus-per-node=4 \
     --ntasks-per-node=1 --cpus-per-task=16 \
     --mem=64g --time=01:00:00 \
     --pty bash

source ~/benchmark_env/bin/activate

# Dry run — small model, fast
python ~/benchmark/zero_bench.py    --output /tmp/test_zero.json    --dry_run
python ~/benchmark/megatron_bench.py --output /tmp/test_megatron.json --dry_run --nproc_per_node 4

# Single model size
python ~/benchmark/zero_bench.py    --output /tmp/test.json --model_size 1.3B
python ~/benchmark/megatron_bench.py --output /tmp/test.json --model_size 1.3B --nproc_per_node 4
```

---

## Files

| File | Role |
|---|---|
| `ZeRO.py` | DeepSpeed benchmark core — model init, training loop, memory profiling |
| `zero_run_config.py` | Single-config torchrun entrypoint called by `zero_bench.py` |
| `zero_bench.py` | Orchestrator — spawns one subprocess per (stage, model_size) config |
| `Megatron.py` | Megatron-core benchmark core — TP+PP model, 1F1B schedule |
| `megatron_run_config.py` | Single-config torchrun entrypoint called by `megatron_bench.py` |
| `megatron_bench.py` | Orchestrator — spawns one subprocess per (model_size, tp, pp) config |
