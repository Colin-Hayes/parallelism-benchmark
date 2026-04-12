To run the models, it generally took ~5 minutes:

```bash
# 1. Clone the repo in u/user/~
git clone https://github.com/Colin-Hayes/parallelism-benchmark.git benchmark

# 2. Set up the environment
module load pytorch-conda
python -m venv ~/benchmark_env --system-site-packages
source ~/benchmark_env/bin/activate
pip install deepspeed fairscale transformers

# 3. Create output directories in work
mkdir -p /work/hdd/bdes/$USER/benchmark/results
mkdir -p /work/hdd/bdes/$USER/benchmark/logs

# 4. Submit jobs
sbatch ~/benchmark/run_zero_bench.slurm
sbatch ~/benchmark/run_pp_bench.slurm

# You can request an interactive session to debug if it is not running
srun --account=bdes-delta-gpu \
     --partition=gpuA100x4-interactive \
     --nodes=1 --gpus-per-node=4 \
     --ntasks-per-node=1 --cpus-per-task=16 \
     --mem=64g --time=01:00:00 \
     --pty bash

# ZeRO test, dry_run = small model
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29510 \
    zero_bench.py \
    --output ~/benchmark/test_results.json \
    --dry_run

# Pipeline — run after ZeRO completes
python pp_bench.py \
    --output ~/benchmark/test_pp_results.json \
    --dry_run

exit #when done
```

### What to work on 

We may be able to get access to more GPUs. I have not tried to connect GPUs across nodes, so I don't know if that would work, but with more GPUs, we can make a better Megatron like model combining PP with TP and DP. Right now it's just PP.  

We need to debug the code. I think ZeRO 3 should be able to run the largest model, but this is the output:

We most likely need to change the deepspeed config. Zero 0 is just standard DP without sharding, ZeRO 3 is supposed to be fully sharded so it should have much less gpu memeory usage. Maybe the all gather is incorrectly getting too much information. We can offload the optimizer states to the CPU, but I think it should fit without this.

My assumptions could be wrong
```json
[
  {
    "strategy": "zero0",
    "throughput_samples_per_sec": 85.53,
    "peak_gpu_mem_gb_rank0": 4.198,
    "status": "ok",
    "error": null,
    "model_size": "125M",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "zero3",
    "throughput_samples_per_sec": 33.54,
    "peak_gpu_mem_gb_rank0": 4.209,
    "status": "ok",
    "error": null,
    "model_size": "125M",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "zero0",
    "throughput_samples_per_sec": 16.22,
    "peak_gpu_mem_gb_rank0": 33.104,
    "status": "ok",
    "error": null,
    "model_size": "1.3B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "zero3",
    "throughput_samples_per_sec": 15.48,
    "peak_gpu_mem_gb_rank0": 17.348,
    "status": "ok",
    "error": null,
    "model_size": "1.3B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "zero0",
    "throughput_samples_per_sec": null,
    "peak_gpu_mem_gb_rank0": null,
    "status": "OOM",
    "error": "CUDA out of memory. Tried to allocate 9.87 GiB. GPU 0 has a total capacity of 39.49 GiB of which 5.12 GiB is free. Including non-PyTorch memory, this process has 34.37 GiB memory in use. Of the allocated memory 31.99 GiB is allocated by PyTorch, and 824.61 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)",
    "model_size": "2.7B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "zero3",
    "throughput_samples_per_sec": 9.59,
    "peak_gpu_mem_gb_rank0": 33.174,
    "status": "ok",
    "error": null,
    "model_size": "2.7B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "zero0",
    "throughput_samples_per_sec": null,
    "peak_gpu_mem_gb_rank0": null,
    "status": "OOM",
    "error": "CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 32.31 MiB is free. Including non-PyTorch memory, this process has 39.45 GiB memory in use. Of the allocated memory 37.78 GiB is allocated by PyTorch, and 98.90 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)",
    "model_size": "6.7B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "zero3",
    "throughput_samples_per_sec": null,
    "peak_gpu_mem_gb_rank0": null,
    "status": "OOM",
    "error": "CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 14.31 MiB is free. Including non-PyTorch memory, this process has 39.47 GiB memory in use. Of the allocated memory 37.80 GiB is allocated by PyTorch, and 98.89 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)",
    "model_size": "6.7B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  }
]
```

here is the output for PP:

```json
[
  {
    "strategy": "pipeline_mp",
    "num_stages": 4,
    "num_chunks": 4,
    "throughput_samples_per_sec": 48.76,
    "peak_gpu_mem_gb": 1.675,
    "status": "ok",
    "error": null,
    "model_size": "125M",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "pipeline_mp",
    "num_stages": 4,
    "num_chunks": 4,
    "throughput_samples_per_sec": 27.35,
    "peak_gpu_mem_gb": 4.095,
    "status": "ok",
    "error": null,
    "model_size": "1.3B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "pipeline_mp",
    "num_stages": 4,
    "num_chunks": 4,
    "throughput_samples_per_sec": 19.29,
    "peak_gpu_mem_gb": 7.655,
    "status": "ok",
    "error": null,
    "model_size": "2.7B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  },
  {
    "strategy": "pipeline_mp",
    "num_stages": 4,
    "num_chunks": 4,
    "throughput_samples_per_sec": 9.99,
    "peak_gpu_mem_gb": 18.236,
    "status": "ok",
    "error": null,
    "model_size": "6.7B",
    "num_gpus": 4,
    "batch_size": 4,
    "seq_len": 512,
    "gpu": "NVIDIA A100-SXM4-40GB"
  }
]
```
