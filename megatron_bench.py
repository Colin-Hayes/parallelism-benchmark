"""
megatron_bench.py
-----------------
Runs Megatron TP+PP across model sizes and parallel configurations.
Launch with: torchrun --nproc_per_node=4 megatron_bench.py --output PATH

Tests two layouts on 4 GPUs:
  TP=4, PP=1 — all 4 GPUs split each layer (no pipeline, 2 all-reduces/layer)
  TP=2, PP=2 — 2 TP ranks per stage, 2 pipeline stages (1F1B schedule)

Throughput is reported as per-GPU-equivalent samples/sec to match zero_bench.py.
"""
import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from Megatron import run_megatron

MODEL_CONFIGS = {
    "125M": dict(n_layer=12, n_head=12, n_embd=768),
    "1.3B": dict(n_layer=24, n_head=16, n_embd=2048),
    "2.7B": dict(n_layer=32, n_head=32, n_embd=2560),
    "6.7B": dict(n_layer=32, n_head=32, n_embd=4096),
}

# micro_batch_size × num_microbatches = global_batch_size
# Set to match zero_bench: 4 samples/GPU × 4 GPUs = 16 total
BATCH_SIZE       = 4    # micro-batch size per microbatch
SEQ_LEN          = 512
NUM_MICROBATCHES = 4    # global_batch = BATCH_SIZE * NUM_MICROBATCHES = 16

# (tp_size, pp_size) pairs — must satisfy tp × pp == world_size
PARALLEL_CONFIGS = [
    (4, 1),   # full tensor parallelism, no pipeline
    (2, 2),   # mixed TP + PP
]


def save_results(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.extend(records)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(seconds=60),
    )
    world_size = dist.get_world_size()

    configs          = {"125M_tiny": dict(n_layer=2, n_head=4, n_embd=64)} if args.dry_run else MODEL_CONFIGS
    batch_size       = 1 if args.dry_run else BATCH_SIZE
    seq_len          = 16 if args.dry_run else SEQ_LEN
    num_microbatches = 1 if args.dry_run else NUM_MICROBATCHES

    # Filter to configs that evenly divide world_size
    valid_parallel = [(tp, pp) for tp, pp in PARALLEL_CONFIGS if tp * pp == world_size]
    if not valid_parallel:
        if local_rank == 0:
            print(f"No valid TP/PP configs for world_size={world_size}. "
                  f"Expected tp*pp={world_size}, got {PARALLEL_CONFIGS}")
        dist.destroy_process_group()
        return

    all_results = []

    for size_name, model_cfg in configs.items():
        for tp_size, pp_size in valid_parallel:
            dist.barrier()
            result = run_megatron(
                tp_size=tp_size,
                pp_size=pp_size,
                model_cfg=model_cfg,
                batch_size=batch_size,
                seq_len=seq_len,
                local_rank=local_rank,
                num_microbatches=num_microbatches,
                debug=args.dry_run,
            )
            result.update({
                "model_size": size_name,
                "num_gpus":   world_size,
                "batch_size": batch_size,
                "seq_len":    seq_len,
                "gpu":        torch.cuda.get_device_name(local_rank),
            })
            if local_rank == 0:
                all_results.append(result)
                print(result)

    if local_rank == 0:
        save_results(args.output, all_results)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
