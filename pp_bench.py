"""
pp_bench.py
-----------
Runs Pipeline Parallelism across model sizes using fairscale.nn.Pipe.
Launch with: python pp_bench.py --output PATH
(No torchrun — Pipe manages all GPUs from a single process)
"""
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

import argparse
import json
import torch
import torch.distributed as dist
from MP import run_pipeline_mp

MODEL_CONFIGS = {
    "125M": dict(n_layer=12, n_head=12,  n_embd=768),
    "1.3B": dict(n_layer=24, n_head=16,  n_embd=2048),
    "2.7B": dict(n_layer=32, n_head=32,  n_embd=2560),
    "6.7B": dict(n_layer=32, n_head=32,  n_embd=4096),
}
BATCH_SIZE = 4
SEQ_LEN    = 512
NUM_GPUS   = torch.cuda.device_count()


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
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # fairscale Pipe uses gloo internally for pipeline communication
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    configs    = {"125M_tiny": dict(n_layer=2, n_head=2, n_embd=64)} if args.dry_run else MODEL_CONFIGS
    batch_size = 1 if args.dry_run else BATCH_SIZE
    seq_len    = 16 if args.dry_run else SEQ_LEN

    print(f"Pipeline benchmark — {NUM_GPUS} GPUs, dry_run={args.dry_run}")

    all_results = []

    for size_name, model_cfg in configs.items():
        print(f"\nModel: {size_name}")
        result = run_pipeline_mp(
            model_cfg=model_cfg,
            batch_size=batch_size,
            seq_len=seq_len,
            local_rank=0,
            num_stages=NUM_GPUS,
            num_chunks=4,
        )
        result.update({
            "model_size": size_name,
            "num_gpus":   NUM_GPUS,
            "batch_size": batch_size,
            "seq_len":    seq_len,
            "gpu":        torch.cuda.get_device_name(0),
        })
        all_results.append(result)

        status = result["status"]
        tp     = result["throughput_samples_per_sec"]
        mem    = result["peak_gpu_mem_gb"]
        print(
            f"  status: {status} | "
            f"throughput: {tp if tp else 'N/A'} | "
            f"peak mem: {mem if mem else 'N/A'} GB"
        )

    save_results(args.output, all_results)
    print(f"\nResults written to {args.output}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()