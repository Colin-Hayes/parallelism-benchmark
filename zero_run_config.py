"""
zero_run_config.py
------------------
Runs a single (stage, model_size) ZeRO config inside a torchrun process group.
Writes the result dict as JSON to --output (rank 0 only), then exits cleanly.

Called by zero_bench.py — do not run directly.
"""

import argparse
import json
import os

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

import torch
import torch.distributed as dist

from ZeRO import run_zero

MODEL_CONFIGS = {
    "125M": dict(n_layer=12, n_head=12,  n_embd=768),
    "1.3B": dict(n_layer=24, n_head=16,  n_embd=2048),
    "2.7B": dict(n_layer=32, n_head=32,  n_embd=2560),
    "6.7B": dict(n_layer=32, n_head=32,  n_embd=4096),
    "10B":  dict(n_layer=32, n_head=40,  n_embd=5120),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      type=int, required=True, choices=[0, 3])
    parser.add_argument("--model_size", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len",    type=int, default=512)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    world_size = dist.get_world_size()

    if args.dry_run:
        model_cfg  = dict(n_layer=2, n_head=2, n_embd=64)
        batch_size = 1
        seq_len    = 16
    else:
        model_cfg  = MODEL_CONFIGS[args.model_size]
        batch_size = args.batch_size
        seq_len    = args.seq_len

    result = run_zero(args.stage, model_cfg, batch_size, seq_len, local_rank)
    result.update({
        "model_size": args.model_size,
        "num_gpus":   world_size,
        "batch_size": batch_size,
        "seq_len":    seq_len,
        "gpu":        torch.cuda.get_device_name(local_rank),
    })

    if local_rank == 0:
        print(result)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
