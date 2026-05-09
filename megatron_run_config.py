"""
megatron_run_config.py
----------------------
Single-config runner for one (model_size, tp_size, pp_size) combination.
Launched by megatron_bench.py via torchrun. Do not run directly.

Writes a JSON result to --output from rank 0, then exits cleanly.
"""
import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from Megatron import run_megatron


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size",          type=int, required=True)
    parser.add_argument("--pp_size",          type=int, required=True)
    parser.add_argument("--n_layer",          type=int, required=True)
    parser.add_argument("--n_head",           type=int, required=True)
    parser.add_argument("--n_embd",           type=int, required=True)
    parser.add_argument("--model_size",       required=True)
    parser.add_argument("--batch_size",       type=int, required=True)
    parser.add_argument("--seq_len",          type=int, required=True)
    parser.add_argument("--num_microbatches", type=int, required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--dry_run",          action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(seconds=30),
    )
    dist.barrier()
    world_size = dist.get_world_size()

    model_cfg = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)

    result = run_megatron(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        model_cfg=model_cfg,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        local_rank=local_rank,
        num_microbatches=args.num_microbatches,
    )
    result.update({
        "model_size": args.model_size,
        "num_gpus":   world_size,
        "batch_size": args.batch_size,
        "seq_len":    args.seq_len,
        "gpu":        torch.cuda.get_device_name(local_rank),
    })

    if local_rank == 0:
        print(result, flush=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
