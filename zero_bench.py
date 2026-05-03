"""
zero_bench.py
-------------
Runs ZeRO Stage 0 and ZeRO Stage 3 across model sizes.
Launch with: torchrun --nproc_per_node=4 zero_bench.py --output PATH
"""
import argparse, json, os, socket
import torch
import torch.distributed as dist
from ZeRO import run_zero

MODEL_CONFIGS = {
    "125M": dict(n_layer=12, n_head=12,  n_embd=768),
    "1.3B": dict(n_layer=24, n_head=16,  n_embd=2048),
    "2.7B": dict(n_layer=32, n_head=32,  n_embd=2560),
    "6.7B": dict(n_layer=32, n_head=32,  n_embd=4096),
}
BATCH_SIZE = 4
SEQ_LEN    = 512

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
    parser.add_argument("--model_size", choices=list(MODEL_CONFIGS), default=None,
                        help="Run a single model size instead of all four")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    world_size = dist.get_world_size()

    if args.dry_run:
        configs = {"125M_tiny": dict(n_layer=2, n_head=2, n_embd=64)}
    elif args.model_size:
        configs = {args.model_size: MODEL_CONFIGS[args.model_size]}
    else:
        configs = MODEL_CONFIGS
    batch_size = 1 if args.dry_run else BATCH_SIZE
    seq_len    = 16 if args.dry_run else SEQ_LEN

    all_results = []

    for size_name, model_cfg in configs.items():
        for stage in [0, 3]:
            dist.barrier()
            result = run_zero(stage, model_cfg, batch_size, seq_len, local_rank)
            result.update({
                "model_size": size_name, "num_gpus": world_size,
                "batch_size": batch_size, "seq_len": seq_len,
                "gpu": torch.cuda.get_device_name(local_rank),
            })
            if local_rank == 0:
                all_results.append(result)
                print(result)

    if local_rank == 0:
        save_results(args.output, all_results)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()