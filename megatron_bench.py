"""
megatron_bench.py
-----------------
Runs Megatron TP+PP across model sizes and parallel configurations.
Launch with: python megatron_bench.py --output PATH [--nproc_per_node 4]

Each (model_size, tp, pp) config is isolated in its own torchrun subprocess.
An NCCL crash or OOM in one config does not kill the benchmark — it is recorded
as status="crash" and the next config runs in a fresh process group.

Tests two layouts on 4 GPUs:
  TP=4, PP=1 — all 4 GPUs split each layer (no pipeline, 2 all-reduces/layer)
  TP=2, PP=2 — 2 TP ranks per stage, 2 pipeline stages (1F1B schedule)

Throughput is reported as per-GPU-equivalent samples/sec to match zero_bench.py.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

MODEL_CONFIGS = {
    "125M": dict(n_layer=12, n_head=12, n_embd=768),
    "1.3B": dict(n_layer=24, n_head=16, n_embd=2048),
    "2.7B": dict(n_layer=32, n_head=32, n_embd=2560),
    "6.7B": dict(n_layer=32, n_head=32, n_embd=4096),
}

# micro_batch_size × num_microbatches = global_batch_size
# Set to match zero_bench: 4 samples/GPU × 4 GPUs = 16 total
BATCH_SIZE       = 4
SEQ_LEN          = 512
NUM_MICROBATCHES = 4

# (tp_size, pp_size) pairs — must satisfy tp × pp == world_size
PARALLEL_CONFIGS = [
    (4, 1),   # full tensor parallelism, no pipeline
    (2, 2),   # mixed TP + PP
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SCRIPT = os.path.join(SCRIPT_DIR, "megatron_run_config.py")


def save_results(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.extend(records)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def run_one_config(tp_size, pp_size, size_name, model_cfg,
                   batch_size, seq_len, num_microbatches,
                   nproc, port, dry_run):
    """
    Spawn a torchrun subprocess for one (size_name, tp_size, pp_size) config.
    Returns the result dict, or a crash dict if the subprocess died unexpectedly.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        RUN_SCRIPT,
        "--tp_size",          str(tp_size),
        "--pp_size",          str(pp_size),
        "--n_layer",          str(model_cfg["n_layer"]),
        "--n_head",           str(model_cfg["n_head"]),
        "--n_embd",           str(model_cfg["n_embd"]),
        "--model_size",       size_name,
        "--batch_size",       str(batch_size),
        "--seq_len",          str(seq_len),
        "--num_microbatches", str(num_microbatches),
        "--output",           tmp_path,
    ]
    if dry_run:
        cmd.append("--dry_run")

    proc = subprocess.run(cmd, timeout=900)

    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        with open(tmp_path) as f:
            result = json.load(f)
        os.unlink(tmp_path)
    else:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        result = {
            "strategy":                   f"megatron_tp{tp_size}_pp{pp_size}",
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "status":                     "crash",
            "error":                      f"subprocess exited with code {proc.returncode}",
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",         required=True)
    parser.add_argument("--dry_run",        action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=4)
    args = parser.parse_args()

    world_size = args.nproc_per_node
    valid_parallel = [(tp, pp) for tp, pp in PARALLEL_CONFIGS if tp * pp == world_size]
    if not valid_parallel:
        print(f"No valid TP/PP configs for world_size={world_size}. "
              f"Expected tp*pp={world_size}, got {PARALLEL_CONFIGS}")
        return

    configs          = {"125M_tiny": dict(n_layer=2, n_head=4, n_embd=64)} if args.dry_run else MODEL_CONFIGS
    batch_size       = 1 if args.dry_run else BATCH_SIZE
    seq_len          = 16 if args.dry_run else SEQ_LEN
    num_microbatches = 1 if args.dry_run else NUM_MICROBATCHES

    all_results = []

    for size_name, model_cfg in configs.items():
        for tp_size, pp_size in valid_parallel:
            port = random.randint(20000, 40000)
            print(f"\n=== {size_name} TP={tp_size} PP={pp_size} (port {port}) ===",
                  flush=True)
            result = run_one_config(
                tp_size, pp_size, size_name, model_cfg,
                batch_size, seq_len, num_microbatches,
                world_size, port, args.dry_run,
            )
            print(result, flush=True)
            all_results.append(result)

    save_results(args.output, all_results)


if __name__ == "__main__":
    main()
