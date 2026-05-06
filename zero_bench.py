"""
zero_bench.py
-------------
Orchestrator for ZeRO Stage 0 and Stage 3 benchmarks across model sizes.

Each (stage, model_size, batch_size, seq_len) config runs in its own torchrun
subprocess so that CUDA/NCCL state from one run cannot inflate peak memory
readings in the next. Results are collected from per-config temp JSON files
and merged into --output.

Launch with: python zero_bench.py --output PATH [--nproc_per_node 4]
"""

import argparse
import json
import os
import random
import subprocess
import tempfile

MODEL_CONFIGS = {
    "125M": dict(n_layer=12, n_head=12,  n_embd=768),
    "1.3B": dict(n_layer=24, n_head=16,  n_embd=2048),
    "2.7B": dict(n_layer=32, n_head=32,  n_embd=2560),
    "6.7B": dict(n_layer=32, n_head=32,  n_embd=4096),
}

BATCH_SEQ_CONFIGS = {
    "125M": [(4, 512), (8, 512), (16, 512), (4, 1024), (8, 1024), (4, 2048)],
    "1.3B": [(4, 512), (8, 512), (4, 1024), (8, 1024)],
    "2.7B": [(4, 512), (8, 512), (4, 1024)],
    "6.7B": [(4, 512), (8, 512), (4, 1024)],
}

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero_run_config.py")


def _run_config(stage, model_size, batch_size, seq_len, nproc, dry_run):
    port = random.randint(20000, 40000)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name

    cmd = [
        "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        SCRIPT,
        "--stage",      str(stage),
        "--model_size", model_size,
        "--batch_size", str(batch_size),
        "--seq_len",    str(seq_len),
        "--output",     tmp,
    ]
    if dry_run:
        cmd.append("--dry_run")

    proc = subprocess.run(cmd, timeout=600)

    if os.path.exists(tmp):
        with open(tmp) as f:
            result = json.load(f)
        os.unlink(tmp)
    else:
        result = {
            "strategy":                   f"zero{stage}",
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "mem_profile":                None,
            "model_size":                 model_size,
            "batch_size":                 batch_size,
            "seq_len":                    seq_len,
            "status":                     "crash",
            "error":                      f"subprocess exited with code {proc.returncode}",
        }

    return result


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
    parser.add_argument("--output",         required=True)
    parser.add_argument("--dry_run",        action="store_true")
    parser.add_argument("--model_size",     choices=list(MODEL_CONFIGS), default=None,
                        help="Run a single model size instead of all")
    parser.add_argument("--nproc_per_node", type=int, default=4)
    args = parser.parse_args()

    sizes = [args.model_size] if args.model_size else list(MODEL_CONFIGS)
    all_results = []

    for model_size in sizes:
        batch_seq = [(1, 16)] if args.dry_run else BATCH_SEQ_CONFIGS[model_size]
        for stage in [0, 3]:
            for batch_size, seq_len in batch_seq:
                print(f"\n--- zero{stage} {model_size} bs={batch_size} seq={seq_len} ---", flush=True)
                result = _run_config(stage, model_size, batch_size, seq_len, args.nproc_per_node, args.dry_run)
                all_results.append(result)
                print(result, flush=True)

    save_results(args.output, all_results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
