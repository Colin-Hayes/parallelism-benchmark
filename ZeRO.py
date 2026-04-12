"""
ZeRO.py
-------
Benchmarks DeepSpeed ZeRO Stage 0 (DDP) and ZeRO Stage 3 for a given model
and batch configuration.

Returns throughput (samples/sec) and peak GPU memory (GB) per rank.
Peak memory is measured via torch.cuda.max_memory_allocated(), which captures
the high-water mark including temporary all-gather buffers in ZeRO-3.

Called by benchmark.py — do not run directly.
"""

import os
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

import gc
import time

import torch
import torch.distributed as dist
import deepspeed
from transformers import GPT2Config, GPT2LMHeadModel

WARMUP_STEPS = 5
BENCH_STEPS  = 20


# ── DeepSpeed config builders ─────────────────────────────────────────────────

def _ds_config(stage: int, batch_size: int) -> dict:
    """
    Build a minimal DeepSpeed config for the given ZeRO stage.

    Stage 0 : DeepSpeed wrapper around standard DDP — no sharding at all.
              Gradients are all-reduced exactly like vanilla DDP.
    Stage 3 : Full sharding of weights, gradients, and optimizer states.
              Parameters are all-gathered on demand during forward/backward,
              which is where the peak memory spike we measure comes from.
    """
    cfg = {
        "train_batch_size": batch_size * dist.get_world_size(),
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps":   1e-8,
                "weight_decay": 0.0,
            },
        },
        "zero_optimization": { "stage": stage },
        "steps_per_print": 10000,
    }

    if stage == 3:
        # Prefetch parameters during forward/backward to overlap communication
        # with compute — this is the standard fair-comparison setting.
        cfg["zero_optimization"].update({
            "stage3_prefetch_bucket_size":        5e7,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters":         1e9,
            "stage3_max_reuse_distance":          1e9,
            "overlap_comm":                       False,
        })

    return cfg


# ── Core benchmark loop ───────────────────────────────────────────────────────

def _benchmark_engine(
    engine,
    local_rank: int,
    batch_size: int,
    seq_len:    int,
) -> tuple[float, float]:
    """
    Run WARMUP_STEPS warmup steps, then BENCH_STEPS timed steps.

    Memory stats are reset just before the timed section so warmup
    allocations don't inflate the peak reading.

    During ZeRO-3, all-gather ops temporarily materialise full parameter
    tensors on each GPU — torch.cuda.max_memory_allocated() captures that
    high-water mark, which is the number we care about.

    Returns
    -------
    (throughput_samples_per_sec, peak_mem_gb)
    """
    def _make_batch():
        ids = torch.randint(
            0, 50257,
            (batch_size, seq_len),
            device=f"cuda:{local_rank}",
        )
        return {"input_ids": ids, "labels": ids}

    # Warmup — let CUDA kernels and DeepSpeed comm buffers settle
    for _ in range(WARMUP_STEPS):
        loss = engine(**_make_batch()).loss
        engine.backward(loss)
        engine.step()

    torch.cuda.synchronize(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)

    # Timed section
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        loss = engine(**_make_batch()).loss
        engine.backward(loss)
        engine.step()
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0

    throughput  = round((BENCH_STEPS * batch_size) / elapsed, 2)
    
    peak_this_rank = torch.cuda.max_memory_allocated(local_rank) / 1e9
    peak_tensor = torch.tensor(peak_this_rank, device=f"cuda:{local_rank}")
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_mem_gb = round(peak_tensor.item(), 3)

    return throughput, peak_mem_gb


# ── Public entry point ────────────────────────────────────────────────────────

def run_zero(
    stage:      int,
    model_cfg:  dict,
    batch_size: int,
    seq_len:    int,
    local_rank: int,
) -> dict:
    """
    Initialise a GPT-2-style model with DeepSpeed ZeRO stage `stage` and
    run the benchmark loop.

    Parameters
    ----------
    stage      : 0 (DDP-equivalent) or 3 (full sharding)
    model_cfg  : GPT2Config kwargs, e.g. {"n_layer": 12, "n_embd": 768}
    batch_size : per-GPU micro-batch size
    seq_len    : sequence length in tokens
    local_rank : CUDA device index for this process

    Returns
    -------
    dict with keys:
        strategy                     "zero0" or "zero3"
        throughput_samples_per_sec   float or None
        peak_gpu_mem_gb_rank0        float or None  (high-water mark on rank 0,
                                                     includes all-gather spikes)
        status                       "ok" | "OOM" | "error"
        error                        None or exception string
    """
    strategy = f"zero{stage}"

    try:
        model = GPT2LMHeadModel(GPT2Config(vocab_size=50257, **model_cfg))

        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=_ds_config(stage, batch_size),
        )

        throughput, peak_mem = _benchmark_engine(
            engine, local_rank, batch_size, seq_len
        )

        # Explicit teardown so the next strategy starts with a clean slate
        del engine, model
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "strategy":                   strategy,
            "throughput_samples_per_sec": throughput,
            "peak_gpu_mem_gb_rank0":      peak_mem,
            "status":                     "ok",
            "error":                      None,
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "strategy":                   strategy,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb_rank0":      None,
            "status":                     "OOM",
            "error":                      str(e),
        }

    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "strategy":                   strategy,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb_rank0":      None,
            "status":                     "error",
            "error":                      str(e),
        }