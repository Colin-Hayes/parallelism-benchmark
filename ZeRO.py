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
        # bf16 is preferred over fp16 on A100: no loss scaling needed, more
        # numerically stable, and DeepSpeed's BF16Optimizer skips the fp32
        # master-weight copy that fp16 mode maintains — saving ~4 bytes/param.
        "bf16": {"enabled": True},
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
        # DeepSpeed activation checkpointing replaces torch.utils.checkpoint.checkpoint
        # with a version that packs saved activations into a contiguous buffer
        # (contiguous_memory_optimization) to reduce fragmentation. partition_activations
        # further splits those buffers across data-parallel ranks so each GPU only
        # holds 1/world_size of the checkpointed activations.
        "activation_checkpointing": {
            "partition_activations":          True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing":              False,
        },
        # Per-step forward/backward/communication timing breakdown in DS logs.
        "wall_clock_breakdown": True,
        "steps_per_print": 10000,
    }

    if stage == 3:
        cfg["zero_optimization"].update({
            # overlap_comm prefetches the next parameter shard during current compute,
            # hiding all-gather latency at the cost of ~200 MB extra (two shards live
            # simultaneously). Safe now that sub_group_size=1e7 dropped peak from
            # 44 GB to 26.8 GB, leaving >13 GB headroom on 40 GB A100s.
            "overlap_comm": True,
            "contiguous_gradients": True,
            # 5e7 (50M params = 100 MB bf16) caps the all-gather prefetch cache.
            # bf16 saves ~4 bytes/param vs fp16 (no fp32 master weights) giving
            # more headroom, but 5e7 remains conservative and confirmed-safe.
            "stage3_max_live_parameters":       5e7,
            "stage3_max_reuse_distance":        5e7,
            "stage3_prefetch_bucket_size":      5e6,
            # Default sub_group_size=1e9 causes a ~10 GB spike during the optimizer
            # step: DeepSpeed materialises the fp32 shard + bf16 cast-back for 1B
            # params at once. 1e7 reduces that spike to ~120 MB at the cost of more
            # optimizer micro-steps.
            "sub_group_size":                   1e7,
        })

    return cfg


# ── Memory profiling helpers ──────────────────────────────────────────────────

def _alloc_gb(device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1e9

def _optimizer_state_gb(engine) -> float:
    """Actual GPU memory used by Adam m and v tensors on this rank."""
    gb = 0.0
    inner = getattr(engine.optimizer, "optimizer", engine.optimizer)
    for state in inner.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                gb += v.numel() * v.element_size() / 1e9
    return gb

def _profile_step(engine, local_rank: int, batch_size: int, seq_len: int) -> dict:
    """
    Instrument one training step to capture per-phase memory on rank 0.

    Runs after warmup so allocator and NCCL buffers are settled. Does NOT
    reset peak stats — the caller owns that so the timed section peak is clean.
    """
    def _make_batch():
        ids = torch.randint(0, 50257, (batch_size, seq_len), device=f"cuda:{local_rank}")
        return {"input_ids": ids, "labels": ids}

    m_base    = _alloc_gb(local_rank)

    batch = _make_batch()
    loss  = engine(**batch).loss
    m_fwd = _alloc_gb(local_rank)

    engine.backward(loss)
    m_bwd = _alloc_gb(local_rank)

    engine.step()
    m_step = _alloc_gb(local_rank)

    free, total = torch.cuda.mem_get_info(local_rank)
    non_pytorch_gb  = (total - free) / 1e9 - torch.cuda.memory_reserved(local_rank) / 1e9
    reserved_gb     = torch.cuda.memory_reserved(local_rank) / 1e9
    opt_state_gb    = _optimizer_state_gb(engine)

    return {
        "baseline_gb":          round(m_base,               3),
        "after_forward_gb":     round(m_fwd,                3),
        "after_backward_gb":    round(m_bwd,                3),
        "after_step_gb":        round(m_step,               3),
        "delta_activations_gb": round(m_fwd  - m_base,      3),
        "delta_gradients_gb":   round(m_bwd  - m_fwd,       3),
        "reserved_gb":          round(reserved_gb,          3),
        "non_pytorch_gb":       round(non_pytorch_gb,       3),
        "optimizer_states_gb":  round(opt_state_gb,         3),
    }


# ── Core benchmark loop ───────────────────────────────────────────────────────

def _benchmark_engine(
    engine,
    local_rank: int,
    batch_size: int,
    seq_len:    int,
) -> tuple[float, float, dict]:
    """
    Run WARMUP_STEPS warmup steps, one profiled step, then BENCH_STEPS timed steps.

    Memory stats are reset just before the timed section so warmup
    allocations don't inflate the peak reading.

    During ZeRO-3, all-gather ops temporarily materialise full parameter
    tensors on each GPU — torch.cuda.max_memory_allocated() captures that
    high-water mark, which is the number we care about.

    Returns
    -------
    (throughput_samples_per_sec, peak_mem_gb, mem_profile)
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

    # One instrumented step after warmup to capture per-phase memory breakdown.
    # Runs before peak reset so it doesn't affect the timed-section high-water mark.
    mem_profile = _profile_step(engine, local_rank, batch_size, seq_len)

    torch.cuda.reset_peak_memory_stats(local_rank)

    # Barrier ensures all ranks start timing together, so elapsed is not
    # inflated by one rank arriving late from previous work.
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        loss = engine(**_make_batch()).loss
        engine.backward(loss)
        engine.step()
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0

    # Total system throughput: samples processed by the entire cluster per second.
    # With DDP each GPU processes batch_size independent samples simultaneously.
    throughput = round((BENCH_STEPS * batch_size * dist.get_world_size()) / elapsed, 2)

    peak_this_rank = torch.cuda.max_memory_allocated(local_rank) / 1e9
    peak_tensor = torch.tensor(peak_this_rank, device=f"cuda:{local_rank}")
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_mem_gb = round(peak_tensor.item(), 3)

    return throughput, peak_mem_gb, mem_profile


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
        throughput_samples_per_sec   float or None  (total cluster samples/sec,
                                                     = batch_size * world_size * steps / elapsed)
        peak_gpu_mem_gb              float or None  (max across all ranks;
                                                     includes all-gather spikes for ZeRO-3)
        status                       "ok" | "OOM" | "error"
        error                        None or exception string
    """
    strategy = f"zero{stage}"

    try:
        ds_cfg = _ds_config(stage, batch_size)

        config = GPT2Config(
            vocab_size=50257,
            n_positions=seq_len,
            n_ctx=seq_len,
            use_cache=False,
            **model_cfg,
        )

        if stage == 3:
            with deepspeed.zero.Init(
                config_dict_or_path=ds_cfg,
                remote_device="cpu",
                pin_memory=True,
            ):
                model = GPT2LMHeadModel(config)
        else:
            model = GPT2LMHeadModel(config)

        model.gradient_checkpointing_enable()

        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_cfg,
        )

        throughput, peak_mem, mem_profile = _benchmark_engine(
            engine, local_rank, batch_size, seq_len
        )

        # Explicit teardown so the next strategy starts with a clean slate
        del engine, model
        torch.cuda.synchronize(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        return {
            "strategy":                   strategy,
            "throughput_samples_per_sec": throughput,
            "peak_gpu_mem_gb":            peak_mem,
            "mem_profile":                mem_profile,
            "status":                     "ok",
            "error":                      None,
        }

    except torch.cuda.OutOfMemoryError as e:
        try:
            del engine
        except NameError:
            pass
        try:
            del model
        except NameError:
            pass
        gc.collect()
        torch.cuda.synchronize(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return {
            "strategy":                   strategy,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "mem_profile":                None,
            "status":                     "OOM",
            "error":                      str(e),
        }

    except Exception as e:
        try:
            del engine
        except NameError:
            pass
        try:
            del model
        except NameError:
            pass
        gc.collect()
        torch.cuda.synchronize(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return {
            "strategy":                   strategy,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "mem_profile":                None,
            "status":                     "error",
            "error":                      str(e),
        }