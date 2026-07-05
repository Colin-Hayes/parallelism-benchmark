"""
ZeRO.py
-------
Benchmarks DeepSpeed ZeRO Stage 0 (DDP) and ZeRO Stage 3.
Called by zero_run_config.py — do not run directly.
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


def _ds_config(stage: int, batch_size: int) -> dict:
    cfg = {
        "train_batch_size": batch_size * dist.get_world_size(),
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4},
        },
        "zero_optimization": {"stage": stage},
    }
    return cfg


def _alloc_gb(device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1e9


def _optimizer_state_gb(engine) -> float:
    inner = getattr(engine.optimizer, "optimizer", engine.optimizer)
    gb, seen = 0.0, set()
    for group in inner.param_groups:
        for p in group["params"]:
            if isinstance(p, torch.Tensor) and p.is_cuda and id(p) not in seen:
                seen.add(id(p))
                gb += p.numel() * p.element_size() / 1e9
    for state in inner.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                gb += v.numel() * v.element_size() / 1e9
    return gb

def _param_gb(engine) -> float:
    """Resident parameter bytes on this rank (ZeRO-3: local shard via ds_tensor)."""
    gb = 0.0
    for p in engine.module.parameters():
        t = getattr(p, "ds_tensor", None)
        if t is not None:                       # ZeRO-3: partitioned param
            gb += t.numel() * t.element_size() / 1e9
        elif p.is_cuda:                          # ZeRO-0: full replica
            gb += p.numel() * p.element_size() / 1e9
    return gb

def _profile_step(engine, local_rank, batch_size, seq_len, context_floor_gb=None) -> dict:
    dev = local_rank
    def _make_batch():
        ids = torch.randint(0, 50257, (batch_size, seq_len), device=f"cuda:{dev}")
        return {"input_ids": ids, "labels": ids}

    m_base   = _alloc_gb(dev)
    param_gb = _param_gb(engine)
    opt_gb   = _optimizer_state_gb(engine)

    batch = _make_batch()
    loss  = engine(**batch).loss
    m_fwd = _alloc_gb(dev)

    # non_pytorch (context + NCCL + cuBLAS) measured same-instant, while buffers are warm.
    # It is ~constant across the step, so this value is combined with the timed-loop peak later.
    torch.cuda.synchronize(dev)
    free, total    = torch.cuda.mem_get_info(dev)
    reserved_now   = torch.cuda.memory_reserved(dev) / 1e9
    non_pytorch_gb = (total - free) / 1e9 - reserved_now
    capacity_gb    = total / 1e9

    engine.backward(loss)
    m_bwd = _alloc_gb(dev)
    engine.step()
    m_step = _alloc_gb(dev)

    prof = {
        # --- resident baseline, broken down ---
        "baseline_gb":          round(m_base, 3),
        "param_resident_gb":    round(param_gb, 3),
        "optimizer_states_gb":  round(opt_gb, 3),
        "baseline_other_gb":    round(max(m_base - param_gb - opt_gb, 0.0), 3),  # buffers, ds bookkeeping
        # --- transient behavior through the step ---
        "after_forward_gb":     round(m_fwd, 3),
        "after_backward_gb":    round(m_bwd, 3),
        "after_step_gb":        round(m_step, 3),
        "delta_activations_gb": round(m_fwd - m_base, 3),
        "delta_gradients_gb":   round(m_bwd - m_fwd, 3),
        # --- overhead outside PyTorch's allocator ---
        "non_pytorch_gb":       round(non_pytorch_gb, 3),
        "device_capacity_gb":   round(capacity_gb, 3),
        # peak_* and total_* filled in by _benchmark_engine from the timed loop
    }
    if context_floor_gb is not None:
        prof["cuda_context_gb"]  = round(context_floor_gb, 3)
        prof["comm_workspace_gb"] = round(max(non_pytorch_gb - context_floor_gb, 0.0), 3)  # NCCL + cuBLAS + misc
    return prof


def _benchmark_engine(engine, local_rank, batch_size, seq_len, context_floor_gb=None):
    def _make_batch():
        ids = torch.randint(0, 50257, (batch_size, seq_len), device=f"cuda:{local_rank}")
        return {"input_ids": ids, "labels": ids}

    for _ in range(WARMUP_STEPS):
        loss = engine(**_make_batch()).loss
        engine.backward(loss)
        engine.step()

    torch.cuda.synchronize(local_rank)
    mem_profile = _profile_step(engine, local_rank, batch_size, seq_len, context_floor_gb)
    torch.cuda.reset_peak_memory_stats(local_rank)

    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        loss = engine(**_make_batch()).loss
        engine.backward(loss)
        engine.step()
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0
    throughput = round((BENCH_STEPS * batch_size * dist.get_world_size()) / elapsed, 2)

    # True peaks over the timed loop; reduce raw values across ranks, then derive totals.
    peak_alloc = torch.cuda.max_memory_allocated(local_rank) / 1e9
    peak_resv  = torch.cuda.max_memory_reserved(local_rank) / 1e9
    np_gb      = mem_profile["non_pytorch_gb"]
    floor_gb   = context_floor_gb if context_floor_gb is not None else 0.0

    stats = torch.tensor([peak_alloc, peak_resv, np_gb, floor_gb],
                         device=f"cuda:{local_rank}")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    peak_alloc, peak_resv, np_gb, floor_gb = [round(x, 3) for x in stats.tolist()]

    mem_profile["peak_alloc_gb"]    = peak_alloc
    mem_profile["peak_reserved_gb"] = peak_resv
    mem_profile["non_pytorch_gb"]   = np_gb
    if context_floor_gb is not None:
        mem_profile["cuda_context_gb"]   = floor_gb
        mem_profile["comm_workspace_gb"] = round(max(np_gb - floor_gb, 0.0), 3)

    # The OOM-relevant footprint: what is actually committed on the device at peak.
    mem_profile["total_footprint_gb"]            = round(peak_resv + np_gb, 3)   # <-- races the 39.49 ceiling
    mem_profile["total_alloc_plus_nonpytorch_gb"] = round(peak_alloc + np_gb, 3) # your requested combo (lower bound)
    mem_profile["headroom_gb"] = round(mem_profile["device_capacity_gb"] - (peak_resv + np_gb), 3)

    return throughput, peak_alloc, mem_profile


def run_zero(
    stage:      int,
    model_cfg:  dict,
    batch_size: int,
    seq_len:    int,
    local_rank: int,
) -> dict:
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

        model = GPT2LMHeadModel(config)

        model.gradient_checkpointing_enable()

        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_cfg,
        )

        throughput, peak_mem, mem_profile = _benchmark_engine(
            engine, local_rank, batch_size, seq_len, context_floor_gb
        )

        del engine, model
        torch.cuda.synchronize(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        return {
            "strategy":                   strategy,
            "throughput_samples_per_sec": throughput,
            "peak_gpu_mem_gb":            peak_mem,                       
            "total_footprint_gb":         mem_profile["total_footprint_gb"], 
            "mem_profile":                mem_profile,
            "status":                     "ok",
            "error":                      None,
        }

    except torch.cuda.OutOfMemoryError as e:
        try:    del engine
        except NameError: pass
        try:    del model
        except NameError: pass
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
        try:    del engine
        except NameError: pass
        try:    del model
        except NameError: pass
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