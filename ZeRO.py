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
    gb = 0.0
    inner = getattr(engine.optimizer, "optimizer", engine.optimizer)
    for state in inner.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                gb += v.numel() * v.element_size() / 1e9
    return gb


def _profile_step(engine, local_rank: int, batch_size: int, seq_len: int) -> dict:
    def _make_batch():
        ids = torch.randint(0, 50257, (batch_size, seq_len), device=f"cuda:{local_rank}")
        return {"input_ids": ids, "labels": ids}

    m_base = _alloc_gb(local_rank)
    batch  = _make_batch()
    loss   = engine(**batch).loss
    m_fwd  = _alloc_gb(local_rank)
    engine.backward(loss)
    m_bwd  = _alloc_gb(local_rank)
    engine.step()
    m_step = _alloc_gb(local_rank)

    free, total    = torch.cuda.mem_get_info(local_rank)
    non_pytorch_gb = (total - free) / 1e9 - torch.cuda.memory_reserved(local_rank) / 1e9
    reserved_gb    = torch.cuda.memory_reserved(local_rank) / 1e9
    opt_state_gb   = _optimizer_state_gb(engine)

    return {
        "baseline_gb":          round(m_base,          3),
        "after_forward_gb":     round(m_fwd,           3),
        "after_backward_gb":    round(m_bwd,           3),
        "after_step_gb":        round(m_step,          3),
        "delta_activations_gb": round(m_fwd - m_base,  3),
        "delta_gradients_gb":   round(m_bwd - m_fwd,   3),
        "reserved_gb":          round(reserved_gb,     3),
        "non_pytorch_gb":       round(non_pytorch_gb,  3),
        "optimizer_states_gb":  round(opt_state_gb,    3),
    }


def _benchmark_engine(
    engine,
    local_rank: int,
    batch_size: int,
    seq_len:    int,
) -> tuple[float, float, dict]:
    def _make_batch():
        ids = torch.randint(0, 50257, (batch_size, seq_len), device=f"cuda:{local_rank}")
        return {"input_ids": ids, "labels": ids}

    for _ in range(WARMUP_STEPS):
        loss = engine(**_make_batch()).loss
        engine.backward(loss)
        engine.step()

    torch.cuda.synchronize(local_rank)
    mem_profile = _profile_step(engine, local_rank, batch_size, seq_len)
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

    peak_this_rank = torch.cuda.max_memory_allocated(local_rank) / 1e9
    peak_tensor    = torch.tensor(peak_this_rank, device=f"cuda:{local_rank}")
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_mem_gb = round(peak_tensor.item(), 3)

    return throughput, peak_mem_gb, mem_profile


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

        throughput, peak_mem, mem_profile = _benchmark_engine(engine, local_rank, batch_size, seq_len)

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