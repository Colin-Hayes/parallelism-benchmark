"""
profile_zero_memory.py
----------------------
Profiles GPU memory breakdown for ZeRO-3 on a single instrumented training step.

Reports measured deltas for each phase (forward, backward, optimizer step)
plus theoretical estimates for each component (params, gradients, optimizer states).

Run with:
    python -m torch.distributed.run --nproc_per_node=4 --master_port=29512 \
        profile_zero_memory.py [--model_size 6.7B]
"""

import os
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

import argparse
import gc
import torch
import torch.distributed as dist
import deepspeed
from transformers import GPT2Config, GPT2LMHeadModel

WARMUP_STEPS = 3

MODEL_CONFIGS = {
    "125M": dict(n_layer=12, n_head=12,  n_embd=768),
    "1.3B": dict(n_layer=24, n_head=16,  n_embd=2048),
    "2.7B": dict(n_layer=32, n_head=32,  n_embd=2560),
    "6.7B": dict(n_layer=32, n_head=32,  n_embd=4096),
}

BATCH_SIZE = 4
SEQ_LEN    = 512


def _ds_config(batch_size: int) -> dict:
    return {
        "train_batch_size": batch_size * dist.get_world_size(),
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm":                     False,
            "contiguous_gradients":             True,
            "stage3_max_live_parameters":       5e7,
            "stage3_max_reuse_distance":        5e7,
            "stage3_prefetch_bucket_size":      5e6,
            "sub_group_size":                   1e7,
        },
        "activation_checkpointing": {
            "partition_activations":          True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing":              False,
        },
        "steps_per_print": 10000,
    }


def _snap(device) -> dict:
    """Capture a memory snapshot at the current point."""
    torch.cuda.synchronize(device)
    return {
        "allocated": torch.cuda.memory_allocated(device) / 1e9,
        "reserved":  torch.cuda.memory_reserved(device)  / 1e9,
    }


def _non_pytorch_gb(device) -> float:
    """GPU memory used by NCCL, drivers, etc. — not visible to PyTorch allocator."""
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1e9 - torch.cuda.memory_reserved(device) / 1e9


def _optimizer_state_gb(engine) -> float:
    """Actual GPU memory used by Adam m and v tensors (from optimizer.state)."""
    gb = 0.0
    inner_opt = engine.optimizer.optimizer if hasattr(engine.optimizer, "optimizer") else engine.optimizer
    for param_state in inner_opt.state.values():
        for v in param_state.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                gb += v.numel() * v.element_size() / 1e9
    return gb


def _param_shard_gb(engine) -> float:
    """Actual GPU memory used by this rank's sharded bf16 parameters."""
    gb = 0.0
    for p in engine.module.parameters():
        if hasattr(p, "ds_tensor") and p.ds_tensor is not None and p.ds_tensor.is_cuda:
            gb += p.ds_tensor.numel() * p.ds_tensor.element_size() / 1e9
        elif p.is_cuda:
            gb += p.numel() * p.element_size() / 1e9
    return gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="6.7B", choices=list(MODEL_CONFIGS))
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    model_cfg = MODEL_CONFIGS[args.model_size]
    ds_cfg    = _ds_config(BATCH_SIZE)

    config = GPT2Config(
        vocab_size=50257,
        n_positions=SEQ_LEN,
        n_ctx=SEQ_LEN,
        use_cache=False,
        **model_cfg,
    )

    # ── Init ──────────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(local_rank)
    snap_before_init = _snap(local_rank)

    with deepspeed.zero.Init(config_dict_or_path=ds_cfg, remote_device="cpu", pin_memory=True):
        model = GPT2LMHeadModel(config)
    model.gradient_checkpointing_enable()

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg,
    )
    snap_after_init = _snap(local_rank)

    # ── Warmup ────────────────────────────────────────────────────────────────
    def _make_batch():
        ids = torch.randint(0, 50257, (BATCH_SIZE, SEQ_LEN), device=f"cuda:{local_rank}")
        return {"input_ids": ids, "labels": ids}

    for _ in range(WARMUP_STEPS):
        loss = engine(**_make_batch()).loss
        engine.backward(loss)
        engine.step()

    torch.cuda.synchronize(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    snap_warmup = _snap(local_rank)

    # ── Instrumented step ─────────────────────────────────────────────────────
    batch = _make_batch()

    loss = engine(**batch).loss
    snap_fwd = _snap(local_rank)

    engine.backward(loss)
    snap_bwd = _snap(local_rank)

    engine.step()
    snap_step = _snap(local_rank)

    peak_gb      = torch.cuda.max_memory_allocated(local_rank) / 1e9
    non_pt_gb    = _non_pytorch_gb(local_rank)
    opt_state_gb = _optimizer_state_gb(engine)
    param_shard  = _param_shard_gb(engine)

    # ── Theoretical estimates ─────────────────────────────────────────────────
    total_params = sum(
        p.ds_numel for p in engine.module.parameters() if hasattr(p, "ds_numel")
    )
    est_param_shard_gb = total_params * 2 / world_size / 1e9   # bf16
    est_grad_shard_gb  = total_params * 2 / world_size / 1e9   # bf16
    est_fp32_m_gb      = total_params * 4 / world_size / 1e9   # fp32
    est_fp32_v_gb      = total_params * 4 / world_size / 1e9   # fp32

    # ── Gather and print from rank 0 ──────────────────────────────────────────
    dist.barrier()
    if local_rank == 0:
        W = 62
        div = "─" * W
        print()
        print("=" * W)
        print(f"  ZeRO-3 Memory Profile  |  {args.model_size}  |  {world_size} GPUs  |  rank 0")
        print("=" * W)

        print(f"\n  {'MEASURED (torch.cuda.memory_allocated, rank 0)'}")
        print(f"  {div}")
        print(f"  {'After init   (params + optimizer states):':<45} {snap_after_init['allocated']:>6.2f} GB")
        print(f"  {'After warmup (settled baseline):':<45} {snap_warmup['allocated']:>6.2f} GB")
        print(f"  {'After forward  (+activations):':<45} {snap_fwd['allocated']:>6.2f} GB"
              f"  [Δ +{snap_fwd['allocated'] - snap_warmup['allocated']:.2f} GB]")
        print(f"  {'After backward (+gradients / reduce-scatter):':<45} {snap_bwd['allocated']:>6.2f} GB"
              f"  [Δ +{snap_bwd['allocated'] - snap_fwd['allocated']:.2f} GB]")
        print(f"  {'After optimizer step:':<45} {snap_step['allocated']:>6.2f} GB"
              f"  [Δ {snap_step['allocated'] - snap_bwd['allocated']:+.2f} GB]")
        print(f"  {'Peak (high-water mark):':<45} {peak_gb:>6.2f} GB")
        print(f"  {'Reserved (PyTorch allocator cache):':<45} {snap_step['reserved']:>6.2f} GB")
        print(f"  {'Non-PyTorch (NCCL comm buffers, drivers):':<45} {non_pt_gb:>6.2f} GB")

        print(f"\n  {'ACTUAL COMPONENT SIZES (from DeepSpeed internals, rank 0)'}")
        print(f"  {div}")
        print(f"  {'bf16 parameter shard:':<45} {param_shard:>6.2f} GB")
        print(f"  {'fp32 Adam states (m + v combined):':<45} {opt_state_gb:>6.2f} GB")

        print(f"\n  {'THEORETICAL ESTIMATES (rank 0 shard)'}")
        print(f"  {div}")
        print(f"  {f'bf16 param shard  ({total_params/1e9:.2f}B params / {world_size}):':<45} {est_param_shard_gb:>6.2f} GB")
        print(f"  {'bf16 grad shard:':<45} {est_grad_shard_gb:>6.2f} GB")
        print(f"  {'fp32 Adam m shard:':<45} {est_fp32_m_gb:>6.2f} GB")
        print(f"  {'fp32 Adam v shard:':<45} {est_fp32_v_gb:>6.2f} GB")
        floor = est_param_shard_gb + est_grad_shard_gb + est_fp32_m_gb + est_fp32_v_gb
        print(f"  {'Floor (params + grads + opt states):':<45} {floor:>6.2f} GB")

        print(f"\n  NOTE: backward Δ is often near zero for ZeRO-3 because")
        print(f"  gradients are reduce-scattered layer-by-layer and freed")
        print(f"  immediately — they never all exist in memory at once.")
        print("=" * W)
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
