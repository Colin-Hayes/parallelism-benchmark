"""
Megatron.py
-----------
Benchmarks Megatron-style Tensor Parallelism + Pipeline Parallelism.
Called by megatron_run_config.py — do not run directly.
"""

import gc
import math
import time

import torch
import torch.distributed as dist

import megatron.core.parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt import GPTModel

WARMUP_STEPS = 5
BENCH_STEPS  = 20


def _get_layer_spec():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    return get_gpt_layer_local_spec()


def _build_model(model_cfg: dict, seq_len: int) -> GPTModel:
    pp = mpu.get_pipeline_model_parallel_world_size()
    layers_per_stage = model_cfg["n_layer"] // pp
    config = TransformerConfig(
        num_layers=layers_per_stage,
        hidden_size=model_cfg["n_embd"],
        num_attention_heads=model_cfg["n_head"],
        ffn_hidden_size=4 * model_cfg["n_embd"],
        use_cpu_initialization=True,
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        add_bias_linear=True,
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        gradient_accumulation_fusion=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )

    tp = mpu.get_tensor_model_parallel_world_size()
    vocab_size = math.ceil(50257 / tp) * tp

    model = GPTModel(
        config=config,
        transformer_layer_spec=_get_layer_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    return model.cuda().bfloat16()


class _MasterWeightOptimizer:
    """bf16 compute, fp32 master + fp32 Adam, grads accumulated directly into
    persistent fp32 main_grad buffers (no bf16 grad copy kept). Requires torch>=2.1."""

    def __init__(self, model: torch.nn.Module, lr: float = 1e-4, immediate: bool = False):
        self._immediate = immediate
        self._bf16 = [p for p in model.parameters() if p.requires_grad]
        self._fp32 = [p.detach().float().clone() for p in self._bf16]
        for pf in self._fp32:
            pf.requires_grad_(True)
        
        if immediate:
            self._opts = [torch.optim.AdamW([pf], lr=lr) for pf in self._fp32]
            self._handles = [
                pb.register_post_accumulate_grad_hook(self._make_immediate_hook(i))
                for i, pb in enumerate(self._bf16)
            ]
        else:
            for pf in self._fp32:
                pf.grad = torch.zeros_like(pf)        
            self._opt = torch.optim.AdamW(self._fp32, lr=lr)
            self._handles = [
                pb.register_post_accumulate_grad_hook(self._make_hook(pf))
                for pb, pf in zip(self._bf16, self._fp32)
            ]

    @staticmethod
    def _make_hook(pf):
        def hook(pb):
            pf.grad.add_(pb.grad.float())   # fold into fp32 main_grad
            pb.grad = None                  # free bf16 grad right away
        return hook
    
    def _make_immediate_hook(self, idx):
        def hook(pb):
            pf = self._fp32[idx]
            pf.grad = pb.grad.float()       # transient fp32 grad, this parameter only
            pb.grad = None                  # free bf16 grad right away
            self._opts[idx].step()          # applies the AdamW update to pf in place
            pf.grad = None                  # drop the transient fp32 grad — nothing more needs it
            with torch.no_grad():
                pb.data.copy_(pf.data)      # bf16 weight <- updated fp32 master, right away
        return hook

    def zero_grad(self) -> None:
        if self._immediate:
            return   # each hook already frees its own grad the moment it's consumed
        for pb in self._bf16:
            pb.grad = None
        for pf in self._fp32:
            if pf.grad is not None:
                pf.grad.zero_()             # keep buffer, reset values

    def step(self) -> None:
        if self._immediate:
            return   # every parameter was already updated by its hook during backward
        self._opt.step()
        with torch.no_grad():
            for p_bf16, p_fp32 in zip(self._bf16, self._fp32):
                p_bf16.data.copy_(p_fp32.data)
        self.zero_grad()

    @property
    def state(self):
        if self._immediate:
            merged = {}
            for opt in self._opts:
                merged.update(opt.state)
            return merged
        return self._opt.state

    @property
    def fp32_params(self): return self._fp32


class _DataIterator:
    def __init__(self, batch_size: int, seq_len: int, local_rank: int, vocab_size: int):
        self.batch_size = batch_size
        self.seq_len    = seq_len
        self.device     = f"cuda:{local_rank}"
        self.vocab_size = vocab_size

    def __iter__(self):
        return self

    def __next__(self):
        ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        return {"input_ids": ids, "labels": ids}


def _make_forward_step(seq_len: int):
    def forward_step(data_iterator, model):
        data      = next(data_iterator)
        input_ids = data["input_ids"]
        labels    = data["labels"] if mpu.is_pipeline_last_stage() else None

        if mpu.is_pipeline_first_stage():
            position_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
        else:
            input_ids    = None
            position_ids = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
            )

        def loss_func(output_tensor):
            loss = output_tensor.mean()
            return loss, {"loss": loss.detach()}

        return output, loss_func

    return forward_step


def _alloc_gb(device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1e9


def _optimizer_state_gb(opt: _MasterWeightOptimizer) -> float:
    gb = 0.0
    for p in opt.fp32_params:
        if p.is_cuda:
            gb += p.numel() * p.element_size() / 1e9
    for state in opt.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                gb += v.numel() * v.element_size() / 1e9
    return gb


def _param_gb(model) -> float:
    return sum(p.numel() * p.element_size() / 1e9 for p in model.parameters() if p.is_cuda)


def _fwd_bwd(forward_backward_func, forward_step, data_iter, model,
             num_microbatches, seq_len, batch_size):
    forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iter,
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=seq_len,
        micro_batch_size=batch_size,
        forward_only=False,
    )


def _profile_step(model, opt, forward_backward_func, forward_step,
                  data_iter, local_rank, batch_size, seq_len, num_microbatches,
                  context_floor_gb=None) -> dict:
    dev = local_rank
    m_base   = _alloc_gb(dev)
    param_gb = _param_gb(model)                 # bf16 param shard (TP-sharded)
    opt_gb   = _optimizer_state_gb(opt)         # fp32 master + Adam (over the shard)

    _fwd_bwd(forward_backward_func, forward_step, data_iter, model,
             num_microbatches, seq_len, batch_size)
    m_fwdbwd = _alloc_gb(dev)

    torch.cuda.synchronize(dev)
    free, total    = torch.cuda.mem_get_info(dev)
    reserved_now   = torch.cuda.memory_reserved(dev) / 1e9
    non_pytorch_gb = (total - free) / 1e9 - reserved_now
    capacity_gb    = total / 1e9

    opt.step()                                   # zero_grad() called inside
    m_step = _alloc_gb(dev)

    prof = {
        "baseline_gb":          round(m_base, 3),
        "param_resident_gb":    round(param_gb, 3),   # renamed from param_shard_gb for cross-method parity
        "optimizer_states_gb":  round(opt_gb, 3),
        "baseline_other_gb":    round(max(m_base - param_gb - opt_gb, 0.0), 3),
        "after_fwd_bwd_gb":     round(m_fwdbwd, 3),
        "after_step_gb":        round(m_step, 3),
        "delta_fwd_bwd_gb":     round(m_fwdbwd - m_base, 3),
        "delta_step_gb":        round(m_step - m_fwdbwd, 3),
        "non_pytorch_gb":       round(non_pytorch_gb, 3),
        "device_capacity_gb":   round(capacity_gb, 3),
    }
    if context_floor_gb is not None:
        prof["cuda_context_gb"]   = round(context_floor_gb, 3)
        prof["comm_workspace_gb"] = round(max(non_pytorch_gb - context_floor_gb, 0.0), 3)
    return prof


def _benchmark_megatron(model, local_rank, batch_size, seq_len, num_microbatches, vocab_size, context_floor_gb=None):
    forward_backward_func = get_forward_backward_func()
    forward_step          = _make_forward_step(seq_len)
    opt                   = _MasterWeightOptimizer(model, immediate=(num_microbatches == 1))
    data_iter             = iter(_DataIterator(batch_size, seq_len, local_rank, vocab_size))

    def _step():
        _fwd_bwd(forward_backward_func, forward_step, data_iter, model, num_microbatches, seq_len, batch_size)
        opt.step()  # zero_grad called inside step()

    for _ in range(WARMUP_STEPS):
        _step()

    torch.cuda.synchronize(local_rank)
    mem_profile = _profile_step(
        model, opt, forward_backward_func, forward_step,
        data_iter, local_rank, batch_size, seq_len, num_microbatches, context_floor_gb,
    )
    torch.cuda.reset_peak_memory_stats(local_rank)

    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        _step()
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0
    throughput = round((BENCH_STEPS * batch_size * num_microbatches) / elapsed, 2)

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

    mem_profile["total_footprint_gb"]             = round(peak_resv + np_gb, 3)
    mem_profile["total_alloc_plus_nonpytorch_gb"] = round(peak_alloc + np_gb, 3)
    mem_profile["headroom_gb"] = round(mem_profile["device_capacity_gb"] - (peak_resv + np_gb), 3)

    return throughput, peak_alloc, mem_profile


def run_megatron(tp_size, pp_size, model_cfg, batch_size, seq_len, local_rank, num_microbatches=4, context_floor_gb=None):
    strategy = f"megatron_tp{tp_size}_pp{pp_size}"
    model    = None

    try:
        try:
            mpu.destroy_model_parallel()
        except Exception:
            pass

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )

        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(42)

        model      = _build_model(model_cfg, seq_len)
        vocab_size = math.ceil(50257 / tp_size) * tp_size
        dp_size    = mpu.get_data_parallel_world_size()

        throughput, peak_mem, mem_profile = _benchmark_megatron(
            model, local_rank, batch_size, seq_len, num_microbatches, vocab_size, context_floor_gb
        )

        del model
        torch.cuda.synchronize(local_rank)
        mpu.destroy_model_parallel()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        return {
            "strategy":                   strategy,
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "effective_global_batch":     batch_size * num_microbatches * dp_size,
            "throughput_samples_per_sec": throughput,
            "peak_gpu_mem_gb":            peak_mem,
            "total_footprint_gb": mem_profile["total_footprint_gb"],
            "mem_profile":                mem_profile,
            "status":                     "ok",
            "error":                      None,
        }

    except torch.cuda.OutOfMemoryError as e:
        # --- snapshot BEFORE any cleanup: comm buffers + context are still resident here ---
        try:
            torch.cuda.synchronize(local_rank)
        except Exception:
            pass
        try:
            free, total    = torch.cuda.mem_get_info(local_rank)
            in_use_gb      = (total - free) / 1e9
            reserved_gb    = torch.cuda.memory_reserved(local_rank) / 1e9
            allocated_gb   = torch.cuda.memory_allocated(local_rank) / 1e9
            non_pytorch_gb = in_use_gb - reserved_gb
            floor          = context_floor_gb if context_floor_gb is not None else 0.0
            oom_profile = {
                "captured_at":        "oom",           # rank-local; process group may be dead
                "oom_rank":           local_rank,
                "device_capacity_gb": round(total / 1e9, 3),
                "in_use_gb":          round(in_use_gb, 3),        # driver view: everything on the card
                "torch_reserved_gb":  round(reserved_gb, 3),
                "torch_allocated_gb": round(allocated_gb, 3),
                "non_pytorch_gb":     round(non_pytorch_gb, 3),   # context + NCCL + cuBLAS
                "cuda_context_gb":    round(floor, 3),
                "comm_workspace_gb":  round(max(non_pytorch_gb - floor, 0.0), 3),  # NCCL + cuBLAS + misc
            }
        except Exception as snap_err:
            oom_profile = {"captured_at": "oom", "snapshot_error": str(snap_err)}

       

        return {
            "strategy":                   strategy,
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "mem_profile":                oom_profile,   # was None — now carries the OOM snapshot
            "status":                     "OOM",
            "error":                      str(e),
        }

    except Exception as e:
        return {
            "strategy":                   strategy,
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "mem_profile":                None,
            "status":                     "error",
            "error":                      str(e),
        }
