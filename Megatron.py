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
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.distributed import DistributedDataParallelConfig

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
        recompute_num_layers=layers_per_stage,
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
    return model.cuda()


def _make_optimizer(model):
    # get_megatron_optimizer requires ddp_config on the model; with DP=1 we
    # don't wrap in Megatron DDP but must set the attribute manually.
    if not hasattr(model, "ddp_config"):
        model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    optim_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        bf16=True,
        fp16=False,
        use_distributed_optimizer=False,
    )
    return get_megatron_optimizer(optim_config, [model])


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


def _optimizer_state_gb(opt) -> float:
    gb = 0.0
    # fp32 master params (bf16 model → fp32 master copy in BF16Optimizer)
    for attr in ("fp32_from_bf16_groups", "fp32_from_float16_groups"):
        for group in getattr(opt, attr, []):
            for p in group:
                if isinstance(p, torch.Tensor) and p.is_cuda:
                    gb += p.numel() * p.element_size() / 1e9
    # fp32 Adam m and v states in the underlying optimizer
    inner = getattr(opt, "optimizer", opt)
    for state in inner.state.values():
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
                  data_iter, local_rank, batch_size, seq_len, num_microbatches) -> dict:
    m_base = _alloc_gb(local_rank)

    opt.zero_grad()
    _fwd_bwd(forward_backward_func, forward_step, data_iter, model, num_microbatches, seq_len, batch_size)
    m_fwdbwd = _alloc_gb(local_rank)

    opt.step()
    m_step = _alloc_gb(local_rank)

    free, total    = torch.cuda.mem_get_info(local_rank)
    non_pytorch_gb = (total - free) / 1e9 - torch.cuda.memory_reserved(local_rank) / 1e9
    reserved_gb    = torch.cuda.memory_reserved(local_rank) / 1e9

    return {
        "baseline_gb":         round(m_base,             3),
        "after_fwd_bwd_gb":    round(m_fwdbwd,           3),
        "after_step_gb":       round(m_step,             3),
        "delta_fwd_bwd_gb":    round(m_fwdbwd - m_base,  3),
        "delta_step_gb":       round(m_step - m_fwdbwd,  3),
        "reserved_gb":         round(reserved_gb,        3),
        "non_pytorch_gb":      round(non_pytorch_gb,     3),
        "param_shard_gb":      round(_param_gb(model),   3),
        "optimizer_states_gb": round(_optimizer_state_gb(opt), 3),
    }


def _benchmark_megatron(model, local_rank, batch_size, seq_len, num_microbatches, vocab_size):
    forward_backward_func = get_forward_backward_func()
    forward_step          = _make_forward_step(seq_len)
    opt                   = _make_optimizer(model)
    data_iter             = iter(_DataIterator(batch_size, seq_len, local_rank, vocab_size))

    def _step():
        opt.zero_grad()
        _fwd_bwd(forward_backward_func, forward_step, data_iter, model, num_microbatches, seq_len, batch_size)
        opt.step()

    for _ in range(WARMUP_STEPS):
        _step()

    torch.cuda.synchronize(local_rank)
    mem_profile = _profile_step(
        model, opt, forward_backward_func, forward_step,
        data_iter, local_rank, batch_size, seq_len, num_microbatches,
    )
    torch.cuda.reset_peak_memory_stats(local_rank)

    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        _step()
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0

    throughput = round((BENCH_STEPS * batch_size * num_microbatches) / elapsed, 2)

    peak_this_rank = torch.cuda.max_memory_allocated(local_rank) / 1e9
    peak_tensor    = torch.tensor(peak_this_rank, device=f"cuda:{local_rank}")
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_mem_gb    = round(peak_tensor.item(), 3)

    return throughput, peak_mem_gb, mem_profile


def run_megatron(tp_size, pp_size, model_cfg, batch_size, seq_len, local_rank, num_microbatches=4):
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

        throughput, peak_mem, mem_profile = _benchmark_megatron(
            model, local_rank, batch_size, seq_len, num_microbatches, vocab_size
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
            "throughput_samples_per_sec": throughput,
            "peak_gpu_mem_gb":            peak_mem,
            "mem_profile":                mem_profile,
            "status":                     "ok",
            "error":                      None,
        }

    except torch.cuda.OutOfMemoryError as e:
        if model is not None:
            del model
        try:
            mpu.destroy_model_parallel()
        except Exception:
            pass
        gc.collect()
        torch.cuda.synchronize(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return {
            "strategy":                   strategy,
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "mem_profile":                None,
            "status":                     "OOM",
            "error":                      str(e),
        }

    except Exception as e:
        if model is not None:
            del model
        try:
            mpu.destroy_model_parallel()
        except Exception:
            pass
        gc.collect()
        torch.cuda.synchronize(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
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
