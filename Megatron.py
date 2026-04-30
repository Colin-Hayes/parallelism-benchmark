"""
Megatron.py
-----------
Benchmarks Megatron-style Tensor Parallelism (TP) + Pipeline Parallelism (PP)
using megatron-core.

TP splits each transformer layer's weight matrices across TP ranks:
  - QKV projection: ColumnParallel — each rank holds n_head/TP heads
  - Output projection: RowParallel — each rank holds hidden/TP columns
  - FFN up: ColumnParallel, FFN down: RowParallel
  - Communication: 2 all-reduces per layer during forward, 2 during backward,
    on activation-sized tensors (batch × seq × hidden/TP). This is far smaller
    than ZeRO-3's all-gather of full weight tensors (hidden × hidden per layer).

PP assigns num_layers/PP transformer blocks to each GPU using a 1F1B schedule:
  - Pipeline bubble fraction: (p-1) / (m + p-1)
    e.g. p=2 stages, m=4 microbatches → 1/5 = 20% bubble
         p=4 stages, m=4 microbatches → 3/7 = 43% bubble

Process group layout for 4 GPUs:
  TP=4, PP=1: ranks [0,1,2,3] are one TP group, one pipeline stage
  TP=2, PP=2: ranks [0,1] are TP group for stage 0,
              ranks [2,3] are TP group for stage 1

Called by megatron_bench.py — do not run directly.
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


# ── Layer spec ────────────────────────────────────────────────────────────────

def _get_layer_spec():
    """Return the GPT transformer layer spec using local (non-TransformerEngine) ops."""
    try:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        return get_gpt_layer_local_spec()
    except (ImportError, TypeError):
        # Older megatron-core versions export a different name
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        return get_gpt_layer_with_transformer_engine_spec()


# ── Model construction ────────────────────────────────────────────────────────

def _build_model(model_cfg: dict, seq_len: int) -> GPTModel:
    """
    Build the GPT model shard owned by this rank.

    megatron-core partitions layers across PP ranks automatically when
    initialize_model_parallel has been called. pre_process / post_process
    flags tell each rank whether it holds the embedding / LM-head.
    """
    config = TransformerConfig(
        num_layers=model_cfg["n_layer"],
        hidden_size=model_cfg["n_embd"],
        num_attention_heads=model_cfg["n_head"],
        ffn_hidden_size=4 * model_cfg["n_embd"],
        use_cpu_initialization=True,   # shard params at init, avoid GPU OOM
        fp16=True,
        params_dtype=torch.float16,
        pipeline_dtype=torch.float16,
        add_bias_linear=True,
        # Disable fused kernels — they require TransformerEngine / APEX
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        gradient_accumulation_fusion=False,
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

    return model.cuda().half()


# ── Data iterator ─────────────────────────────────────────────────────────────

class _DataIterator:
    """
    Infinite iterator yielding random token batches.

    Called once per microbatch by the pipeline schedule on every rank.
    Non-first-stage ranks ignore input_ids (they receive hidden states from
    the previous stage via pipeline recv). Non-last-stage ranks ignore labels.
    Advancing in lockstep across all ranks keeps the iterator consistent.
    """

    def __init__(self, batch_size: int, seq_len: int, local_rank: int):
        self.batch_size = batch_size
        self.seq_len    = seq_len
        self.device     = f"cuda:{local_rank}"

    def __iter__(self):
        return self

    def __next__(self):
        ids = torch.randint(0, 50257, (self.batch_size, self.seq_len), device=self.device)
        return {"input_ids": ids, "labels": ids}


# ── Forward step for the pipeline schedule ────────────────────────────────────

def _make_forward_step(seq_len: int):
    """
    Return a forward_step function compatible with megatron-core's pipeline schedule.

    The schedule calls this once per microbatch per forward pass. It expects:
      (output_tensor, loss_func)

    On non-last stages:  output_tensor = hidden states sent to next stage
    On the last stage:   output_tensor = loss scalar; loss_func wraps it
    """
    def forward_step(data_iterator, model):
        data         = next(data_iterator)
        input_ids    = data["input_ids"]
        labels       = data["labels"] if mpu.is_pipeline_last_stage() else None
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )

        # On non-first stages, the model ignores input_ids and reads hidden
        # states that the pipeline schedule placed via model.set_input_tensor().
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,   # megatron uses causal mask internally
            labels=labels,
        )

        def loss_func(output_tensor):
            return output_tensor, {"loss": output_tensor.detach()}

        return output, loss_func

    return forward_step


# ── Core benchmark loop ───────────────────────────────────────────────────────

def _benchmark_megatron(
    model,
    local_rank:      int,
    batch_size:      int,
    seq_len:         int,
    num_microbatches: int,
) -> tuple[float, float]:
    """
    WARMUP_STEPS warmup steps then BENCH_STEPS timed steps using the 1F1B schedule.

    Throughput is reported as global_samples / elapsed / world_size so it is
    directly comparable to ZeRO's per-GPU-equivalent metric.

    peak_mem is the max across all ranks (captures pipeline stage imbalance).
    """
    forward_backward_func = get_forward_backward_func()
    forward_step          = _make_forward_step(seq_len)
    opt                   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    data_iter             = iter(_DataIterator(batch_size, seq_len, local_rank))

    def _step():
        opt.zero_grad(set_to_none=True)
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iter,
            model=[model],
            num_microbatches=num_microbatches,
            seq_length=seq_len,
            micro_batch_size=batch_size,
            forward_only=False,
        )
        opt.step()

    for _ in range(WARMUP_STEPS):
        _step()

    torch.cuda.synchronize(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)

    # Barrier ensures all ranks start timing together.
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        _step()
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0

    # Total system throughput: all GPUs cooperate on batch_size * num_microbatches
    # samples per step — directly comparable to ZeRO's batch_size * world_size.
    global_samples_per_step = batch_size * num_microbatches
    throughput = round((BENCH_STEPS * global_samples_per_step) / elapsed, 2)

    peak_this_rank = torch.cuda.max_memory_allocated(local_rank) / 1e9
    peak_tensor    = torch.tensor(peak_this_rank, device=f"cuda:{local_rank}")
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    peak_mem_gb    = round(peak_tensor.item(), 3)

    return throughput, peak_mem_gb


# ── Public entry point ────────────────────────────────────────────────────────

def run_megatron(
    tp_size:          int,
    pp_size:          int,
    model_cfg:        dict,
    batch_size:       int,
    seq_len:          int,
    local_rank:       int,
    num_microbatches: int = 4,
) -> dict:
    """
    Initialise megatron-core TP+PP process groups, build a GPT shard on this
    rank, and run the benchmark loop.

    Parameters
    ----------
    tp_size          : tensor-parallel degree (splits each layer across TP GPUs)
    pp_size          : pipeline-parallel degree (splits layers across PP stages)
    model_cfg        : GPT architecture kwargs (n_layer, n_embd, n_head)
    batch_size       : micro-batch size per microbatch
    seq_len          : sequence length in tokens
    local_rank       : CUDA device index for this process
    num_microbatches : microbatches per training step (hides pipeline bubble)

    tp_size × pp_size must equal world_size (no data parallelism).

    Returns
    -------
    dict with keys:
        strategy                     e.g. "megatron_tp4_pp1"
        tp_size, pp_size, num_microbatches
        throughput_samples_per_sec   float or None
        peak_gpu_mem_gb              float or None (max across all ranks)
        status                       "ok" | "OOM" | "error"
        error                        None or exception string
    """
    strategy = f"megatron_tp{tp_size}_pp{pp_size}"
    model    = None

    try:
        # Reset any existing parallel state from a previous run
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

        model = _build_model(model_cfg, seq_len)

        throughput, peak_mem = _benchmark_megatron(
            model, local_rank, batch_size, seq_len, num_microbatches
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()
        mpu.destroy_model_parallel()

        return {
            "strategy":                   strategy,
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "throughput_samples_per_sec": throughput,
            "peak_gpu_mem_gb":            peak_mem,
            "status":                     "ok",
            "error":                      None,
        }

    except torch.cuda.OutOfMemoryError as e:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()
        try:
            mpu.destroy_model_parallel()
        except Exception:
            pass
        dist.barrier()
        return {
            "strategy":                   strategy,
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "status":                     "OOM",
            "error":                      str(e),
        }

    except Exception as e:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()
        try:
            mpu.destroy_model_parallel()
        except Exception:
            pass
        return {
            "strategy":                   strategy,
            "tp_size":                    tp_size,
            "pp_size":                    pp_size,
            "num_microbatches":           num_microbatches,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":            None,
            "status":                     "error",
            "error":                      str(e),
        }
