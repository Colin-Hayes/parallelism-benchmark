"""
MP.py
-----
Pipeline Parallelism using fairscale.nn.Pipe.

Each GPU owns a slice of the transformer layers. The Pipe class streams
micro-batches through the stages so multiple GPUs overlap:

    GPU 0  →  embedding + first N/4 transformer blocks
    GPU 1  →  next N/4 blocks
    GPU 2  →  next N/4 blocks
    GPU 3  →  final N/4 blocks + LayerNorm + LM head

fairscale.nn.Pipe differences from the old torch.distributed.pipeline.sync.Pipe:
  - Requires a `balance` list: how many nn.Sequential children go on each device
  - Uses Gloo backend (not NCCL) for its internal RPC communication
  - The model must be a flat nn.Sequential — no nested modules as stages

Called by pp_bench.py — do not run directly.
"""

import gc
import time

import torch
import torch.nn as nn
from fairscale.nn import Pipe
from transformers import GPT2Config, GPT2LMHeadModel

WARMUP_STEPS = 5
BENCH_STEPS  = 20


# ── Flatten model into a sequential list of modules ───────────────────────────
#
# fairscale Pipe requires a flat nn.Sequential where each child is one
# "layer" for balancing purposes. We expose:
#
#   [0]        EmbeddingLayer        → GPU 0
#   [1..N]     one GPT2Block each    → distributed across GPUs by balance
#   [N+1]      FinalNormAndHead      → last GPU
#
# balance = [k0, k1, k2, k3] where ki = number of Sequential children on GPU i
# and sum(balance) = total number of children in the Sequential.

class _EmbeddingLayer(nn.Module):
    """Token + positional embedding → hidden states."""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.wte  = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe  = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        pos  = torch.arange(t, device=input_ids.device).unsqueeze(0)
        return self.drop(self.wte(input_ids) + self.wpe(pos))


class _BlockLayer(nn.Module):
    """Single GPT2Block wrapper — returns only hidden states (drops extras)."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.block(hidden)[0]   # GPT2Block returns (hidden, present, ...)


class _FinalNormAndHead(nn.Module):
    """LayerNorm + LM head → logits."""
    def __init__(self, ln_f: nn.Module, lm_head: nn.Linear):
        super().__init__()
        self.ln_f    = ln_f
        self.lm_head = lm_head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_f(hidden))


def _build_sequential(model_cfg: dict):
    """
    Decompose a GPT-2 model into a flat nn.Sequential of individual layers.

    Returns
    -------
    seq      : nn.Sequential  — flat list of layers for Pipe
    n_layers : int            — number of transformer blocks (for balance calc)
    """
    config = GPT2Config(vocab_size=50257, **model_cfg)
    base   = GPT2LMHeadModel(config)

    layers = [_EmbeddingLayer(config)]
    for block in base.transformer.h:
        layers.append(_BlockLayer(block))
    layers.append(_FinalNormAndHead(base.transformer.ln_f, base.lm_head))

    del base
    return nn.Sequential(*layers), config.n_layer


def _compute_balance(num_layers: int, num_stages: int) -> list:
    """
    Compute how many Sequential children go on each GPU.

    Total children = 1 (embedding) + num_layers (blocks) + 1 (head)
    We put the embedding on GPU 0 and the head on the last GPU, then
    distribute the transformer blocks as evenly as possible.

    Example: 12 blocks, 4 GPUs
      total = 14 children
      base  = 14 // 4 = 3, remainder = 2
      balance = [4, 4, 3, 3]  (sums to 14)
    """
    total     = num_layers + 2   # +1 embedding, +1 head
    base      = total // num_stages
    remainder = total % num_stages
    balance   = []
    for i in range(num_stages):
        balance.append(base + (1 if i < remainder else 0))
    return balance


# ── Core benchmark loop ───────────────────────────────────────────────────────

def _benchmark_pipe(
    pipe_model,
    local_rank:  int,
    batch_size:  int,
    seq_len:     int,
    num_stages:  int,
) -> tuple:
    """
    5 warmup steps then 20 timed steps.
    Input goes in on cuda:0, logits come out on the last GPU.
    Loss is computed on the last GPU.
    """
    last_device = str(pipe_model.devices[-1])
    opt = torch.optim.AdamW(pipe_model.parameters(), lr=1e-4)

    def _make_input():
        return torch.randint(0, 50257, (batch_size, seq_len), device="cuda:0")

    def _make_labels():
        return torch.randint(0, 50257, (batch_size, seq_len), device=last_device)

    def _step():
        logits = pipe_model(_make_input())
        loss   = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            _make_labels().view(-1),
        )
        loss.backward()
        opt.step()
        opt.zero_grad()

    for _ in range(WARMUP_STEPS):
        _step()

    torch.cuda.synchronize(local_rank)
    for i in range(num_stages):
        torch.cuda.reset_peak_memory_stats(i)

    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        _step()
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - t0

    throughput  = round((BENCH_STEPS * batch_size) / elapsed, 2)
    peaks = []
    for i in range(num_stages):
        peaks.append(torch.cuda.max_memory_allocated(i) / 1e9)
    
    peak_mem_gb = round(max(peaks), 3)
    
    return throughput, peak_mem_gb


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline_mp(
    model_cfg:  dict,
    batch_size: int,
    seq_len:    int,
    local_rank: int,
    num_stages: int = 4,
    num_chunks: int = 4,
) -> dict:
    """
    Build and benchmark a pipeline-parallel GPT-2 model using fairscale.nn.Pipe.

    Parameters
    ----------
    model_cfg   : GPT2Config kwargs
    batch_size  : total batch size split into num_chunks micro-batches
    seq_len     : token sequence length
    local_rank  : cuda device index (0 for single-process pipeline)
    num_stages  : number of pipeline stages = number of GPUs
    num_chunks  : micro-batches per step (more = smaller bubble, more memory)
    """
    strategy = "pipeline_mp"

    try:
        seq_model, num_layers = _build_sequential(model_cfg)
        balance = _compute_balance(num_layers, num_stages)

        print(f"  Pipeline balance: {balance} across {num_stages} GPUs")

        # fairscale Pipe API:
        #   - model      : flat nn.Sequential (on CPU)
        #   - balance    : list of ints, one per GPU
        #   - devices    : which cuda devices to use
        #   - chunks     : number of micro-batches
        #   - checkpoint : "except_last" saves memory by recomputing activations
        pipe_model = Pipe(
            module=seq_model,
            balance=balance,
            devices=list(range(num_stages)),
            chunks=num_chunks,
            checkpoint="except_last",
        )

        throughput, peak_mem = _benchmark_pipe(
            pipe_model, local_rank, batch_size, seq_len, num_stages
        )

        del pipe_model, seq_model
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "strategy":                   strategy,
            "num_stages":                 num_stages,
            "num_chunks":                 num_chunks,
            "throughput_samples_per_sec": throughput,
            "peak_gpu_mem_gb":      peak_mem,
            "status":                     "ok",
            "error":                      None,
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "strategy":                   strategy,
            "num_stages":                 num_stages,
            "num_chunks":                 num_chunks,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":      None,
            "status":                     "OOM",
            "error":                      str(e),
        }

    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "strategy":                   strategy,
            "num_stages":                 num_stages,
            "num_chunks":                 num_chunks,
            "throughput_samples_per_sec": None,
            "peak_gpu_mem_gb":      None,
            "status":                     "error",
            "error":                      str(e),
        }