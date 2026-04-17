"""
train_gpt_v7_1.py

Diffusion-preserving v7 follow-up with lighter corruption defaults:
- earlier timestep mix
- lower shell and anchor masking ceilings
- clean-sequence passthrough
- gradient clipping on by default
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 4096))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    # Structure diffusion hyperparameters.
    struct_total_steps = int(os.environ.get("STRUCT_TOTAL_STEPS", 16))
    struct_schedule_steps = [int(x) for x in os.environ.get("STRUCT_SCHEDULE_STEPS", "4,5,6,7,8,9").split(",")]
    struct_schedule_weights = [float(x) for x in os.environ.get("STRUCT_SCHEDULE_WEIGHTS", "0.10,0.18,0.24,0.24,0.16,0.08").split(",")]
    struct_shell_floor = float(os.environ.get("STRUCT_SHELL_FLOOR", 0.02))
    struct_shell_ceil = float(os.environ.get("STRUCT_SHELL_CEIL", 0.45))
    struct_anchor_floor = float(os.environ.get("STRUCT_ANCHOR_FLOOR", 0.05))
    struct_anchor_ceil = float(os.environ.get("STRUCT_ANCHOR_CEIL", 0.60))
    struct_span_rate = float(os.environ.get("STRUCT_SPAN_RATE", 0.10))
    struct_mask_keep_prob = float(os.environ.get("STRUCT_MASK_KEEP_PROB", 0.10))
    struct_clean_prob = float(os.environ.get("STRUCT_CLEAN_PROB", 0.25))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("â–"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size}, got {file.stat().st_size}")
    tokens = np.fromfile(file, dtype="<u2", offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Token count mismatch for {file}: expected {num_tokens}, got {tokens.size}")
    return torch.tensor(tokens.astype(np.int32), dtype=torch.int32)


STRUCT_METHOD_NAME = "v7"
ANCHOR_TOKEN_PATTERNS = (
    re.compile(r"^(?:[A-Z][a-z]+){2,}$"),
    re.compile(r"^[A-Z]{2,}[a-z0-9]+"),
    re.compile(r"^[A-Za-z]+_[A-Za-z0-9_]+$"),
    re.compile(r"^[A-Za-z]+\.[A-Za-z0-9.]+$"),
)

def classify_piece_as_anchor(piece: str, sp: spm.SentencePieceProcessor, token_id: int) -> bool:
    if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
        return False
    if sp.is_byte(token_id):
        return False
    token = piece[1:] if piece.startswith("\u2581") else piece
    if not token or token.isspace():
        return False
    if token[:1].isupper() and any(ch.islower() for ch in token[1:]):
        return True
    if any(pattern.match(token) for pattern in ANCHOR_TOKEN_PATTERNS):
        return True
    if token.startswith(("http", "www.", "@", "#")):
        return True
    if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
        return True
    if any(ch in token for ch in "-_/.") and len(token) >= 3:
        return True
    return False

def token_anchor_multiplier(piece: str) -> float:
    token = piece[1:] if piece.startswith("\u2581") else piece
    if len(token) >= 10:
        return 1.35
    if any(ch in token for ch in "-_/."):
        return 1.25
    return 1.0

def build_structure_training_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, int]:
    table_size = max(int(sp.vocab_size()), vocab_size)
    is_anchor_np = np.zeros((table_size,), dtype=np.bool_)
    anchor_mult_np = np.ones((table_size,), dtype=np.float32)
    for token_id in range(int(sp.vocab_size())):
        piece = sp.id_to_piece(token_id)
        is_anchor_np[token_id] = classify_piece_as_anchor(piece, sp, token_id)
        anchor_mult_np[token_id] = token_anchor_multiplier(piece)
    mask_id = int(sp.unk_id()) if hasattr(sp, "unk_id") and int(sp.unk_id()) >= 0 else 0
    return (
        torch.tensor(is_anchor_np, dtype=torch.bool, device=device),
        torch.tensor(anchor_mult_np, dtype=torch.float32, device=device),
        mask_id,
    )

def _vectorized_cosine_mask_fraction(t_index: Tensor, total_steps: int, floor: float, ceil: float) -> Tensor:
    if total_steps <= 1:
        return torch.full_like(t_index, fill_value=ceil, dtype=torch.float32)
    u = t_index.to(dtype=torch.float32) / float(total_steps - 1)
    shaped = 0.5 - 0.5 * torch.cos(math.pi * u)
    return (floor + (ceil - floor) * shaped).clamp_(0.0, 1.0)

def _vectorized_midpoint_focus_multiplier(t_index: Tensor, total_steps: int, strength: float = 0.1) -> Tensor:
    if total_steps <= 1:
        return torch.ones_like(t_index, dtype=torch.float32)
    u = t_index.to(dtype=torch.float32) / float(total_steps - 1)
    focus = (1.0 - torch.abs(2.0 * u - 1.0)).clamp_min_(0.0)
    return (1.0 - strength * focus).clamp_min_(0.84)

def apply_structure_diffusion_corruption(
    x: Tensor,
    is_anchor_lut: Tensor,
    anchor_mult_lut: Tensor,
    mask_token_id: int,
    args: Hyperparameters,
) -> tuple[Tensor, Tensor, float, float]:
    batch_size = x.size(0)
    device = x.device
    weights = torch.tensor(args.struct_schedule_weights, dtype=torch.float32, device=device)
    step_values = torch.tensor(args.struct_schedule_steps, dtype=torch.long, device=device)
    sampled_idx = torch.multinomial(weights, batch_size, replacement=True)
    t_index = step_values[sampled_idx]
    clean_seq = torch.rand(batch_size, device=device) < args.struct_clean_prob
    if batch_size > 1 and 0.0 < args.struct_clean_prob < 1.0:
        if not bool(clean_seq.any().item()):
            clean_seq[torch.randint(0, batch_size, (), device=device)] = True
        elif bool(clean_seq.all().item()):
            clean_seq[torch.randint(0, batch_size, (), device=device)] = False
    t_index = torch.where(clean_seq, torch.zeros_like(t_index), t_index)

    shell_fraction_seq = _vectorized_cosine_mask_fraction(
        t_index, args.struct_total_steps, args.struct_shell_floor, args.struct_shell_ceil
    )
    anchor_fraction_seq = _vectorized_cosine_mask_fraction(
        t_index, args.struct_total_steps, args.struct_anchor_floor, args.struct_anchor_ceil
    )
    anchor_fraction_seq = anchor_fraction_seq * _vectorized_midpoint_focus_multiplier(
        t_index, args.struct_total_steps
    )

    x_out = x.clone()
    token_anchor = is_anchor_lut[x]
    shell_probs = shell_fraction_seq[:, None].expand_as(x_out).to(dtype=torch.float32)
    anchor_probs = anchor_fraction_seq[:, None].expand_as(x_out).to(dtype=torch.float32)
    probs = torch.where(token_anchor, anchor_probs, shell_probs)
    probs = torch.where(token_anchor, probs * anchor_mult_lut[x], probs)
    probs = probs.clamp_(0.0, 0.98)

    mask = torch.rand_like(probs) < probs
    bridge_seed = (~token_anchor) & (torch.rand_like(probs) < (args.struct_span_rate * shell_probs))
    bridge_spans = bridge_seed.clone()
    bridge_spans[:, 1:] |= bridge_seed[:, :-1]
    bridge_spans[:, 2:] |= bridge_seed[:, :-2]
    mask |= bridge_spans

    keep = torch.rand_like(probs) < args.struct_mask_keep_prob
    mask &= ~keep
    mask[clean_seq] = False
    x_out[mask] = mask_token_id

    return x_out, t_index, float(shell_fraction_seq.mean().item()), float(anchor_fraction_seq.mean().item())


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No training files found for pattern: {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(1234 + rank)
        self._reshuffle()

    def _reshuffle(self) -> None:
        perm = torch.randperm(len(self.files), generator=self.generator).tolist()
        self.order = [self.files[i] for i in perm]
        self.file_idx = 0
        self.tokens = torch.empty(0, dtype=torch.int32)
        self.position = 0

    def _load_next(self) -> None:
        if self.file_idx >= len(self.order):
            self._reshuffle()
        self.tokens = load_data_shard(self.order[self.file_idx])
        self.file_idx += 1
        self.position = 0

    def next_batch(self, train_batch_tokens: int, train_seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_batch_tokens = train_batch_tokens // (self.world_size * grad_accum_steps)
        local_batch_seqs = local_batch_tokens // train_seq_len
        needed = local_batch_seqs * train_seq_len + 1
        if local_batch_seqs <= 0:
            raise ValueError("TRAIN_BATCH_TOKENS too small for the current WORLD_SIZE/GRAD_ACCUM_STEPS")

        chunks = []
        while sum(chunk.numel() for chunk in chunks) < needed:
            if self.tokens.numel() == 0 or self.position >= self.tokens.numel() - 1:
                self._load_next()
            remaining = self.tokens.numel() - self.position
            take = min(needed - sum(chunk.numel() for chunk in chunks), remaining)
            if take <= 0:
                break
            chunks.append(self.tokens[self.position : self.position + take])
            self.position += take - 1
        joined = torch.cat(chunks, dim=0)[:needed]
        x = joined[:-1].reshape(local_batch_seqs, train_seq_len).to(device=self.device, dtype=torch.int64, non_blocking=True)
        y = joined[1:].reshape(local_batch_seqs, train_seq_len).to(device=self.device, dtype=torch.int64, non_blocking=True)
        return x, y


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)

class RMSNorm(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(self.dim),))

class Rotary(nn.Module):
    def __init__(self, head_dim: int, rope_base: float):
        super().__init__()
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must divide num_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.kv_repeat = num_heads // num_kv_heads
        self.c_q = CastedLinear(model_dim, model_dim, bias=False)
        self.c_k = CastedLinear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = CastedLinear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.proj = CastedLinear(model_dim, model_dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, rope_base)
        self.attn_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos[None, None, :, :], sin[None, None, :, :])
        k = apply_rotary_emb(k, cos[None, None, :, :], sin[None, None, :, :])
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)
        scale = (self.head_dim ** -0.5) * self.attn_scale.to(dtype=q.dtype)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, model_dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * model_dim
        self.fc = CastedLinear(model_dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, model_dim, bias=False)
        self.mlp_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.fc(x), approximate="tanh")
        return self.proj(h) * self.mlp_scale.to(dtype=x.dtype)

class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(model_dim, mlp_mult)
        self.resid_mix = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        mix = self.resid_mix.to(dtype=x.dtype)
        x = x + self.mlp(F.rms_norm(x + mix * x0, (x.size(-1),)))
        return x


def restore_low_dim_params_to_fp32(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            param.data = param.data.float()

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.skip_weights = nn.Parameter(torch.zeros(self.num_decoder_layers, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def main() -> None:
    print("Helper dependency copy for train_gpt_v9_6_proxy_smoke_record.py")


if __name__ == "__main__":
    main()
