"""
Sanitized reference copy for the RX 9070 Ultra Proxy v9.6 smoke run.

This bundle is included for a compute-grant draft PR together with the exact
run parameters and training log. The run used a larger local script that built
on a selective-denoising / recurrence stack and imported helper functionality
from train_gpt_v7_1.py.

Key ideas exercised in the run:
- progressive two-phase layer looping
- Kronecker loop / override MLP paths
- shared-pass signaling
- selective structural denoising introduced later in training
- mixed GPTQ export with sliding-window evaluation

For this draft PR, the attached log and run_command_sanitized.txt are the source
of truth for the exact smoke-run configuration and outcome.
"""

from __future__ import annotations

import os
import train_gpt_v7_1 as struct_base


class Hyperparameters:
    run_id = os.environ.get("RUN_ID", "rx9070_ULTRA_PROXY_v9_6_smoke")
    seed = int(os.environ.get("SEED", 42))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    iterations = int(os.environ.get("ITERATIONS", 8000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.40))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 150))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0))

    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 65536))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 65536))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 8))

    num_layers = int(os.environ.get("NUM_LAYERS", 14))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", 512))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))

    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 4))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.20))
    loop_phase2_at = float(os.environ.get("LOOP_PHASE2_AT", 0.35))
    untie_loop_mlps = bool(int(os.environ.get("UNTIE_LOOP_MLPS", 1)))
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))

    use_kron_loop_mlps = bool(int(os.environ.get("USE_KRON_LOOP_MLPS", 1)))
    use_kron_override_mlps = bool(int(os.environ.get("USE_KRON_OVERRIDE_MLPS", 1)))
    kron_min_features = int(os.environ.get("KRON_MIN_FEATURES", 64))
    shared_pass_signal = bool(int(os.environ.get("SHARED_PASS_SIGNAL", 1)))
    shared_pass_max = int(os.environ.get("SHARED_PASS_MAX", 8))
    shared_pass_scale = float(os.environ.get("SHARED_PASS_SCALE", 1.0))

    struct_total_steps = int(os.environ.get("STRUCT_TOTAL_STEPS", 16))
    struct_start_frac = float(os.environ.get("STRUCT_START_FRAC", 0.42))
    struct_aux_prob = float(os.environ.get("STRUCT_AUX_PROB", 0.20))
    aux_denoise_weight = float(os.environ.get("AUX_DENOISE_WEIGHT", 0.22))


def main() -> None:
    print("Reference script copy for the RX 9070 Ultra Proxy v9.6 smoke run.")
    print("See run_command_sanitized.txt and rx9070_ULTRA_PROXY_v9_6_smoke.log.txt for the exact run setup and result.")
    print(f"struct helper module loaded: {struct_base.__name__}")


if __name__ == "__main__":
    main()
