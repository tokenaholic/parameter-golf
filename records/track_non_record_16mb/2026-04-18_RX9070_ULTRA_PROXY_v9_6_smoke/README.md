# RX 9070 Ultra Proxy v9.6 Smoke

Draft non-record submission prepared to support a compute grant request.

## Summary
This run tests a compact SP8192 model with:
- progressive two-phase layer looping
- Kronecker loop / override MLP paths
- shared-pass signaling
- selective structural denoising introduced later in training
- mixed GPTQ export with sliding-window evaluation

## Purpose
This is not presented as a final tuned record submission. It is a smoke/proxy run on a single AMD RX 9070 used to verify that the intended schedule changes actually trigger:
- phase1 loop activation
- phase2 loop activation
- structural auxiliary objective activation

## Result
- Final val_bpb: `1.2178`
- Post-EMA val_bpb: `1.21514573`
- Quantized sliding-window val_bpb: `1.20669600`

## Notes
- This run was performed as a local proxy experiment, not on the official 8xH100 track.
- The files here are sanitized copies prepared for a draft PR / compute grant reference.
- The local smoke-run script used a structural helper module with an older filename; in this bundle that dependency is renamed to `struct_helper_reference.py` so it does not imply the run used an older training configuration.
- The exact run setup is captured in `run_command_sanitized.txt` and `rx9070_ULTRA_PROXY_v9_6_smoke.log.txt`.
