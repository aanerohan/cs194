# TerminalBench-GPU (Mini) Starter

Implement `kernel.cu` to compute row-wise, numerically stable softmax in float32.

- Fixed shape: `N=2048`, `D=1024` (but code should accept args)
- Build:
  - `make` -> builds `softmax_bench` and `naive_softmax_bench`
- Run:
  - `./softmax_bench 2048 1024` -> prints JSON: `{ "ok": true, "ms": 3.42, "checksum": "..." }`
  - `./naive_softmax_bench 2048 1024` -> baseline timing JSON (same schema)
- The harness sets `TB_SEED` env var. Your program must use it to generate inputs.
- Correctness: per-row sums ~ 1.0, abs/rel error ≤ 1e-6 vs CPU ref.
- Timing: use CUDA events; report average over 10 runs in `ms`.
