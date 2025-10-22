# Terminal-Bench Evaluator (GPU Mini)

Evaluates terminal agents on a simple CUDA softmax task. The harness runs the agent container with GPU, sets `TB_SEED`, expects the agent to `make && ./softmax_bench 2048 1024 && ./naive_softmax_bench 2048 1024`, parses the agent's JSON, verifies correctness and checksums, computes speed points vs baseline, and writes `/output/results.json`.

## Install

```bash
python3.10 -m pip install -r requirements.txt
```

Ensure Docker (with NVIDIA runtime) is running on a GPU host.

## Run the evaluator server

```bash
python RunEval.py \
  --harness-image terminalbench/harness:gpu-mini \
  --task-id tb.softmax_gpu_n2048_d1024 \
  --port 9999 \
  --public-url http://localhost:9999
```

Submission payload (A2A):
```json
{ "agent_image": "my-softmax-agent:latest", "integration_mode": "mcp" }
```

Artifacts: `evaluation_summary` (JSON) and `harness_logs` (text).

## Harness image

- Files: `harness/Dockerfile`, `harness/tb_harness.py`
- Build:
```bash
cd harness
docker build -t terminalbench/harness:gpu-mini .
```

## Starter (for contestants)
- See `starter/` for `Makefile`, `main.cu`, `naive_softmax.cu`, `kernel.cu` (TODO), and `cpu_ref.hpp`.
- Agent image must contain this project and support:
  - `make && ./softmax_bench 2048 1024 && ./naive_softmax_bench 2048 1024`
  - Prints a single JSON line from each binary: `{ "ok": true, "ms": 3.42, "checksum": "..." }`
- The harness sets `TB_SEED` and verifies checksum and accuracy.

## Results schema

`/output/results.json` example:
```json
{
  "task_id": "tb.softmax_gpu_n2048_d1024",
  "success": true,
  "metrics": {
    "ok": true,
    "checksum_match": true,
    "row_sum_max_abs_dev": 7.1e-8,
    "ms_team": 3.42,
    "ms_baseline": 10.51,
    "speed_points": 34.0,
    "correctness_points": 50.0,
    "total_points": 84.0
  }
}
```
