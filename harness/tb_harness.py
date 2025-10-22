import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def run(cmd, timeout=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def docker_run_agent(agent_image: str, env: dict) -> tuple[int, str, str]:
    env_flags = []
    for k, v in env.items():
        env_flags.extend(["-e", f"{k}={v}"])
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        *env_flags,
        agent_image,
        "bash", "-lc", "make && ./softmax_bench 2048 1024 && ./naive_softmax_bench 2048 1024",
    ]
    return run(cmd, timeout=1200)


def parse_json_lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                out.append(json.loads(s))
            except Exception:
                pass
    return out


def cpu_softmax_and_checksum(seed: int, N: int, D: int) -> tuple[str, float]:
    import numpy as np, hashlib
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D), dtype=np.float32)
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m, dtype=np.float32)
    y = e / e.sum(axis=1, keepdims=True)
    h = hashlib.sha256(y.tobytes()).hexdigest()
    row_dev = float(np.max(np.abs(y.sum(axis=1) - 1.0)))
    return h, row_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--D", type=int, default=1024)
    args = ap.parse_args()

    seed = random.randint(0, 2**31 - 1)
    rc, out, err = docker_run_agent(args.agent_image, {"TB_SEED": str(seed)})
    combined = (out or "") + "\n" + (err or "")

    js = parse_json_lines(combined)
    team = js[0] if js else {}
    base = js[-1] if js else {}

    ok_team = bool(team.get("ok"))
    team_ms = float(team.get("ms", 0.0))
    team_checksum = str(team.get("checksum", ""))

    ref_checksum, row_dev = cpu_softmax_and_checksum(seed, args.N, args.D)
    checksum_match = (team_checksum == ref_checksum)
    correctness_pass = ok_team and checksum_match and (row_dev <= 1e-6)

    base_ms = float(base.get("ms", 0.0)) if isinstance(base, dict) else 0.0
    speed_points = 0.0
    if team_ms > 0.0 and base_ms > 0.0:
        speedup = base_ms / team_ms
        speed_points = min(speedup, 3.0) / 3.0 * 50.0

    correctness_points = 50.0 if correctness_pass else 0.0
    total = correctness_points + speed_points

    results = {
        "task_id": f"tb.softmax_gpu_n{args.N}_d{args.D}",
        "success": bool(correctness_pass),
        "metrics": {
            "ok": bool(ok_team),
            "checksum_match": bool(checksum_match),
            "row_sum_max_abs_dev": float(row_dev),
            "ms_team": float(team_ms),
            "ms_baseline": float(base_ms),
            "speed_points": float(speed_points),
            "correctness_points": float(correctness_points),
            "total_points": float(total),
        },
        "logs_tail": combined[-8000:],
    }

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(json.dumps({"harness_ok": True, "wrote": str(outp)}))


if __name__ == "__main__":
    main()
