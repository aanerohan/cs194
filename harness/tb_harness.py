import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, timeout=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def docker_run_agent(agent_image: str, env: dict, dataset_host_path: str) -> tuple[int, str, str]:
    env_flags = []
    for k, v in env.items():
        env_flags.extend(["-e", f"{k}={v}"])
    cmd = [
        "docker", "run", "--rm",
        *env_flags,
        "-v", f"{dataset_host_path}:/data:ro",
        agent_image,
        "bash", "-lc", "make -C /white || make && /white/dirhash_fast /data --min-bytes 1024 --top 10 --threads 8",
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


def validate_dirhash_json(d: dict) -> tuple[bool, str]:
    required = {"ok", "files_scanned", "bytes_scanned", "groups", "top", "threads", "ms"}
    missing = [k for k in required if k not in d]
    if missing:
        return False, f"missing keys: {', '.join(sorted(missing))}"
    if not bool(d.get("ok")):
        return False, "ok != true"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--max-seconds", type=float, default=6.0)
    args = ap.parse_args()

    rc, out, err = docker_run_agent(args.agent_image, {"DATA_ROOT": args.data_root}, args.data_root)
    combined = (out or "") + "\n" + (err or "")

    js = parse_json_lines(combined)
    result = js[-1] if js else {}
    valid, reason = validate_dirhash_json(result)
    wall_time = float(result.get("ms", 0.0)) / 1000.0
    success = bool(valid and (wall_time <= args.max_seconds))

    results = {
        "task_id": "tb.dirhash_fast",
        "success": bool(success),
        "metrics": {
            "ok": bool(result.get("ok", False)),
            "files_scanned": int(result.get("files_scanned", 0)),
            "bytes_scanned": int(result.get("bytes_scanned", 0)),
            "top": int(result.get("top", 0)),
            "threads": int(result.get("threads", 0)),
            "ms": float(result.get("ms", 0.0)),
            "wall_time": float(wall_time),
        },
        "logs_tail": combined[-8000:],
    }

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(json.dumps({"harness_ok": True, "wrote": str(outp)}))


if __name__ == "__main__":
    main()
