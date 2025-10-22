#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def fail(reason: str) -> None:
    print(f"❌ FAIL ({reason})")
    sys.exit(1)


def main() -> None:
    # In AgentBeats style, green/ and white/ are siblings
    white_dir = (Path(__file__).parent / "../white").resolve()
    if not white_dir.exists():
        fail(f"white directory not found: {white_dir}")

    # 1) Build the white submission
    try:
        build = subprocess.run(
            ["make"], cwd=str(white_dir), capture_output=True, text=True, check=False
        )
    except Exception as e:
        fail(f"make invocation failed: {e}")

    if build.returncode != 0:
        err = (build.stderr or build.stdout or "make failed").strip()
        fail(f"make failed: {err.splitlines()[-1] if err else 'nonzero exit'}")

    # Dataset root (env override supported)
    dataset_root = os.environ.get("DATA_ROOT", "/data")
    # Fallback to repo-local ./data if /data doesn't exist
    if not Path(dataset_root).exists():
        local_data = (Path(__file__).parent / "../data").resolve()
        if local_data.exists():
            dataset_root = str(local_data)
    if not Path(dataset_root).exists():
        fail(f"dataset not found: {dataset_root}")

    binary = white_dir / "dirhash_fast"
    if not binary.exists():
        fail(f"compiled binary not found: {binary}")

    cmd = [
        str(binary),
        dataset_root,
        "--min-bytes",
        "1024",
        "--top",
        "10",
        "--threads",
        "8",
    ]

    # 2) Load config and run with timing
    config_path = Path(__file__).parent / "config.json"
    max_seconds = 6.0
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict) and "max_seconds" in cfg:
                max_seconds = float(cfg["max_seconds"])
        except Exception:
            pass

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(white_dir),
            capture_output=True,
            text=True,
            timeout=max(12.0, max_seconds + 4.0),
        )
    except subprocess.TimeoutExpired:
        fail(f"timeout > {max(12.0, max_seconds + 4.0):.0f}s")
    except FileNotFoundError as e:
        fail(f"run failed: {e}")
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "nonzero exit").strip()
        fail(f"nonzero exit: {err.splitlines()[-1] if err else proc.returncode}")

    # 3) Parse stdout JSON
    raw = (proc.stdout or "").strip()
    if not raw:
        fail("empty stdout")

    # The program is expected to emit one JSON line; be tolerant and
    # take the last non-empty line if multiple lines are present.
    last_line = [ln for ln in raw.splitlines() if ln.strip()]
    last_line = last_line[-1] if last_line else raw

    try:
        data = json.loads(last_line)
    except json.JSONDecodeError as e:
        fail(f"invalid JSON: {e.msg}")

    # 4) Validate schema and constraints
    required_keys = {
        "ok",
        "files_scanned",
        "bytes_scanned",
        "groups",
        "top",
        "threads",
        "ms",
    }
    missing = [k for k in required_keys if k not in data]
    if missing:
        fail(f"missing keys: {', '.join(sorted(missing))}")

    if data.get("ok") is not True:
        fail("ok != true")

    if elapsed > max_seconds:
        fail(f"slow: {elapsed:.2f}s > {max_seconds:.2f}s")

    print(f"✅ PASS (time={elapsed:.2f}s)")
    sys.exit(0)


if __name__ == "__main__":
    main()


