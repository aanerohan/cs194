#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from RunEval import main as run_eval_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentBeats-compatible green agent for dirhash-fast")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--public-url", default="http://localhost:9999")
    args = parser.parse_args()

    # Configure task for dirhash-fast using our existing A2A server scaffold
    os.execvp(
        "python3",
        [
            "python3",
            str(Path(__file__).parents[1] / "RunEval.py"),
            "--harness-image",
            "terminalbench/harness:cpu-mini",
            "--task-id",
            "tb.dirhash_fast",
            "--port",
            str(args.port),
            "--public-url",
            args.public_url,
            "--agent-name",
            "TerminalBench dirhash-fast Green",
        ],
    )


if __name__ == "__main__":
    main()


