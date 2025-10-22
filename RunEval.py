import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

if sys.version_info < (3, 10):
    python310 = shutil.which("python3.10")
    if python310:
        os.execv(python310, [python310, *sys.argv])
    raise SystemExit(
        "Terminal-Bench server requires Python 3.10 or newer. Install dependencies with "
        "python3.10 -m pip install -r requirements.txt and re-run the command."
    )

import uvicorn

from TerminalBenchServer import create_terminalbench_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TerminalBench evaluation service (A2A server)")
    parser.add_argument("--harness-image", required=True, help="Docker image for the TerminalBench harness")
    parser.add_argument("--task-id", required=True, help="TerminalBench task identifier")
    parser.add_argument("--max-time-seconds", type=int, default=1800, help="Max allowed runtime for evaluation.")
    parser.add_argument("--max-memory-mb", type=int, default=6144, help="Max memory in MB for docker container.")
    parser.add_argument("--max-cpus", type=float, default=2.0, help="CPU limit exposed to docker.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the A2A HTTP server.")
    parser.add_argument("--port", type=int, default=9999, help="Port for the A2A HTTP server.")
    parser.add_argument("--public-url", help="Public URL advertised in the agent card.")
    parser.add_argument("--agent-name", default="TerminalBench Evaluator", help="Display name for the agent card.")
    parser.add_argument("--agent-description", help="Optional override for the agent card description.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the server process.",
    )
    return parser.parse_args()


def build_task_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "benchmark": "terminalbench",
        "harness_image": args.harness_image,
        "task_id": args.task_id,
        "constraints": {
            "max_time_seconds": args.max_time_seconds,
            "max_memory_mb": args.max_memory_mb,
            "max_cpus": args.max_cpus,
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    task_config = build_task_config(args)

    app = create_terminalbench_app(
        task_config,
        public_url=args.public_url,
        agent_name=args.agent_name,
        agent_description=args.agent_description,
    )

    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
