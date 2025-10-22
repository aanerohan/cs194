import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import chromadb
import docker
from docker.errors import APIError, ContainerError, NotFound
from docker.types import DeviceRequest

from Memory import collection

logger = logging.getLogger(__name__)


class TerminalBenchAgent:
    def __init__(self, task_config: dict):
        self.harness_image = task_config["harness_image"]
        self.task_id = task_config["task_id"]

        raw = task_config.get("constraints") or {}
        self.constraints = {
            "max_time_seconds": raw.get("max_time_seconds", 1800),
            "max_memory_mb": raw.get("max_memory_mb", 6144),
            "max_cpus": float(raw.get("max_cpus", 2.0)),
        }

        # Optional shape parameters for this task (defaults per spec)
        self.N = int(task_config.get("N", 2048))
        self.D = int(task_config.get("D", 1024))

        self.db_client = chromadb.PersistentClient(path="./agent_memory_db")
        self.collection_name = "terminalbench_results"
        self.eval_collection = self.db_client.get_or_create_collection(name=self.collection_name)

    def evaluate(self, submission: dict) -> dict:
        execution = self.run_terminalbench(
            harness_image=self.harness_image,
            agent_image=submission["agent_image"],
            task_id=self.task_id,
            integration_mode=submission.get("integration_mode", "mcp"),
            extra_args=submission.get("harness_args", []),
        )

        constraints = self.check_constraints(execution, self.constraints)
        performance = self.evaluate_performance(execution)

        result = {
            "execution": execution,
            "constraints": constraints,
            "performance": performance,
        }
        self._record_run(result)
        return result

    def run_terminalbench(
        self,
        harness_image: str,
        agent_image: str,
        task_id: str,
        integration_mode: str = "mcp",
        extra_args: Optional[Sequence[str]] = None,
    ) -> dict:
        client = docker.from_env()
        output_dir = Path(f"/tmp/tb_outputs_{uuid.uuid4().hex}")
        output_dir.mkdir(parents=True, exist_ok=True)
        volumes = {
            str(output_dir): {"bind": "/output", "mode": "rw"},
            "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
        }

        # GPU access for harness to run agent with --gpus all
        device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

        # Invoke harness CLI (expects tb_harness.py inside harness image)
        cmd = [
            "python3", "/app/tb_harness.py",
            "--agent-image", agent_image,
            "--output", "/output/results.json",
            "--N", str(self.N),
            "--D", str(self.D),
        ]
        if extra_args:
            cmd.extend(list(extra_args))

        container = None
        try:
            container = client.containers.run(
                harness_image,
                command=cmd,
                tty=True,
                stdin_open=False,
                volumes=volumes,
                mem_limit=f"{int(self.constraints['max_memory_mb'])}m",
                cpus=float(self.constraints["max_cpus"]),
                detach=True,
                remove=False,
                environment={
                    "TB_TASK_ID": task_id,
                    "TB_OUTPUT_DIR": "/output",
                },
                device_requests=device_requests,
            )
            start = time.time()
            result = container.wait(timeout=int(self.constraints["max_time_seconds"]))
            elapsed = time.time() - start

            stats = container.stats(stream=False)
            logs = container.logs().decode("utf-8", errors="replace")
            results_path = output_dir / "results.json"

            exec_ok = result.get("StatusCode", 1) == 0
            exec_meta: Dict[str, Union[bool, str, float, int, dict]] = {
                "success": exec_ok,
                "output_dir": str(output_dir),
                "results_path": str(results_path) if results_path.exists() else None,
                "time_seconds": elapsed,
                "memory_used_mb": self._extract_memory_usage(stats),
                "logs": logs,
            }
            if results_path.exists():
                try:
                    exec_meta["results"] = json.loads(results_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    exec_meta["results_error"] = str(exc)
            return exec_meta

        except (ContainerError, APIError, NotFound) as exc:
            err_logs = container.logs().decode("utf-8", errors="replace") if container else ""
            return {
                "success": False,
                "error": str(exc),
                "output_dir": str(output_dir),
                "results_path": None,
                "time_seconds": 0.0,
                "memory_used_mb": 0.0,
                "logs": err_logs,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "output_dir": str(output_dir),
                "results_path": None,
                "time_seconds": 0.0,
                "memory_used_mb": 0.0,
                "logs": "",
            }
        finally:
            if container:
                try:
                    container.remove(force=True)
                except NotFound:
                    pass

    def check_constraints(self, execution: dict, constraints: dict) -> dict:
        if not execution.get("success"):
            return {"passed": False, "violations": ["Execution failed"]}

        violations = []
        if constraints.get("max_time_seconds") is not None and execution.get("time_seconds", 0) > constraints["max_time_seconds"]:
            violations.append(f"Time limit exceeded: {execution['time_seconds']}s > {constraints['max_time_seconds']}s")
        if constraints.get("max_memory_mb") is not None and execution.get("memory_used_mb", 0) > constraints["max_memory_mb"]:
            violations.append(f"Memory limit exceeded: {execution['memory_used_mb']}MB > {constraints['max_memory_mb']}MB")
        if not execution.get("results_path") or "results" not in execution:
            violations.append("Results JSON not found or unreadable")
        return {"passed": len(violations) == 0, "violations": violations}

    def evaluate_performance(self, execution: dict) -> Dict[str, Union[float, int, bool]]:
        results = execution.get("results") or {}
        success = bool(results.get("success") or results.get("passed") or False)
        # Prefer metrics.ms_* if present; keep compatibility with earlier keys
        metrics = results.get("metrics") or {}
        steps = int(results.get("steps") or results.get("num_steps") or 0)
        wall_time = float(
            results.get("wall_time")
            or results.get("duration_seconds")
            or metrics.get("ms_team", 0.0) / 1000.0
            or execution.get("time_seconds")
            or 0.0
        )
        # Composite: success primary; faster gets higher score if metrics available
        speed_points = float(metrics.get("speed_points", 0.0))
        correctness_points = float(metrics.get("correctness_points", 50.0 if success else 0.0))
        total_points = float(metrics.get("total_points", correctness_points + speed_points))
        return {
            "success": success,
            "steps": steps,
            "wall_time": wall_time,
            "total_points": round(total_points, 4),
        }

    @staticmethod
    def _extract_memory_usage(stats: dict) -> float:
        try:
            memory_bytes = stats["memory_stats"]["max_usage"]
            return memory_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    def _record_run(self, results: dict) -> None:
        doc_id = f"tb_run_{uuid.uuid4().hex}"
        payload = json.dumps(
            {
                "performance": results.get("performance", {}),
                "constraints": results.get("constraints", {}),
                "execution": results.get("execution"),
            },
            default=str,
        )
        self.eval_collection.upsert(documents=[payload], metadatas=[{"collection": self.collection_name}], ids=[doc_id])
        try:
            collection.upsert(documents=[payload], metadatas=[{"collection": self.collection_name}], ids=[doc_id])
        except Exception as exc:
            logger.warning("Unable to upsert into shared memory collection: %s", exc)
