import asyncio
import copy
import json
import logging
import os
from typing import Any, Dict, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import (
    new_agent_text_message,
    new_data_artifact,
    new_text_artifact,
)

from TerminalBenchAgent import TerminalBenchAgent

logger = logging.getLogger(__name__)


class TerminalBenchExecutor(AgentExecutor):
    MAX_LOG_CHARACTERS = 10_000

    def __init__(self, task_config: Dict[str, Any]):
        self._task_config = copy.deepcopy(task_config)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            submission = self._extract_submission(context)
        except ValueError as exc:
            await self._send_status(context, event_queue, TaskState.failed, str(exc), final=True)
            return

        await self._send_status(context, event_queue, TaskState.working, "Running TerminalBench harness with submitted agent.")

        agent = TerminalBenchAgent(copy.deepcopy(self._task_config))
        try:
            result = await asyncio.to_thread(agent.evaluate, submission)
        except Exception as exc:
            logger.exception("TerminalBench evaluation failed.")
            await self._send_status(context, event_queue, TaskState.failed, f"Evaluation error: {exc}", final=True)
            return

        await self._publish_results(context, event_queue, result)
        await self._send_status(context, event_queue, TaskState.completed, "Evaluation complete.", final=True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._send_status(context, event_queue, TaskState.rejected, "Cancellation is not supported for this agent.", final=True)

    def _extract_submission(self, context: RequestContext) -> Dict[str, Any]:
        metadata = context.metadata or {}
        payload: Any = metadata.get("submission")
        if isinstance(payload, dict):
            submission = payload
        else:
            text = payload if isinstance(payload, str) else context.get_user_input()
            if not text.strip():
                raise ValueError("Submission payload required. Provide JSON with agent_image.")
            try:
                submission = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid submission JSON: {exc}") from exc
            if not isinstance(submission, dict):
                raise ValueError("Submission payload must be a JSON object.")

        required = {"agent_image"}
        missing = sorted(required - submission.keys())
        if missing:
            raise ValueError(f"Submission missing required fields: {', '.join(missing)}")

        return submission

    async def _publish_results(self, context: RequestContext, event_queue: EventQueue, results: Dict[str, Any]) -> None:
        execution = results.get("execution", {})
        summary = {
            "constraints": results.get("constraints"),
            "performance": results.get("performance"),
            "execution": {
                "success": execution.get("success"),
                "results_path": execution.get("results_path"),
                "output_dir": execution.get("output_dir"),
                "time_seconds": execution.get("time_seconds"),
                "memory_used_mb": execution.get("memory_used_mb"),
            },
        }

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                context_id=context.context_id or "",
                task_id=context.task_id or "",
                artifact=new_data_artifact(
                    name="evaluation_summary",
                    data=summary,
                    description="TerminalBench evaluation summary.",
                ),
                last_chunk=True,
            )
        )

        logs = execution.get("logs")
        if isinstance(logs, str) and logs:
            display_logs = logs
            if len(logs) > self.MAX_LOG_CHARACTERS:
                display_logs = f"{logs[: self.MAX_LOG_CHARACTERS]}\n... [truncated {len(logs) - self.MAX_LOG_CHARACTERS} characters]"
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    context_id=context.context_id or "",
                    task_id=context.task_id or "",
                    artifact=new_text_artifact(
                        name="harness_logs",
                        text=display_logs,
                        description="Stdout/stderr captured from the harness run.",
                    ),
                    last_chunk=True,
                )
            )

    async def _send_status(self, context: RequestContext, event_queue: EventQueue, state: TaskState, message: Optional[str] = None, final: bool = False) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context.context_id or "",
                task_id=context.task_id or "",
                status=TaskStatus(state=state, message=new_agent_text_message(message) if message else None),
                final=final,
            )
        )


def create_terminalbench_app(
    task_config: Dict[str, Any],
    *,
    public_url: Optional[str] = None,
    agent_name: str = "TerminalBench Evaluator",
    agent_description: Optional[str] = None,
    extended_agent_card: Optional[AgentCard] = None,
) -> A2AStarletteApplication:
    url = public_url or os.environ.get("TERMINAL_BENCH_PUBLIC_URL", "http://localhost:8000")
    skill_example = json.dumps(
        {
            "agent_image": "my-terminal-agent:latest",
            "integration_mode": "mcp"
        },
        indent=2,
    )

    skill = AgentSkill(
        id="evaluate_terminalbench_submission",
        name="Evaluate TerminalBench submission",
        description="Run a submitted terminal agent on a TerminalBench task and return success/steps/time.",
        tags=["evaluation", "terminalbench", "docker"],
        examples=[skill_example],
    )

    capabilities = AgentCapabilities(streaming=True)
    card = AgentCard(
        name=agent_name,
        description=agent_description or "Evaluates terminal agents on TerminalBench tasks.",
        url=url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )

    executor = TerminalBenchExecutor(task_config)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())

    return A2AStarletteApplication(agent_card=card, http_handler=handler, extended_agent_card=extended_agent_card)
