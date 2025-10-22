# Terminal-Bench: dirhash-fast (AgentBeats-style Green Agent)

This repo now focuses on the `dirhash-fast` task and includes a minimal Green Agent in `green/run_green.py`, following the simple grading style from AgentBeats green agents.

## Install

```bash
python3.10 -m pip install -r requirements.txt
```

Ensure Docker (with NVIDIA runtime) is running on a GPU host.

## Run the green agent locally

Repo structure (AgentBeats-style):

```
/white   # contestant submission (must contain Makefile that builds dirhash_fast)
/green   # grader
```

Green agent entry point:

```bash
python3 green/run_green.py
```

What it does:

- Builds the white agent: runs `make` in `../white`
- Runs the compiled binary with dataset at `/data`:
  - `./dirhash_fast /data --min-bytes 1024 --top 10 --threads 8`
- Measures wall-clock time (enforces 18s timeout)
- Parses the program stdout as JSON (last non-empty line)
- Passes if:
  - finishes within 18s
  - JSON parses
  - contains `ok: true` and keys: `files_scanned`, `bytes_scanned`, `groups`, `top`, `threads`, `ms`
- Otherwise prints a short failure reason

Output:

- `✅ PASS (time=X.XXs)` on success
- `❌ FAIL (<reason>)` on failure

## Docker and white agent notes

- To align with AgentBeats, you can package `white/` (the submission) in a Docker image and run the green agent inside a harness container. The simple local flow above does not require Docker; it runs directly on the host.
- If you prefer full containerized evaluation, adapt the existing `harness/` image and have it invoke `python3 /app/green/run_green.py` with `/white` and `/data` mounted inside the container.

Reference: AgentBeats repo and green agent orchestration concepts are documented in `agentbeats` (`green` as orchestrator): [agentbeats/agentbeats](https://github.com/agentbeats/agentbeats)
