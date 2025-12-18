# Terminal Bench - Agent Evaluation Framework

A benchmark for evaluating LLM agents on terminal-based programming tasks.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python launcher.py --task-ids dirhash-fast --model gpt-4o --dataset-path /path/to/tasks
```

## Project Structure

```
sauce/
├── green_agent/          # Evaluator agent (runs tests, scores output)
├── white_agent/          # LLM wrapper agent (being evaluated)
├── results/              # Evaluation outputs
├── launcher.py           # Main entry point
└── test_green_agent_evaluation.py  # Ground truth tests
```

## How It Works

1. **White Agent**: LLM-powered agent that executes bash commands to solve tasks
2. **Green Agent**: Evaluator that runs pytest tests and scores the solution
3. **Docker**: Tasks run in isolated containers

## Running Evaluations

```bash
# Single task
python launcher.py --task-ids dirhash-fast --model gpt-4o

# With custom dataset path
python launcher.py --task-ids dirhash-fast --model gpt-4o --dataset-path /Users/rohanaanegola/my_tasks/tasks
```

## Results

Results are saved to `results/`:
- `gpt-4o.jsonl` - Full trajectory with reasoning
- `evaluation_gpt4o_success.json` - Formatted results

## Evaluation Metrics

The green agent checks:
- Valid JSON output format
- Required fields present (ok, files_scanned, groups, etc.)
- Correct field types
- Performance (execution time < 1 second)

## Example Results

| Model | Tests Passed | Accuracy |
|-------|--------------|----------|
| gpt-4o | 19/19 | 100% |
| gpt-4o-mini | 0/19 | 0% |

## Faithfulness Validation
### Reproducing Terminal-Bench Baseline Comparison

To validate faithfulness, run the same tasks with both implementations and compare results:

**1. Run with this implementation:**
```bash
python launcher.py --model gpt-4o --all-tasks --dataset-path data/tasks --max-workers 1 --results-dir ./results/our_implementation
```

**2. Run with Terminal-Bench's original harness** (requires Terminal-Bench installation):
```bash
# Using Terminal-Bench's official harness
python -m terminal_bench.run_eval --tasks data/tasks --output baseline_results.jsonl
```

**3. Compare results** (requires a comparison script):
```bash
python scripts/validate_faithfulness.py --baseline baseline_results.jsonl --ours results/our_implementation/gpt-4o.jsonl
```

### Manual Validation Test Cases

**Test Case 1: Successful Task Completion (dirhash-fast)**
```bash
python launcher.py --model gpt-4o --task-ids dirhash-fast --dataset-path data/tasks --max-workers 1
```
Expected: `passed=true`, `score=1.0`, all pytest tests pass

**Test Case 2: Failed Task (Incomplete Solution)**
```bash
python launcher.py --model gpt-4o --task-ids example-fail-task --dataset-path data/tasks --max-workers 1
```
Expected: `passed=false`, `score=0.0`, some pytest tests fail

**Test Case 3: All Tests Pass (Parser Edge Case)**
```bash
python launcher.py --model gpt-4o --task-ids example-all-pass-task --dataset-path data/tasks --max-workers 1
```
Expected: `passed=true`, `score=1.0`, all tests pass (tests fallback parser)
