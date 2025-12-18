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
