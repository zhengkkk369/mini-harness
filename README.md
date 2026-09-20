<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img src="assets/logo-light.svg" alt="mini-harness" width="420">
  </picture>
</div>

<p align="center">
  <img src="assets/demo.gif" alt="Real DeepSeek Agent writes a Python file, runs it in a Docker sandbox, and answers a follow-up question" width="900">
  <br>
  <sub>Real DeepSeek V4 Flash · 12.8 s replay · <a href="assets/demo.mp4">120 fps video</a></sub>
</p>

## Why mini-harness?

- **Small, but complete.** About 1,700 lines of Python: nine tools, context
  compaction, request retries, streaming responses, and session memory.
- **Tested offline.** `uv run pytest` runs 258 tests with no network, no API key
  and no Docker. They cover the agent loop, the tool executor's file-state
  gates, the nine tools, context compaction, configuration and the sandbox
  command builder.
- **Tools defined with Pydantic.** Typed inputs, generated JSON Schema, and
  validation before execution make tools easier to compose and orchestrate.
  The definition contract is enforced in code, not just written in the prompt.
- **Bounded and observable.** Optional token, cost and wall-clock budgets stop a
  run before it gets expensive, and an opt-in JSONL [trace](src/mini_harness/trace.py)
  records every turn, tool call, retry and compaction.
- **A practical baseline.** Evaluated on SWE-bench Verified and Terminal-Bench
  2.1 with DeepSeek V4 Flash. See the results below.
- **Built for learning.** Follow the [agent loop](src/mini_harness/agent.py),
  [tools](src/mini_harness/tool/box.py), [compaction](src/mini_harness/compact.py),
  and [retry policy](src/mini_harness/retry_request.py) in ordinary Python.
  The optional [TUI](tui.py) is a single file you can read and modify.

## Benchmarks

Historical, self-reported results with **DeepSeek V4 Flash**. The latest source
changes have not been re-evaluated on these benchmarks.

| Benchmark | Solved / attempts | Score |
| --- | ---: | ---: |
| SWE-bench Verified | 401 / 500 | **80.2%** |
| Terminal-Bench 2.1 | 309 / 445 | **69.44%** |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/terminal-bench-ranking-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/terminal-bench-ranking-light.svg">
    <img src="assets/terminal-bench-ranking-light.svg" alt="Terminal-Bench 2.1 score comparison: mini-harness at 69.44% would place 13th among 18 public entries plus this self-reported run, as of September 4, 2026. This is not an official rank." width="900">
  </picture>
</p>

The figure inserts our score into the [public Terminal-Bench 2.1 leaderboard](https://www.tbench.ai/?version=2.1)
as of **September 4, 2026**; **13th of 19 is an illustrative position, not an official
rank**. The Terminal-Bench result averages five attempts on each of 89 tasks,
not pass@5. See [evaluation details, recorded failures, and reproduction limits](BENCHMARKS.md).

## Define a tool with Pydantic

A tool combines a Pydantic input model, a Python function, and a `ToolDefinition`.
After installing and exporting `DEEPSEEK_API_KEY` (see [Get started](#get-started)),
pass your definitions to the agent to choose which tools it can use:

```python
from pydantic import BaseModel, ConfigDict, Field

from mini_harness.agent import DeepSeekAgent
from mini_harness.tool.box import TOOLS, ToolDefinition


class CountWordsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, description="Text to count words in.")


def count_words(args: CountWordsInput) -> str:
    return str(len(args.text.split()))


count_words_tool = ToolDefinition(
    name="count_words",
    description="Count whitespace-separated words in text.",
    parameters=CountWordsInput,
    function=count_words,
    risky=False,
)

agent = DeepSeekAgent([*TOOLS, count_words_tool])
```

The agent uses `model_json_schema()` to describe each tool to the model.
Before dispatch, the [tool executor](src/mini_harness/tool/box.py) calls
`model_validate_json()` to validate its arguments; invalid calls return an error
for the agent to correct. One definition keeps the schema, validation, and
execution connected. See the [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/).

A `ToolDefinition` is checked when it is built, so a definition that breaks the
contract fails at import instead of at dispatch:

- `name` must be a valid API function name (letters, digits, `_`, `-`, up to 64).
- `description` must be non-empty; it is the model's only documentation.
- `parameters` must be a Pydantic `BaseModel` subclass.
- `function` must accept the arguments model as its first parameter, and take
  `cfg` as a normal keyword if it accepts it at all. Functions written as
  `def tool(args)` and as `def tool(args, cfg=None)` both work; the definition
  records which convention yours uses.

`validate_tools()` adds the checks a single definition cannot make: names must
be unique, the built-in read and write tools must be present, and any tool whose
file bookkeeping keys on `file_path` must declare that field. Skipping the last
rule is exactly how an edit gate silently stops gating.

## Budgets and tracing

A run stops for one of four reasons, and the reason is reported rather than
guessed at: the model stopped calling tools (`completed`), the turn limit was
reached (`exhausted`), a budget was spent (`timeout` for the wall clock,
`budget` for tokens or cost), or something failed.

Set any of these to bound a run. All are off by default:

```sh
export MINI_HARNESS_WALL_BUDGET=600      # seconds, checked before each request
export MINI_HARNESS_TOKEN_BUDGET=500000  # prompt plus completion tokens
export MINI_HARNESS_COST_BUDGET=2.50     # US dollars, needs both prices below
export MINI_HARNESS_PRICE_IN=0.28        # dollars per million input tokens
export MINI_HARNESS_PRICE_OUT=0.42       # dollars per million output tokens
```

Cost stays at zero unless both prices are set, so a token budget needs no
pricing data. `Result` reports `cost` and which budget stopped the run, and the
CLI turns a spent budget into exit code 5.

Set `MINI_HARNESS_TRACE` to a path to record what the run actually did. The
[trace](src/mini_harness/trace.py) is an append-only JSONL file, one object per
event, with a monotonic `seq` and a `ts`: `run_start`, `turn`, `usage`,
`tool_call`, `tool_result`, `retry`, `compact`, `subagent_start`,
`subagent_end`, `run_end`. It is off by default and every emit is a no-op when
it is off, so it costs nothing when unused. A write failure disables the trace
instead of failing the run. The bench profile writes one next to its session
file. The session file and the compaction audit log are unchanged: the trace is
for observing a run, not for resuming it.

## Get started

### 1. Download and install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not
have it, then clone the repository. Python 3.12+ is required; uv can install it.

```sh
# macOS / Linux — install uv once, then restart your terminal
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```sh
git clone https://github.com/mini-harness/mini-harness.git
cd mini-harness
uv python install 3.12
uv sync --locked
```

### 2. Connect the agent

Set your [DeepSeek API key](https://platform.deepseek.com/api_keys) in the shell,
then launch the TUI. No `.env` file is needed.

```sh
export DEEPSEEK_API_KEY="your-api-key"
uv run tui.py
```

`Enter` sends a message · `Ctrl+N` starts a session · `F2` opens sessions ·
`Ctrl+E` expands tools · `Ctrl+C` cancels · `Ctrl+Q` quits.

The agent reads the project workspace; local file tools write to `sandbox/`.
The TUI asks before shell, sandbox, or subagent calls. Sessions stay in `.local/` and are
ignored by Git. Cancelling a task keeps file edits already completed.

To run generated code with `run_sandbox`, install and start
[Docker](https://docs.docker.com/get-started/get-docker/), then pull its Python image once:

```sh
docker pull python:3.12-slim
```

`run_sandbox` is a Pydantic-defined tool accepting `command` and
`timeout_seconds` (1–300, default 30). It runs in a disposable Python 3.12
container with networking disabled, a read-only system, and limits of one CPU,
256 MB RAM, and 64 processes. Only `sandbox/` is mounted at `/workspace`; changes
there persist. For example, `{"command": "python hello.py"}` runs
`sandbox/hello.py`. No host environment variables are forwarded into the container.
The host `run_bash` tool remains available and is not isolated.

For the plain terminal interface:

```sh
uv run --locked mini-harness
```

## Tests

The suite is fully offline: it uses a dummy API key and fake model clients, so
it runs in CI and on a laptop without credentials. No test requires Docker.

```sh
uv run --locked pytest
```

| File | Covers |
| --- | --- |
| `tests/test_agent.py` | the turn loop, outcomes, truncation recovery, memory, telemetry, CLI exit codes |
| `tests/test_executor.py` | argument validation, duplicate calls, approvals, and the read/stale/overwrite file gates |
| `tests/test_tools.py` | glob, grep, read, write, edit and run_bash behaviour, including the environment filter |
| `tests/test_path.py` | read/write path guards and the sensitive-file deny list |
| `tests/test_compact.py` | cut-point selection, the summary prompt, and the compaction audit log |
| `tests/test_config.py` | provider inference, environment overrides, and the bench profile |
| `tests/test_sandbox.py` | the Docker command line, isolation flags, mount scope, and cleanup |
| `tests/test_budget.py` | token, cost and wall-clock arithmetic and the stop decision |
| `tests/test_trace.py` | the JSONL event log and retry backoff reporting |
| `tests/test_contract.py` | the tool-definition and registry contracts |

Two environment notes. `run_bash` spawns a real shell, so its tests record the
subprocess call rather than capturing a child's output, which keeps them
working in environments that forbid pipes. Temporary directories are created
under `.pytest-work/` by the `workspace` and `session_dir` fixtures instead of
pytest's `tmp_path`, because some sandboxes make `mkdtemp` directories
read-only.

## Benchmark adapters

Run these commands from the repository root after installation. The adapters in
[`bench/`](bench/) connect mini-harness to Harbor and SWE-bench cloud scoring.
The examples use OpenAI; for DeepSeek, export `DEEPSEEK_API_KEY` and change the
model to `deepseek/deepseek-v4-flash`.

### Harbor

With Docker running, evaluate one SWE-bench Verified task using the
[Harbor adapter](bench/adapter.py):

```sh
export OPENAI_API_KEY="your-openai-api-key"
PYTHONPATH=. uv run --with "harbor==0.20.0" harbor run \
  -d swebench-verified \
  --agent bench.adapter:MiniHarnessAgent \
  --model openai/gpt-4.1 --env docker --n-tasks 1 -n 1
```

Harbor runs the agent and verifier, saving results and ATIF trajectories under
`jobs/`. Remove `--n-tasks 1` to evaluate the full dataset; `-n` controls concurrency.
See [Harbor's custom-agent guide](https://www.harborframework.com/docs/agents).

### SWE-bench cloud scoring with sb-cli

The [cloud entry point](bench/sb_cli.py) uses Modal to run the agent, exports its
patches, then submits them to sb-cli for scoring. Set `OPENAI_API_KEY` as above,
obtain a [verified sb-cli API key](https://www.swebench.com/sb-cli/authentication/),
and sign in to [Modal](https://modal.com/docs/cli/latest/setup):

```sh
export SWEBENCH_API_KEY="your-verified-sb-cli-api-key"
uv run --with-requirements bench/requirements-sb-cli.txt modal setup
uv run --with-requirements bench/requirements-sb-cli.txt \
  python -m bench.sb_cli run \
  --job-dir jobs/sb-cli-smoke --run-id mini-harness-smoke \
  --model openai/gpt-4.1 --n-tasks 1
```

Add `--dry-run` to the final command to preview without launching a job. Replace
`--n-tasks 1` with `--all` for all 500 tasks, using a new job directory and run ID
for each run. Predictions are saved to `jobs/sb-cli-smoke/predictions.json` and
reports to `jobs/sb-cli-smoke/sb-cli-reports/`. The `generate`, `export`, and
`submit` subcommands also let you run each stage separately.
