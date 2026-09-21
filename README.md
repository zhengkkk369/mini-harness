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

- **Small, but complete.** About 3,100 lines of Python across 20 modules: eleven
  tools, context compaction, request retries, streaming responses, and session
  memory.
- **Tested offline.** `uv run pytest` runs 554 tests with no network, no API key
  and no Docker. They cover the agent loop, the tool executor's file-state
  gates, the tools, context compaction, retrievable memory, tool exposure,
  embedding backends, MCP bridging, configuration, the sandbox command builder,
  and the TUI's worker protocol.
- **Tools defined with Pydantic.** Typed inputs, generated JSON Schema, and
  validation before execution make tools easier to compose and orchestrate.
  The definition contract is enforced in code, not just written in the prompt,
  and tools can be bridged in from an [MCP](https://modelcontextprotocol.io) server.
- **Memory that survives compaction, and a tool surface that can be capped.**
  What compaction removes stays searchable through `recall`, ranked lexically by
  default or by any embeddings provider behind a
  [pluggable backend](src/mini_harness/embed.py). When the tool list grows past a
  budget, the best-ranked tools are exposed and `find_tools` is the way back to
  the rest.
- **Bounded and observable.** Optional token, cost and wall-clock budgets stop a
  run before it gets expensive, and an opt-in JSONL [trace](src/mini_harness/trace.py)
  records every turn, tool call, retry and compaction. The measured cost of that
  trace is in [EXPERIMENTS.md](EXPERIMENTS.md).
- **Concurrent where it is safe, refused where it is not.** Read-only tool
  batches and batches of subagents run in a thread pool; a policy layer refuses
  denied tools and commands before approval, and there is a read-only mode.
  Finishing with unverified edits costs one bounded extra turn.
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
export MINI_HARNESS_PRICE_IN=0.15        # dollars per million uncached input tokens
export MINI_HARNESS_PRICE_OUT=0.60       # dollars per million output tokens
export MINI_HARNESS_PRICE_CACHE_IN=0.003 # dollars per million cached input tokens
```

Set `MINI_HARNESS_PRICE_CACHE_IN` too if you want a real figure: cached input can
be most of the input, and pricing it at the uncached rate makes the estimate a
conservative upper bound instead of an accurate one. In the recorded runs 95% of
input was cached, and a single input price overstated the cost by 4.2x.

Cost stays at zero unless both prices are set, so a token budget needs no
pricing data. `Result` reports `cost` and which budget stopped the run, and the
CLI turns a spent budget into exit code 5.

Set `MINI_HARNESS_TRACE` to a path to record what the run actually did. The
[trace](src/mini_harness/trace.py) is an append-only JSONL file, one object per
event, with a monotonic `seq` and a `ts`: `run_start`, `turn`, `usage`,
`tool_call`, `tool_result`, `batch_parallel`, `retry`, `compact`,
`subagent_start`, `subagent_end`, `verify_nudge`, `budget_stop`, `run_end`. It is
off by default and every emit is a no-op when it is off, so it costs nothing
when unused. The file is opened once per run and flushed per event, so a crash
keeps everything already written. A write failure disables the trace instead of
failing the run. The bench profile writes one next to its session file. The
session file and the compaction audit log are unchanged: the trace is for
observing a run, not for resuming it.

## Execution policy and verification

A tool call passes four independent checks, cheapest first:

1. **Policy** — no human needed, so it also spares one the question. Deny tools
   by name, or deny commands by pattern. In `read_only` mode everything that can
   mutate is refused: `write_file`, `edit_file`, `run_bash`, `run_sandbox`,
   `run_subagent`.
2. **File-state gate** — an edit needs a prior read of that exact file, and the
   file must not have changed since.
3. **Approval** — risky tools ask, unless the caller supplies its own rule.
4. **Duplicate check** — the same call with the same arguments twice in a row.

A policy refusal is tagged `policy_denied` rather than `denied`, so it stays
distinguishable from a human saying no.

```sh
export MINI_HARNESS_DENY_TOOLS=run_sandbox          # comma separated tool names
export MINI_HARNESS_DENY_PATTERNS='rm -rf*,curl * | sh'
export MINI_HARNESS_READ_ONLY=true                  # analyse, never mutate
```

Patterns are matched against the `command` argument when the tool takes one, and
against the raw arguments otherwise.

Read-only tool batches (`read_file`, `grep_file`, `glob_file`) run concurrently,
because they cannot affect each other. A batch of subagents does too: they are
independent by construction and spend their time blocked on their own model
calls. Anything else — a write, a shell call, an unknown name, or a subagent
mixed with other tools — makes the whole batch serial. Results are returned in
the order the model asked for them regardless, and the file-state bookkeeping is
lock-protected. Approval for a batch is collected once per call, on the calling
thread, before anything runs, so prompts never interleave.

```sh
export MINI_HARNESS_PARALLEL_TOOLS=false   # force serial execution
export MINI_HARNESS_MAX_PARALLEL_TOOLS=4   # workers per batch
```

The system prompt asks the agent to verify its work before finishing. That is
now also a mechanism: if a run made edits and nothing has been run since, the
agent gets one bounded extra turn asking it to run something, and `Result.verified`
reports whether it did. Measured cost and behaviour are in
[EXPERIMENTS.md](EXPERIMENTS.md).

```sh
export MINI_HARNESS_VERIFY_REQUIRED=false  # do not spend the extra turn
export MINI_HARNESS_VERIFY_NUDGES=2        # how many times to ask
```

`verified` means a shell command succeeded after the last edit. It does not prove
the command tested the change, and it is reported as a flag rather than treated
as proof.

## Retrievable memory

Compaction is lossy by design: when the conversation passes `compact_limit`, a
sub-model replaces the oldest turns with a summary. The removed messages were
always written to `mini_harness_history.jsonl`, but only as an audit trail —
nothing read them back, so a detail the summary dropped was gone for the rest of
the run.

`recall(query)` closes that loop. It scores the archived messages against the
query and returns the best matches, so the agent can get back to an exact path,
value, command or error message instead of guessing or re-reading files:

```
[recall]: 3 of 412 archived messages match 'deploy window'

--- user (score 2.41, archived 04:52:11) ---
the deploy window is 02:00-04:00 UTC on weekdays
```

Retrieval is lexical — token overlap weighted by inverse document frequency — so
it needs no model, no embedding and no new dependency, and the same query always
gives the same answer. The injected compaction summary tells the agent the
archive exists and how to search it. The journal format is unchanged and stays
readable by [`bench/atif.py`](bench/atif.py), which reconstructs compaction
boundaries from it.

```sh
export MINI_HARNESS_RECALL=false        # no recall tool at all
export MINI_HARNESS_RECALL_LIMIT=10     # matches per query, default 5
export MINI_HARNESS_RECALL_SNIPPET=800  # characters per match, default 400
```

### The ranking is pluggable

Lexical scoring is exact, free and dependency-free, but it cannot match a query
to a message that says the same thing in different words. `recall_backend`
selects how the archive is ranked:

| Backend | Ranking | Needs |
| --- | --- | --- |
| `lexical` (default) | term overlap weighted by IDF | nothing |
| `hash` | deterministic local vectors | nothing; offline stand-in for the vector path |
| `vector` | cosine similarity between embeddings | an embeddings endpoint |
| `hybrid` | the two rankings fused by reciprocal rank | an embeddings endpoint |

```sh
export MINI_HARNESS_RECALL_BACKEND=hybrid
export MINI_HARNESS_EMBED_MODEL=text-embedding-3-small
export MINI_HARNESS_EMBED_BASE_URL=https://api.openai.com/v1   # an /embeddings provider
export MINI_HARNESS_EMBED_API_KEY=...                          # defaults to the main key
export MINI_HARNESS_EMBED_BATCH=96                             # texts per request
```

The default provider for this harness serves no `/embeddings` route, so `vector`
and `hybrid` need `embed_base_url` pointed at a provider that does. `hybrid` is
the safe choice of the two: the lexical ranking keeps its place, and the vector
half can still surface a message lexical scoring missed.

Two things worth knowing before switching:

- **Embedding sends the archive to that provider.** It is the conversation
  compaction removed, so this is a data-egress decision, not just a cost one.
- **The archive is embedded once.** Vectors are cached per text for the process,
  unit-normalised on the way in, so a repeated query embeds nothing. A provider
  failure is not fatal: the tool falls back to lexical ranking and says so in its
  output, so a run never loses its memory to an API error.

`hash` exists so the vector path, the cache and the fusion can be exercised
without a network call. It hashes tokens, so it is a weaker lexical scorer, not a
semantic one — [EXPERIMENTS.md](EXPERIMENTS.md) reports what it measures, and is
explicit that it says nothing about the quality of a real embedding model.

## MCP servers

Tools can also come from an [MCP](https://modelcontextprotocol.io) server: a
subprocess speaking JSON-RPC over stdio. The bridge performs the handshake, asks
for the tool list, and wraps each remote tool in a `ToolDefinition`, so the model
sees it exactly like a local one — same registry, same policy, same approval,
same trace.

```sh
export MINI_HARNESS_MCP_SERVERS='fs=python -m fs_server /tmp;db=python -m db_server'
uv run --locked mini-harness
```

Entries are separated by semicolons because a command contains spaces. Each is
`name=command`, and its tools appear as `name__tool`, so two servers cannot
collide. The CLI and the TUI own the server processes and stop them when the run
ends.

Two deliberate choices:

- **The server's own schema is what the model is told.** Converting a JSON Schema
  through a Pydantic model would flatten detail the server published, so the
  declared schema is passed through unchanged; a generated model is used only to
  refuse an obviously wrong call before it reaches the server.
- **Remote tools ask for approval by default**, because a server can do anything.
  Set `MINI_HARNESS_MCP_RISKY=false` to trust one.

One unreachable server never stops a run: the failure is printed and traced as
`mcp_error`, and the rest start normally.

```sh
export MINI_HARNESS_MCP_TIMEOUT=10   # seconds, bounds the handshake and every call
```

This was validated end to end against two real servers: the
`@modelcontextprotocol/server-filesystem` (v0.2.0, 14 tools) and the
`@modelcontextprotocol/server-memory` (v0.6.3, nine tools). Between them they
covered reading, creating, editing, moving, and state that survives across calls
to a second server. That run is recorded in [MODEL_EVAL.md](MODEL_EVAL.md).

A tool list longer than one page is followed through `nextCursor`, and a server
that never stops paging is cut off rather than trusted. Neither of the two
servers tested actually pages its list, so that path is only covered by tests.

### What a server does not receive

Each server gets the same environment as the agent's shell, and credential-shaped
variables are filtered out of that. A server authenticating through
`GITHUB_TOKEN`, `BRAVE_API_KEY` or anything matching `*KEY*`, `*TOKEN*`,
`*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`, `*_PWD` or `*AUTH*` receives nothing and
has to read its own configuration instead. There is also no per-server
environment, so two servers cannot be given different values of one variable.

### What MCP tools are not subject to

The workspace guards described above apply to the **built-in** file tools. A
bridged tool is a remote call, so `guard_read`, `guard_write` and the sensitive
file deny list do not reach it: the boundary is whatever the server was granted.
In the run above, `read_file` refused a path outside the workspace while
`fs__read_text_file` was refused by the *server* for the same reason, on the
server's own rules.

The controls that do apply to a bridged tool are the dispatch policy
(`policy_deny_tools`, `policy_deny_patterns`, `read_only`), approval (remote
tools are risky by default), the executor's argument validation, and the trace.
Treat an MCP server as a trusted component you chose to run.

### A note for Python MCP servers

The protocol is UTF-8 on stdout. A Python server on a non-UTF-8 console has to
reconfigure its own stdout, or it dies the first time it sends a non-ASCII
character:

```python
sys.stdout.reconfigure(encoding='utf-8', newline='\n')
```

The client reads UTF-8 regardless of console locale, so the failure is entirely
server-side — and it looks like the server crashing rather than an encoding
problem.

## On-demand tool exposure

Every tool schema is sent on every request, so the tool surface is a fixed cost
per turn whether or not the model uses it. It is bounded for the built-ins and
unbounded once servers are attached: a bridged surface of 24 tools plus the
built-ins serialises to about 17,900 characters (~4,500 tokens) of schema on
every single request.

`tool_budget` caps how many tools are exposed. Tools are ranked against the
current task — the same IDF scoring `recall` uses — and the best matches fill the
budget:

```sh
export MINI_HARNESS_TOOL_BUDGET=12   # unset (default) exposes every tool
```

Two rules keep the cap safe rather than lossy:

- **The core four are never hidden** — `find_tools`, `read_file`, `grep_file`,
  `glob_file`. Locating and reading is where every task starts, and `find_tools`
  is the way back to everything else.
- **A tool the model asked for stays exposed**, and a call to a hidden tool is
  answered with an instruction to use `find_tools` rather than executed. Guessing
  a name gets nowhere, so nothing is reachable only by luck.

```
[find_tools]: 1 of 35 tools match 'search a vector index for similar documents'; they are available from the next turn

- vector_search: search a vector index for similar documents
```

With 24 bridged tools and a budget of 8, the request carries 5,834 characters
instead of 17,869 — a third of the schema payload — and the ranked tool is back
in one call. The measurement, including that recovery, is in
[EXPERIMENTS.md](EXPERIMENTS.md).

The budget is off by default, and deliberately conservative when on: the ranking
is lexical, so a task whose wording does not resemble a tool description is
exactly where a cap would hurt. The `bench` profile pins it to `0`.

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
| `tests/test_trace.py` | the JSONL event log, flushing and failure handling |
| `tests/test_contract.py` | the tool-definition and registry contracts |
| `tests/test_parallel.py` | batch overlap (proved with a barrier, not a stopwatch) and eligibility |
| `tests/test_policy.py` | deny rules, read-only mode and dispatch attribution |
| `tests/test_memory.py` | journal indexing, lexical ranking, and the recall tool |
| `tests/test_embed.py` | the embedders, the cache, rank fusion, and the lexical fallback |
| `tests/test_selector.py` | the exposure budget, the ranking, `find_tools`, and the refusal of a hidden call |
| `tests/test_mcp.py` | handshake, tool discovery, dispatch, timeouts and failure handling |
| `tests/test_history.py` | repairing a stored conversation so the API accepts it |
| `tests/test_tui.py` | the TUI worker protocol, driven as a real subprocess in demo mode |

For measurements rather than pass/fail, see [EXPERIMENTS.md](EXPERIMENTS.md) and
`uv run python -m bench.experiments`.

Two environment notes. `run_bash` spawns a real shell, so its tests record the
subprocess call rather than capturing a child's output, which keeps them
working in environments that forbid pipes. Temporary directories are created
under `.pytest-work/` by the `workspace` and `session_dir` fixtures instead of
pytest's `tmp_path`, because some sandboxes make `mkdtemp` directories
read-only.

## Evaluations

Two kinds of evidence live alongside the tests, and they answer different
questions:

| | measures | needs |
| --- | --- | --- |
| [EXPERIMENTS.md](EXPERIMENTS.md), `bench/experiments.py` | the cost and effect of a mechanism, with a scripted model | nothing |
| [MODEL_EVAL.md](MODEL_EVAL.md), `bench/mini_bench.py` | whether a real model solves tasks, and what a run costs in tokens and turns | an API key |

```sh
uv run python -m bench.experiments     # offline, scripted client, no API key
uv run python -m bench.mini_bench      # real model, 12 tasks x 4 configurations
uv run python -m bench.tui_run         # drive the TUI headlessly (needs textual)
```

`bench/mini_bench.py` checks every task with an assertion this process runs, so
a pass never depends on the model claiming success. These are small,
self-written tasks: they exercise the harness end to end and are not comparable
to SWE-bench or Terminal-Bench.

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
