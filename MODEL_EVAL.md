# Model-in-the-loop evaluation

Results from running mini-harness against the **real DeepSeek API**: the TUI, a
small task suite with deterministic checks, and an attempt at the repository's
documented SWE-bench path through Harbor.

This is a different kind of evidence from [EXPERIMENTS.md](EXPERIMENTS.md).
Those numbers measure the harness with a scripted client and no model. These
numbers involve a model, so they can answer whether a task actually got solved —
and they carry all the variance that comes with a model.

## Environment

| | |
| --- | --- |
| Model | `deepseek-v4-flash`, thinking enabled (the repository default) |
| API | `https://api.deepseek.com`, reachable through the machine's proxy |
| Python | 3.12.13 |
| Docker | 29.2.1, daemon running |
| Machine | Intel Core i5-10200H, 4 cores / 8 threads, Windows |

The API key is read from a gitignored `.env`. Nothing here records the key.

## A blocker worth documenting: NO_PROXY broke every HTTP client

On this machine `NO_PROXY` is `localhost,127.0.0.1,::1,[::1]`. The bracketed form
is a legitimate way to write an IPv6 loopback, but httpx 0.28 builds a
`URLPattern` from every entry and raises while doing it:

```
httpx.InvalidURL: Invalid port: ':1]'
```

That happens **inside client construction**, so it is not a per-request failure:
no OpenAI client, and therefore no agent, could be created at all. The same
defect stopped Harbor, whose registry client builds an async httpx client for
Supabase. It was diagnosed by elimination:

- `cmd /c "echo %NO_PROXY%"` after setting `$env:NO_PROXY` in the parent shell
  printed the override, but a Python child of that same shell printed the
  original value, and an in-process `os.environ` change did not reach a child
  Python either. A parent process therefore cannot fix this for its children on
  this machine.
- The fix has to run **inside** the process that builds the client.

So `mini_harness.config` now normalises the value at import
(`normalize_no_proxy()`), and `sitecustomize.py` on `PYTHONPATH` does the same
for third-party processes such as Harbor. Covered by three tests in
`tests/test_config.py`.

## 1. The TUI, driven headlessly

Textual's `run_test()` pilot replaces the terminal driver, so the interface runs
for real — including its agent subprocess — without a TTY. The prompt was
read-only so that no approval modal had to be driven.

```
prompt: "Use glob_file to list the Python files under src/mini_harness, then read
         src/mini_harness/budget.py and tell me how many lines it has."

phase  : COMPLETED
tokens : prompt=4693 output=193 calls=2
events : 5

[user]      Use glob_file to list the Python files under src/mini_harness, then read ...
[assistant] I'll locate the Python files under src/mini_harness and read budget.py.
[tool]      glob_file status=OK args={"pattern": "src/mini_harness/*.py"}
[tool]      read_file status=OK args={"file_path": "src/mini_harness/budget.py"}
[assistant] The glob found ten Python files under src/mini_harness (...) I read budget.py
            in full and it has 93 lines, covering the Budget dataclass with token, cost,
            and wall-clock tracking plus the from_config, add, exceeded, render, and
            limits methods.
```

The count and the line count are both correct, so the whole path works: TUI
event loop, JSON worker protocol, real tool execution, real model, streaming
into the feed.

## 2. Mini benchmark: 15 tasks, 4 configurations, 2 repeats

`bench/mini_bench.py` — every task is stdlib-only and every check is an
assertion run by the harness process, so a pass never depends on the model
saying it passed. There are fifteen tasks: four state their contract only in
prose, three ship a runnable specification, and the rest ship a checker in the
sandbox. Whether the code is ever run is entirely the model's choice.

| config | what changes |
| --- | --- |
| `baseline` | repository defaults |
| `no_verify` | `verify_required=False` |
| `serial_tools` | `parallel_tools=False` |
| `unverifiable` | `run_bash` and `run_sandbox` denied, so the nudge cannot be satisfied |

### Aggregate over 72 runs of the first twelve tasks

| config | passed | rate | nudges | total tokens |
| --- | ---: | ---: | ---: | ---: |
| baseline | 23/24 | 96% | 0 | 915,522 |
| no_verify | 24/24 | 100% | 0 | 972,039 |
| serial_tools | 24/24 | 100% | 0 | 1,003,774 |

**The suite still cannot rank the configurations.** 71 of 72 runs passed, and the
mechanisms under test never fired in any of them. What it does show is that the
tasks are solved robustly and that the harness behaves the same with the
mechanisms on and off. See Limitations.

One mechanism did engage on its own: the model emitted several side-effect-free
calls in a single turn in most runs, so `parallel_tools` produced real batches
(1–4 per task) in `baseline` and `no_verify`, and none in `serial_tools`, as it
must. The concurrent path is exercised in normal use rather than only by tests.

### The one failure was my defect, not the model's

`baseline/numeric_ids` failed on one of its two repeats. The trace shows the
model read the file, grepped for a checker, and then **stopped without editing
anything** — `mutations=0`, three turns — with this answer:

> How should a *mixed* list behave, e.g. `["10", "2", "a"]`? ... Both satisfy
> your stated examples (`2` < `9` < `10`), so I can't derive the mixed case.

It was right. The task said "numerically when they are all digits, and lexically
otherwise", and the checker exercised only an all-digit list and an all-letter
list, so **both readings passed**. The model noticed the ambiguity and asked
rather than guessing, which is what the local system prompt asks for ("when the
task is ambiguous ... ask before acting"). `run_task` is unattended, so nobody
could answer and the run simply ended.

Two things came out of it:

- The task now settles the mixed case explicitly and the checker tests it. The
  fixed task passes 4/4.
- A run that edits nothing reports `verified=True`, because there are no
  outstanding edits to verify. That is vacuously true rather than reassuring, so
  the report prints `n/a` when a run made no edits; `Result.mutations` is the
  field to read alongside the flag.

### The verification loop, where it can be observed

Because the nudge never fires on tasks the model verifies itself, a fourth
configuration removes the ability to verify at all: `run_bash` and `run_sandbox`
are denied by policy.

| | |
| --- | --- |
| runs | 8 |
| passed | 8/8 |
| nudges fired | **8 — exactly one per run** |
| `verified` | `0/2` on every task |
| median turns | 9–12 |
| cost | $0.0181 |

- **It fires under its precondition.** With verification impossible, every run
  stopped unverified and every run was nudged.
- **It is bounded.** Exactly one nudge per run, never a loop, so a run that
  cannot satisfy it still terminates.
- **It reports the truth.** `verified` is false in all eight, because nothing
  could be run.
- **Its benefit is still unmeasured.** All eight tasks passed anyway. Denying the
  shell did not stop the model from reading the contract and editing the file; it
  only removed the ability to check the result. Showing a benefit needs tasks
  where the first attempt is wrong, and these are not those.

This configuration also exposes the mechanism's criterion. It asks whether a
shell command *succeeded after the last edit*, not whether that command tested
the change. The failure above is a case in point: `verified` was true there
because nothing had been edited at all, and the criterion would not have objected
even if something had, since a command that proves nothing satisfies it just as
well as one that proves the fix.

### A harder tier, and the outlier that repeats removed

The ceiling was the next thing to attack. If the model solves every task, turning
a mechanism on and off cannot change the outcome, so three tasks were added whose
specification is a **runnable test file inside the sandbox**. The prompt says
where the file is and forbids editing it, and deliberately does not restate the
cases, so getting it right means going to read it and running it. The harness
check refuses a modified suite as well as a failing one, so weakening the test is
not a solution — the rule the prompt states is enforced rather than trusted.

| task | what the shipped suite pins down |
| --- | --- |
| `rolling_window` | window count, a window as long as the input, a window longer than it, an empty input |
| `round_half_up` | ties away from zero, both signs, and the same rule at one decimal place |
| `sample_variance` | the sample (n-1) definition, one value, and an empty list |

Thirty runs: three tasks, two configurations, five repeats.

| config | passed | nudges | median turns | median tokens | total cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 15/15 | **0** | 9 | 50,212 | $0.0343 |
| `no_verify` | 15/15 | **0** | 8 | 43,858 | $0.0404 |

Three readings, and the second one is the point:

- **The harder tier did not lift the ceiling.** These tasks are harder to *read*
  — nothing in the prompt says what the expected values are — but not harder for
  this model, which finds `test_*.py`, runs it, and fixes the code. The ceiling is
  a property of small tasks and a capable model, not of task wording.
- **The nudge never fired: 0 times in 30 runs with `verify_required` on.** Its
  precondition is "the run is about to stop with unverified edits", and a model
  that runs the shipped suite after editing never satisfies it. Adding tasks
  cannot show the mechanism's benefit; on this evidence the only way to observe
  it is to remove the ability to verify at all, which is what the `unverifiable`
  configuration does — and there its remedy is unavailable by construction.
- **Repeats changed the answer, which is the reason for them.** The same tier at
  two repeats suggested `no_verify` cost 3.5x the tokens (882,901 against
  253,141) and looked like a real effect. At five repeats it is gone: the mean
  difference is +11,921 tokens with a two-sided permutation p = 0.77 (turns
  p = 0.51, wall time p = 0.93, 20,000 permutations), and the medians point the
  other way. A single 26-turn run had produced the entire gap. Same tasks, same
  runner, more samples.

### Cost, and why the cache rate matters

Prices are the provider's published ones, fetched on the day of the run, for
`deepseek-flash` — which is also what the legacy name `deepseek-v4-flash`
resolves to:

| per 1M tokens | off-peak | peak |
| --- | ---: | ---: |
| input, cache hit | $0.003 | $0.006 |
| input, cache miss | $0.15 | $0.30 |
| output | $0.60 | $1.20 |

The runs happened on a Sunday evening UTC, which is off-peak. Of the 412,099
input tokens the eight priced runs sent, **392,832 came from the cache — 95.3%**.
That is expected rather than surprising: every turn resends the same system
prompt and tool schemas.

One input price cannot express that. `Budget` now takes an optional
`price_cache_in` and splits the prompt, and the trace carries the per-turn cached
count. Without the split those eight runs look like $0.0758 instead of $0.0181 —
an overstatement of **4.2x**.

Pricing stays optional: unset means cost is zero, and setting
`price_in`/`price_out` without `price_cache_in` bills cached input at the full
rate, which is an upper bound rather than an undercount.

## 3. The MCP bridge, against a real server

The bridge was validated against the actual
[`@modelcontextprotocol/server-filesystem`](https://github.com/modelcontextprotocol/servers)
(v0.2.0), started through `npx`, exposing a scratch directory.

```sh
export MINI_HARNESS_MCP_SERVERS="fs=npx -y @modelcontextprotocol/server-filesystem /path/to/dir"
```

| step | outcome |
| --- | --- |
| Handshake with the real server | worked; it identifies as `secure-filesystem-server` 0.2.0 |
| Tool discovery | 14 tools, with required-argument lists read correctly |
| `list_allowed_directories`, `list_directory`, `read_text_file`, `get_file_info` | all returned results |
| Missing required argument | refused locally as `invalid_args`, before the call |
| A path outside the granted directory | refused **by the server** |
| `policy_deny_tools=('fs__write_file',)` | refused as `policy_denied` |
| Non-ASCII round trip | exact, including an em dash |

Then a real model, through the CLI:

```
[mini_harness]: profile = local, work_space = ..., guard = True/True, ..., mcp = 14 tool(s)

fs__list_allowed_directories: {}
fs__list_directory:        {"path": "...\\exposed"}
fs__read_text_file:        {"path": "...\\exposed\\notes.md"}
fs__list_directory:        {"path": "...\\exposed\\config"}
answer: "The release train leaves at 17:45 UTC, per notes.md. The config
         subdirectory contains a file named app.toml."

outcome completed, 4 turns, 4 calls, 0 failures, 22,981 prompt tokens,
239 completion tokens, 7.1 s
```

Both facts in the answer are correct, and the model used only bridged tools even
though the local `read_file` and `glob_file` were available alongside them.

### Three defects this found, that the stub server could not

A stub I wrote myself agreed with my own assumptions. A real server did not:

1. **`npx` could not be started by name.** On Windows it is `npx.CMD`, and
   `CreateProcess` does not apply `PATHEXT`, so spawning `npx` raised
   `OSError: [WinError 2]`. The bridge now resolves the executable through
   `shutil.which` first.
2. **One bad byte killed the reader.** The pipe was opened in text mode without
   an encoding, so it was decoded with the machine's locale codec (GBK here). A
   single non-GBK byte raised `UnicodeDecodeError` **inside the reader thread**,
   which died silently; every later call then sat out its full timeout and looked
   like a slow server rather than a broken pipe. The pipe is now UTF-8 with
   `errors='replace'`.
3. **A dead server kept callers waiting.** Even with the decode fixed, a server
   that exits mid-call left the request waiting for the whole timeout. The reader
   now marks the client closed and wakes anyone waiting, so the failure is
   immediate and named.

All three have regression tests, including one that kills a server mid-call and
asserts the caller is released quickly.

### A second server, and the write path

The read path above was one server and mostly reads. A second pass bridged two
real servers at once and exercised writing:

| | |
| --- | --- |
| servers | `secure-filesystem-server` 0.2.0 (14 tools) and `memory-server` 0.6.3 (9 tools) |
| bridged tools | 23, with no name collisions between the two |
| failures | none |

The filesystem write path, all through the bridge:

| call | result |
| --- | --- |
| `fs__create_directory` | created |
| `fs__write_file` | wrote `retries = 3` |
| `fs__edit_file` | rewrote it to `retries = 5`, returning a diff |
| `fs__move_file` | moved the file into place |
| `fs__read_text_file` | read it back |

Checked against the disk rather than against the server's own reply: the file
exists, contains `retries = 5`, and the pre-move path is gone. The memory server
was written to and then queried — `create_entities`, `create_relations`,
`search_nodes` — and the entity came back with its relation, so state really does
survive across calls to a second, independent server.

`fs__write_file` also asked for approval, like any other risky tool.

### Pagination is implemented, but no real server exercised it

`list_tools` now follows `nextCursor` instead of taking the first page and
silently reporting a subset, and it stops after 50 pages rather than trusting a
server that never ends. That behaviour is covered offline against a stub with
`--paginate=2` and `--paginate-loop`.

**Neither real server pages its tool list** — both answered in one page — so this
path is stub-verified only. It is implemented because the spec allows it, not
because a server needed it.

### A limitation worth knowing before choosing a server

`MCPBridge` gives each server the same environment the agent's shell gets, and
that environment has credential-shaped variables filtered out of it. A server
that authenticates through `GITHUB_TOKEN`, `BRAVE_API_KEY` or anything matching
`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`, `*_PWD` or `*AUTH*`
therefore receives nothing.

That filter exists to keep secrets out of a shell the model can drive, and a
server the operator configured by hand is arguably a different trust context. For
now the consequence is a documented limit rather than an oversight, with a test
asserting it: a server needing a token has to read it from its own configuration
file. There is also no per-server environment, so two servers cannot be given
different values of one variable.

## 4. Harbor and SWE-bench Verified: one task, not solved

The repository's documented benchmark path is Harbor against SWE-bench Verified.
It ran end to end on this machine — one task, as a smoke test:

```sh
PYTHONPATH=.;.experiments/pyshim uv run --with harbor==0.20.0 harbor run \
  -d swebench-verified --agent bench.adapter:MiniHarnessAgent \
  --model deepseek/deepseek-v4-flash --env docker --n-tasks 1 --n-concurrent 1 --yes
```

| step | outcome |
| --- | --- |
| Install Harbor 0.20.0 | worked |
| CLI flags match the README | verified against `harbor run --help` |
| Registry query | initially failed with the same `InvalidURL`; worked after the `sitecustomize` fix |
| Dataset resolution | worked — selected `astropy__astropy-7606` |
| Environment build | worked — a 4.18 GB image |
| Agent install inside the container | worked (`uv tool install` over the network) |
| Agent run | completed, 79.6 s |
| Verifier | ran, **FAILED** |
| **Reward** | **0 / 1** |

### The agent's run, from its own telemetry

| | |
| --- | --- |
| outcome | `completed` |
| turns / calls / failures | 17 / 17 / 0 |
| tools | `run_bash` 8, `read_file` 4, `grep_file` 3, `edit_file` 2 |
| prompt tokens | 120,127 (last request 9,725) |
| completion tokens | 2,599 |
| wall | 79.6 s |
| verified / mutations | true / 2 |
| stopped by | nothing — it stopped on its own with 283 turns still available |

The trace written by the benchmark run holds 70 events (`run_start`, 17 ×
`turn`, 17 × `usage`, 17 × `tool_call`, 17 × `tool_result`, `run_end`), so the
event trace added earlier in this project works in the real evaluation flow, not
just in tests. The `verified` and `mutations` fields also survive into Harbor's
telemetry.

So the agent ran cleanly, made two edits, ran commands to check its work, and
still did not pass the task's FAIL_TO_PASS tests. That is what a 0 on a
SWE-bench task looks like from the harness side: no crash, no error tag, no
budget stop, just an incorrect patch.

### A last environment defect

Harbor's CLI **crashed after the trial finished**, while printing the summary
table:

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2022'
```

rich's legacy Windows renderer writes to a GBK console and cannot emit the
bullet rich puts in its table title. The trial had already been written to disk,
so the result survives; only the console summary is lost. It is a Harbor/rich
issue on a non-UTF-8 Windows console, not something this repository controls —
but it is why a run can look like a failure while having succeeded.

### What this is not

**One task is not a score.** The repository's historical numbers are 401/500 on
SWE-bench Verified; a single task says nothing about that, and this result must
not be read as a regression against it. Nothing here is comparable to it: same
model name, but a different source revision, a different day, and one sampled
task instead of 500.

## Limitations

- **Ceiling effect.** The mini suite is fifteen small tasks, and the model solved
  71 of 72 runs of the first twelve and 30 of 30 runs of the harder tier. It
  cannot rank configurations. Its value is that it exercises the harness end to
  end against a real model and produces token, turn and trace data — and that its
  one failure was informative.
- **Two repeats per cell is still few.** The turn and token differences between
  configurations should not be read as effects; only the mechanism counts
  (nudges, parallel batches) are structural rather than statistical. The harder
  tier above shows what more repeats do to a difference that two repeats
  suggested was large. The other configurations remain at two.
- **The ambiguity lesson.** One task was underspecified and the model asked
  instead of guessing, which in an unattended run scores as a failure. Any task
  whose stated examples do not pin down the expected behaviour will produce
  these, and they are the task author's defect.
- **The verification loop's benefit is unmeasured.** It fires only when a run
  stops unverified, which this model does not do on a task whose specification
  it can run — 0 nudges in 30 harder-tier runs with the mechanism on. What is
  measured is that it fires when it must, that it is bounded, and that it does
  not lie.
- **The TUI run is one prompt**, driven through a test pilot rather than a real
  keyboard. Approval modals, session switching and cancellation were not driven.
- **Task choice is mine.** These are the tasks I wrote; they are not a
  standardised benchmark and are not comparable to SWE-bench or Terminal-Bench.
- **One model, one machine, one day.** No repetition across time, and no second
  model to compare against.
- **The SWE-bench figure is a single task.** 0/1 is an anecdote that shows the
  pipeline works end to end; it is not a score and does not compare to the
  401/500 recorded in [BENCHMARKS.md](BENCHMARKS.md).
- **Prices move, and peak is double off-peak.** The cost figures come from the
  provider's published rates on the day, at off-peak. Re-check them before
  quoting any of these numbers, and pass `--price-in`, `--price-out` and
  `--price-cache-in` to reproduce them.

## Reproducing

```sh
uv run python -m bench.mini_bench                 # default configs, writes MINI_BENCH.json
uv run python -m bench.mini_bench --repeats 2     # repeat every cell
uv run python -m bench.mini_bench --only unverifiable --price-in 0.15 \
    --price-out 0.60 --price-cache-in 0.003       # the priced run above
uv run python -m bench.mini_bench --tasks rolling_window --tasks round_half_up \
    --tasks sample_variance --only baseline --only no_verify --repeats 5 \
    --out MINI_BENCH_HARD.json --price-in 0.15 --price-out 0.60 \
    --price-cache-in 0.003                        # the harder tier above
```

The task definitions are also covered offline, without a model:
`tests/test_mini_bench.py` checks that every task starts unsolved, that a
reference fix passes its checker, that a weakened or deleted test suite is
refused, and that every configuration override names a real config field.

The raw results of the recorded runs are in [MINI_BENCH.json](MINI_BENCH.json)
(the twelve-task suite), [MINI_BENCH_PRICED.json](MINI_BENCH_PRICED.json) (the
priced `unverifiable` run) and [MINI_BENCH_HARD.json](MINI_BENCH_HARD.json) (the
harder tier).
