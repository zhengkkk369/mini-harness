# Harness experiments

Measurements of the mini-harness mechanisms added in this repository: parallel
tool execution, subagent concurrency, the event trace, the verification loop,
and the dispatch policy.

## What these numbers are, and what they are not

These are **harness-level** measurements. Every run drives the real agent loop
and the real tools through a scripted model client, so the numbers describe what
the harness itself costs and saves. There is no model in the loop, no network,
and no API key involved.

That is a deliberate scope choice. It means these numbers say nothing about task
success rates, and they are **not** a substitute for the SWE-bench and
Terminal-Bench evaluations described in [BENCHMARKS.md](BENCHMARKS.md). What they
do establish is the cost of a mechanism and whether it does what it claims,
which is what a wall-clock and turn-count experiment with a fixed script can
answer exactly.

## Environment

| | |
| --- | --- |
| CPU | Intel Core i5-10200H, 4 cores / 8 threads |
| OS | Windows (see `platform` in `EXPERIMENTS.json`) |
| Python | 3.12.13 |
| Repeats | 7 per timed configuration (median reported) |

## Reproducing

```sh
uv run python -m bench.experiments                 # prints a report, writes EXPERIMENTS.json
uv run python -m bench.experiments --repeats 15    # more samples
```

The raw results of the recorded run are in [EXPERIMENTS.json](EXPERIMENTS.json).
Numbers will differ on other hardware; the shape of the results is what matters.

## 1. Parallel tool execution

A batch of six read-only `read_file` calls through
`ToolExecution.execute_batch()`, serial against concurrent. Tool latency is
injected with a fixed sleep in front of the real reader, to model tools that are
not instant (a network fetch, a subprocess, a large file). Six calls with
`max_parallel_tools` set to six.

| Injected latency | Serial (median) | Concurrent (median) | Speedup |
| ---: | ---: | ---: | ---: |
| 0 ms | 32.18 ms | 22.75 ms | 1.41x |
| 5 ms | 66.59 ms | 26.60 ms | 2.50x |
| 20 ms | 157.55 ms | 40.95 ms | 3.85x |

Reading this:

- The gain grows with per-call latency and approaches the worker count. That is
  the expected shape: with six calls in flight and no shared bottleneck, the
  floor is one call's latency rather than six.
- The speedup never reaches the ideal 6x. At 20 ms the concurrent batch takes
  41.0 ms, not the ~20 ms a perfect pool would give. Some of the per-call work
  (argument validation, the file-state record, output handling) still runs under
  the GIL, and the pool has its own start-up cost.
- Even with **no injected latency** the concurrent path is 1.41x faster, so real
  file reads and MD5 digests do overlap. I did not profile further, so I cannot
  attribute that split between filesystem concurrency and `hashlib` releasing the
  GIL.
- A batch is only ever overlapped when every call is safe to overlap:
  side-effect free (`read_file`, `grep_file`, `glob_file`), or a batch consisting
  entirely of subagents. One write, one shell call, one unknown name, or a mix of
  subagents with anything else makes the whole batch serial, which is asserted by
  `tests/test_parallel.py`.

## 2. Subagent concurrency

A batch of four `run_subagent` calls through the same path. A subagent spends
its time blocked on its own model calls, which is what the injected latency
models.

| Injected latency | Serial (median) | Concurrent (median) | Speedup |
| ---: | ---: | ---: | ---: |
| 20 ms | 82.23 ms | 23.95 ms | 3.43x |
| 100 ms | 402.64 ms | 103.90 ms | 3.88x |

This is close to the ideal 4x, and closer than the read batch gets, because the
stubbed subagent releases the GIL for the whole of its latency while a file read
does real CPU work around its I/O. It is the clearest case for overlapping:
independent units of work that each wait on the network.

Approval is collected for every call **before** the batch fans out, on the
calling thread, so prompts never interleave and a denial cannot race an
approval. A single denial drops the whole batch back to the serial path, which
keeps a refusal reading exactly as it did before this existed.

## 3. Event trace

Twenty tool turns plus a final answer, with the trace off and on. Each tool turn
emits `turn`, `usage`, `tool_call` and `tool_result`; the run adds `run_start`
and `run_end`, so about 82 events.

| Trace | Median | Min | Max | Bytes written |
| --- | ---: | ---: | ---: | ---: |
| off | 90.22 ms | 88.47 ms | 106.26 ms | 0 |
| on | 104.99 ms | 97.48 ms | 135.74 ms | 10,871 |
| overhead | **+14.77 ms** | | | |

That is roughly 0.18 ms and 133 bytes per event, or about 16% of a run whose only
work is tool calls. The variance is high (max 135.74 ms against a 97.48 ms
minimum) because the trace writes to disk on every event.

### This experiment found and fixed a real defect

The first version of the trace opened the file, wrote one line, and closed it
for **every** event. The same experiment measured:

| Trace implementation | Overhead, 21 turns | Bytes |
| --- | ---: | ---: |
| reopen the file per event | +171.92 ms | 10,910 |
| persistent handle, flushed per event (current) | +9.23 ms | 10,882 |

Both of those figures come from the same `--repeats 5` run of the same
experiment in this session; the recorded 7-repeat figure above is +14.77 ms. The
per-event `open`/`close` cost about 2 ms each on this platform, which is more
than the event is worth, so `Trace` now opens the file once per run and flushes
after each event. Flushing is kept so a crash still leaves everything already
written on disk.

This is the clearest argument for having run the experiment at all: the feature
looked correct and its cost was only visible when measured.

## 4. Verification loop

A scripted agent that reads a file, edits it, and then stops. `verify_required`
is on by default; the nudge is bounded by `verify_nudges` (1 here).

| Scenario | Turns | Mutations | Nudges | `verified` | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| off, edit then answer | 3 | 1 | 0 | False | completed |
| on, edit then answer | 4 | 1 | 1 | False | completed |
| on, edit then run then answer | 4 | 1 | 0 | True | completed |

Reading this:

- With verification off, the run stops unverified and **says so**: `verified` is
  False. The flag reports the truth whether or not the mechanism is enforcing it.
- With verification on, stopping unverified costs exactly one extra turn (3 to
  4) and one nudge.
- The nudge is bounded. In the second scenario the scripted agent ignores it,
  and the run still terminates after one nudge rather than looping; `verified`
  stays False.
- When the agent runs something after editing, no nudge is needed and `verified`
  is True — the mechanism distinguishes "ran something since the last edit" from
  "did not", and costs nothing in the good case.

What this does **not** measure is whether running something actually proves the
change works. The mechanism checks that a verification-shaped action happened,
not that it was a good one. A model could satisfy it with `echo`. That is a real
limitation of a mechanical check, and it is why the result is reported as a
`verified` flag rather than being treated as proof.

## 5. Dispatch policy

Policy decisions for a representative set of calls, with
`policy_deny_tools=('run_sandbox',)` and `policy_deny_patterns=('rm -rf*',)`.

| Call | Allowed | Tag |
| --- | --- | --- |
| `run_bash` with `rm -rf /` | no | `policy_denied` |
| `run_bash` with `ls -la` | yes | success |
| `run_sandbox` with `ls` | no | `policy_denied` |
| `read_file` on `tree/f000.py` | yes | success |

A policy refusal is tagged `policy_denied`, not `denied`, so it stays
distinguishable from a human refusing the same call. The policy is checked
before approval, so a denied call never reaches the human at all — asserted in
`tests/test_policy.py`.

In `read_only` mode the same rules deny `write_file`, `edit_file`, `run_bash`,
`run_sandbox` and `run_subagent` while leaving reads alone.

## Limitations

- **No model.** Task success, prompt-following and whether a nudge actually
  improves a model's behaviour cannot be measured here at all. Those need a
  model in the loop and belong with the real benchmarks.
- **One machine, one platform.** Timings come from a single 4-core Windows
  laptop. The parallel speedup in particular will vary with core count, disk and
  the tool mix.
- **Synthetic latency.** Both concurrency experiments inject sleeps rather than
  using genuinely slow work, because local file reads are too fast to show the
  effect. The `0 ms` row is the only one with no injected latency. The subagent
  stub returns a fixed string instead of running a nested agent, so it measures
  the batch scheduler, not real subagent work.
- **Parallel batches change event order.** Within one overlapped batch the
  `tool_call`/`tool_result` events interleave nondeterministically. Sequence
  numbers stay contiguous (they are assigned under a lock) and result order
  always matches the order the model asked for, but the trace file is not
  byte-stable between runs.
- **The 21-turn trace figure is small.** ~82 events is a short run; a long
  session writes proportionally more, and flush-per-event cost scales with it.

## Files

| Path | Contents |
| --- | --- |
| `bench/experiments.py` | the experiment runner; prints a report and writes the raw JSON |
| `EXPERIMENTS.json` | raw results of the recorded run |
| `tests/test_parallel.py` | overlap is proved with a barrier, not a stopwatch |
| `tests/test_policy.py` | policy decisions and dispatch attribution |
| `tests/test_trace.py` | trace format, flushing and failure handling |
