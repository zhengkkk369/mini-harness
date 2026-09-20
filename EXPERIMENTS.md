# Harness experiments

Measurements of the mini-harness mechanisms added in this repository: parallel
tool execution, subagent concurrency, the event trace, retrievable memory, the
verification loop, and the dispatch policy.

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
answer exactly. For numbers from a real model, see [MODEL_EVAL.md](MODEL_EVAL.md).

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
| 0 ms | 37.25 ms | 23.59 ms | 1.58x |
| 5 ms | 71.93 ms | 27.14 ms | 2.65x |
| 20 ms | 161.22 ms | 42.68 ms | 3.78x |

Reading this:

- The gain grows with per-call latency and approaches the worker count. That is
  the expected shape: with six calls in flight and no shared bottleneck, the
  floor is one call's latency rather than six.
- The speedup never reaches the ideal 6x. At 20 ms the concurrent batch takes
  42.7 ms, not the ~20 ms a perfect pool would give. Some of the per-call work
  (argument validation, the file-state record, output handling) still runs under
  the GIL, and the pool has its own start-up cost.
- Even with **no injected latency** the concurrent path is 1.58x faster, so real
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
| 20 ms | 82.45 ms | 24.14 ms | 3.42x |
| 100 ms | 402.54 ms | 104.00 ms | 3.87x |

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
| off | 103.44 ms | 92.09 ms | 160.08 ms | 0 |
| on | 98.17 ms | 95.18 ms | 120.04 ms | 10,868 |
| overhead | **-5.27 ms** | | | |

**The overhead is now below this measurement's noise floor.** In this run the
traced variant came out 5 ms *faster*, which is not a real effect: the spread
between the fastest and slowest run of the same configuration (92 to 160 ms) is
larger than the difference being measured. Earlier runs of the same experiment
put the overhead at +9.2 ms, +13.2 ms and +14.8 ms. The honest summary is
"somewhere between zero and about 15 ms for ~82 events, which is 0-15% of a run
whose only work is tool calls" -- and that the measurement is not precise enough
to say more without many more samples.

### This experiment found and fixed a real defect

The first version of the trace opened the file, wrote one line, and closed it
for **every** event. That was far above the noise:

| Trace implementation | Overhead, 21 turns | Bytes |
| --- | ---: | ---: |
| reopen the file per event | +171.92 ms | 10,910 |
| persistent handle, flushed per event (current) | +9.23 ms | 10,882 |

Both of those come from the same `--repeats 5` run of the same experiment. The
per-event `open`/`close` cost about 2 ms each on this platform, which is more
than the event is worth, so `Trace` now opens the file once per run and flushes
after each event. Flushing is kept so a crash still leaves everything already
written on disk.

This is the clearest argument for having run the experiment at all: the feature
looked correct, and its cost was an order of magnitude above the noise floor
once measured.

## 4. Retrievable memory

A planted fact is loaded into a journal alongside *N* deterministic distractor
messages, and `recall` is asked for the fact. Five different facts per size; a
hit counts only when the planted text comes back.

| Distractors | Archived messages | Top-1 | Top-3 | Search (median) |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 55 | 5/5 | 5/5 | 0.51 ms |
| 200 | 205 | 5/5 | 5/5 | 1.16 ms |
| 800 | 805 | 5/5 | 5/5 | 3.87 ms |

Reading this:

- Ranking is exact at every size tested: the planted fact is first, not merely
  present. The distractors share vocabulary with the queries (the filler includes
  words like `deploy`, `retry` and `config`), so this is not a trivial separator.
- Search cost grows linearly with the archive, because the scorer scans every
  entry. At 805 entries that is under 4 ms, which is nothing next to a model
  call; at hundreds of thousands of entries it would need an inverted index.
  This is the main scaling limit of the current implementation.
- Five facts per size is a small sample. It demonstrates that the ranking works,
  not how well it generalises to real conversation text, where queries are
  messier and the useful answer may share no rare token with them.

## 5. Verification loop

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

## 6. Dispatch policy

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
  It is also small enough that its cost sits at the noise floor of this
  measurement, so treat the trace overhead as "under ~15 ms for 82 events"
  rather than as a precise number.
- **Retrieval was measured on synthetic text.** Five planted facts per size,
  against filler that shares vocabulary with the queries. Real conversation
  queries are messier, and the useful message may share no rare token with them.

## Files

| Path | Contents |
| --- | --- |
| `bench/experiments.py` | the experiment runner; prints a report and writes the raw JSON |
| `EXPERIMENTS.json` | raw results of the recorded run |
| `tests/test_parallel.py` | overlap is proved with a barrier, not a stopwatch |
| `tests/test_policy.py` | policy decisions and dispatch attribution |
| `tests/test_trace.py` | trace format, flushing and failure handling |
