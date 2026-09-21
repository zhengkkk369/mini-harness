# Harness experiments

Measurements of the mini-harness mechanisms added in this repository: parallel
tool execution, subagent concurrency, the event trace, retrievable memory, the
pluggable vector backend, the verification loop, the dispatch policy, and the
tool-exposure budget.

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

The same scripted run with the trace off and on, at two lengths: 20 tool turns
(84 events) and 200 (804 events). Each tool turn emits `turn`, `usage`,
`tool_call` and `tool_result`; the run adds `run_start` and `run_end`.

| Trace | Turns | Events | Median | Min | Max | Bytes written |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 20 | 0 | 213.30 ms | 176.74 ms | 285.70 ms | 0 |
| on | 20 | 84 | 238.64 ms | 180.87 ms | 687.28 ms | 11,440 |
| off | 200 | 0 | 2168.54 ms | 1457.16 ms | 3154.18 ms | 0 |
| on | 200 | 804 | 2371.81 ms | 1473.22 ms | 3722.76 ms | 108,747 |

Those medians are not the measurement. The two conditions are timed
**alternately inside each repeat** and the paired difference is what gets a
median, because timing one block and then the other measures the machine:

| Turns | Events | Overhead (paired median) | Min pair | Max pair | Per event |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 84 | **+2.93 ms** | -36.06 ms | +408.98 ms | 34.9 us |
| 200 | 804 | **+208.47 ms** | -41.57 ms | +1003.70 ms | 259.3 us |

Reading it:

- **At 84 events the cost is inside the noise**, as the earlier runs of this
  experiment found from the other direction (+9.2, +13.2, +14.8, +6.2, -5.3 ms).
  The paired spread (-36 to +409 ms) is wider than the effect.
- **At 804 events it is no longer inside the noise**: +208 ms for ten times the
  events, or about **0.26 ms per event**, which agrees with the 0.28-0.30 ms per
  event that the block-timed version produced at the same size. Two methods, one
  answer.
- **Conclusion for a real run.** Tracing a 10,000-event session costs roughly
  2.6 s of wall clock. Model latency is 10-90 s per request, so this is not a
  reason to leave the trace off -- but it is also not free, and the earlier
  version of this code was 20x worse per event.

The paired design is not decoration. Block-timing the same 804-event run
reported the trace as **-252 ms** -- faster with tracing on -- because the first
block was still being written back to disk while the second was timed. That is
the number this table exists to replace.

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
| 50 | 55 | 5/5 | 5/5 | 2.04 ms |
| 200 | 205 | 5/5 | 5/5 | 8.81 ms |
| 800 | 805 | 5/5 | 5/5 | 13.16 ms |

Reading this:

- Ranking is exact at every size tested: the planted fact is first, not merely
  present. The distractors share vocabulary with the queries (the filler includes
  words like `deploy`, `retry` and `config`), so this is not a trivial separator.
- Search cost grows linearly with the archive, because the scorer scans every
  entry. At 805 entries that is 13 ms, which is nothing next to a model call; at
  hundreds of thousands of entries it would need an inverted index. This is the
  main scaling limit of the current implementation.
- Five facts per size is a small sample. It demonstrates that the ranking works,
  not how well it generalises to real conversation text, where queries are
  messier and the useful answer may share no rare token with them.
- These numbers are higher than the first recorded run because the filler is now
  unique per line. It used to repeat every 15 lines, so the "800 distractor" row
  was really a 15-document archive scanned 805 times.

## 5. Vector retrieval

The same archives, searched through the pluggable backend with the offline
`hash` stand-in, at the embedder's batch size of 96. Cost first:

| Distractors | Archived | Texts embedded | Provider calls | Cold query (ms) | Repeat texts | Repeat calls | Warm query (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 55 | 56 | 2 | 38.97 | 0 | 0 | 13.09 |
| 200 | 205 | 206 | 4 | 99.43 | 0 | 0 | 21.67 |
| 800 | 805 | 806 | 10 | 326.22 | 0 | 0 | 78.43 |

Then quality and steady-state cost, with the archive already embedded:

| Distractors | Lexical (ms) | Vector (ms) | Hybrid (ms) | Top-1 lexical | Top-1 vector | Top-1 hybrid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 1.57 | 7.22 | 7.62 | 5/5 | 4/5 | 5/5 |
| 200 | 2.83 | 21.93 | 23.27 | 5/5 | 3/5 | 5/5 |
| 800 | 8.21 | 79.32 | 96.18 | 5/5 | 2/5 | 5/5 |

Reading this, including the parts that do not flatter the change:

- **The cache is what makes it affordable.** The first vector query embeds the
  whole archive -- 806 texts, 10 provider round trips at a batch of 96. Every
  later query embeds only its own text, and repeating a query embeds nothing at
  all. Vectors are unit-normalised on the way into the cache, so a comparison is
  a dot product rather than two square roots per candidate.
- **The offline stand-in ranks worse than lexical, and the table says so.** It
  hashes tokens into slots, so it sees the same word-overlap signal the lexical
  scorer already uses, with less precision: top-1 falls from 5/5 to 2/5 as the
  archive grows. That is a statement about the stand-in, not about embeddings.
  **No number here supports "vector retrieval is better"** -- that needs a real
  embeddings model, and the default provider for this harness serves no
  `/embeddings` route at all.
- **Hybrid is the safe combination.** Fusing the two rankings by reciprocal rank
  holds 5/5 at every size here, because the lexical head keeps its place, while
  the vector half can still surface an entry lexical scoring missed.
- **The vector scan is pure Python**, so it costs roughly 6-10x a lexical scan at
  the same size. A real deployment would put both behind a vector index rather
  than scanning a list.

## 7. Verification loop

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

## 8. Dispatch policy

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

## 9. Tool exposure

The built-in tools plus a synthetic bridged surface of 24 tools, serialised the
way the request carries them. Budget `0` means "expose everything".

| Bridged tools | Budget | Exposed | Hidden | Schema chars | Est. tokens | vs no budget |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 0 | 35 | 0 | 17,869 | 4,467 | 1.00x |
| 24 | 8 | 8 | 27 | 5,834 | 1,458 | 0.33x |
| 24 | 12 | 12 | 23 | 8,977 | 2,244 | 0.50x |

The task behind these rows is `fix the failing test in the parser`. The four
always-exposed tools are `find_tools`, `read_file`, `grep_file` and `glob_file`;
the rest of the budget goes to the best-ranked matches, which for that task are
`edit_file`, `run_sandbox`, `browser_open` and `run_bash`.

A budget of 8 costs a third of the schema payload. Characters are reported
rather than tokens so the arithmetic stays checkable; `chars / 4` is the usual
rough estimate and is labelled as such.

## 10. Hidden tool recovery

Hiding a tool is only safe if the model can get it back. The experiment hides
`vector_search` behind the budget above, then asks `find_tools` for it in its own
words (the tool's description, which is what a model has to work from).

| Wanted | Hidden before | Budget | Exposed before | Recovered |
| --- | --- | ---: | ---: | --- |
| `vector_search` | yes | 8 | 8 | yes |

Recovery is by construction rather than by ranking luck: `find_tools` adds the
name to a pinned set that survives the next selection, and a call to a hidden
tool is refused with the instruction to use `find_tools` instead of being
executed. Both paths are covered in `tests/test_selector.py`.

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
- **The trace cost is measured per event, with a wide spread.** The paired
  median at 804 events is +208 ms, but individual pairs ranged from -42 ms to
  +1004 ms, and the 84-event run is entirely inside that spread. Treat ~0.26 ms
  per event as an order of magnitude, not a constant.
- **Retrieval was measured on synthetic text.** Five planted facts per size,
  against filler that shares vocabulary with the queries. Real conversation
  queries are messier, and the useful message may share no rare token with them.
- **The exposure experiment measures a payload, not a request.** It counts the
  serialised schemas of the tools a budget selects. The token column is a
  `chars / 4` estimate, not a tokenizer count, and the bridged surface is
  synthetic filler rather than a real server. The recovery path is real: it runs
  the shipped `find_tools` and selection code.
- **Vector retrieval quality is not measured here.** The only embedder this suite
  can run offline hashes tokens, so it is a weaker lexical scorer, not a semantic
  one. The measured part is the plumbing: batching, the cache, normalisation, the
  cosine scan and the fallback. A real embeddings endpoint is needed to say
  anything about quality, and the harness's default provider does not offer one.

## Files

| Path | Contents |
| --- | --- |
| `bench/experiments.py` | the experiment runner; prints a report and writes the raw JSON |
| `EXPERIMENTS.json` | raw results of the recorded run |
| `tests/test_parallel.py` | overlap is proved with a barrier, not a stopwatch |
| `tests/test_policy.py` | policy decisions and dispatch attribution |
| `tests/test_embed.py` | the embedders, the fusion, the cache and the fallback |
| `tests/test_selector.py` | the exposure budget, `find_tools`, and the hidden-call refusal |
| `tests/test_trace.py` | trace format, flushing and failure handling |
