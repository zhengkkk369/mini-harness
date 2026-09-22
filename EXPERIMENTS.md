# Harness experiments

Measurements of the mini-harness mechanisms added in this repository: parallel
tool execution, subagent concurrency, the event trace, retrievable memory,
compaction fidelity, the pluggable vector backend, the verification loop, the
dispatch policy, and the tool-exposure budget.

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
| Repeats | 15 per timed configuration (median reported, paired where two conditions are compared) |

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

| Injected latency | Calls | Serial (median) | Concurrent (median) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0 ms | 6 | 41.80 ms | 29.82 ms | 1.40x |
| 5 ms | 6 | 75.50 ms | 28.05 ms | 2.69x |
| 20 ms | 6 | 166.73 ms | 44.65 ms | 3.73x |

Reading this:

- The gain grows with per-call latency and approaches the worker count. That is
  the expected shape: with six calls in flight and no shared bottleneck, the
  floor is one call's latency rather than six.
- The speedup never reaches the ideal 6x. At 20 ms the concurrent batch takes
  46.5 ms, not the ~20 ms a perfect pool would give. Some of the per-call work
  (argument validation, the file-state record, output handling) still runs under
  the GIL, and the pool has its own start-up cost.
- Even with **no injected latency** the concurrent path is 1.67x faster, so real
  file reads and MD5 digests do overlap. I did not profile further, so I cannot
  attribute that split between filesystem concurrency and `hashlib` releasing the
  GIL.
- The reading moves between runs with machine load: recorded runs of this same
  experiment put the three rows anywhere between 1.55 and 1.90x, 1.28 and 2.98x,
  and 2.68 and 4.04x. The shape is stable; the third digit is not.
- A batch is only ever overlapped when every call is safe to overlap:
  side-effect free (`read_file`, `grep_file`, `glob_file`), or a batch consisting
  entirely of subagents. One write, one shell call, one unknown name, or a mix of
  subagents with anything else makes the whole batch serial, which is asserted by
  `tests/test_parallel.py`.

## 2. Subagent concurrency

A batch of four `run_subagent` calls through the same path. A subagent spends
its time blocked on its own model calls, which is what the injected latency
models.

| Injected latency | Calls | Serial (median) | Concurrent (median) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 20 ms | 4 | 82.47 ms | 24.75 ms | 3.33x |
| 100 ms | 4 | 402.83 ms | 105.56 ms | 3.82x |

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
| off | 20 | 0 | 130.58 ms | 120.75 ms | 166.82 ms | 0 |
| on | 20 | 84 | 162.39 ms | 134.26 ms | 196.20 ms | 11,588 |
| off | 200 | 0 | 1599.89 ms | 1488.83 ms | 1850.67 ms | 0 |
| on | 200 | 804 | 1704.48 ms | 1487.83 ms | 1982.77 ms | 108,791 |

Those medians are not the measurement. The two conditions are timed
**alternately inside each repeat** and the paired difference is what gets a
median, because timing one block and then the other measures the machine:

| Turns | Events | Overhead (paired median) | Min pair | Max pair | Per event |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 84 | **+25.94 ms** | -25.62 ms | +68.70 ms | 308.9 us |
| 200 | 804 | **+74.62 ms** | -298.47 ms | +452.75 ms | 92.8 us |

Reading it (these two sentences are generated from the artifact, like the tables):

The 84-event row is not a measurement on this machine: **+25.94 ms** over pairs spanning -25.62 to +68.70 ms.

Tracing the 804-event run costs **+74.62 ms** over pairs spanning -298.47 to +452.75 ms, about **0.09 ms per event**.

- **The per-event figure moves between runs, and the doc says so.** Earlier
  recordings of this same experiment put the 804-event row at +58.63 ms (0.07),
  +82.25 ms (0.10), +174.02 ms (0.22) and +208.47 ms (0.26); individual pairs
  have reached +1004 ms. Timing a process that is writing to disk is noisy, and
  one slow pair can carry the median. The honest form is **"of order 0.1 ms per
  event"**, not a precise constant -- the sample has moved it on every
  re-recording. The numbers in this list are history: they are deliberately kept
  as recorded, and the current recording is the generated line above.
- **Conclusion for a real run.** Tracing a 10,000-event session costs on the
  order of a second of wall clock. Model latency is 10-90 s per request, so this
  is not a reason to leave the trace off -- but it is also not free, and the
  earlier version of this code was 20x worse per event.

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
| 50 | 55 | 5/5 | 5/5 | 1.28 ms |
| 200 | 205 | 5/5 | 5/5 | 2.20 ms |
| 800 | 805 | 5/5 | 5/5 | 6.92 ms |

Reading this:

- Ranking is exact at every size tested: the planted fact is first, not merely
  present. The distractors share vocabulary with the queries (the filler includes
  words like `deploy`, `retry` and `config`), so this is not a trivial separator.
- Search cost grows linearly with the archive, because the scorer scans every
  entry. At 805 entries that is 6.5 ms, which is nothing next to a model call; at
  hundreds of thousands of entries it would need an inverted index. This is the
  main scaling limit of the current implementation. The absolute figure moves
  with machine load (recorded runs span 6.5-13.2 ms for the same row); the
  linear growth is what is stable.
- Five facts per size is a small sample. It demonstrates that the ranking works,
  not how well it generalises to real conversation text, where queries are
  messier and the useful answer may share no rare token with them.
- These numbers are higher than the first recorded run because the filler is now
  unique per line. It used to repeat every 15 lines, so the "800 distractor" row
  was really a 15-document archive scanned 805 times.

## 5. Compaction fidelity

A conversation with six planted facts is compacted once. Two stub summarisers
bracket the machinery: `verbatim` returns the removed text, which no real model
does, and `losing` returns a sentence that mentions nothing. A third row runs the
real model in the loop, which is the row that answers the question the other two
cannot. `Facts verbatim` is an exact-substring test; `Words kept` is the share of
each fact's own words still present and `Facts over 60%` counts the facts with at
least that much of their wording left, so a summary that paraphrases is not
scored as a loss.

| Turns | Summariser | Messages before | After | Removed | Facts verbatim | Words kept | Facts over 60% | Archived | Recall top-1 | Recall top-3 | Compact (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | verbatim | 55 | 10 | 45 | 6/6 | 1.00 | 6/6 | 5 | 5/5 | 5/5 | 7.62 |
| 6 | losing | 55 | 10 | 45 | 1/6 | 0.17 | 1/6 | 5 | 5/5 | 5/5 | 13.92 |
| 6 | model | 55 | 10 | 45 | 1/6 | 1.00 | 6/6 | 5 | 5/5 | 5/5 | 11767.57 |
| 12 | verbatim | 103 | 10 | 93 | 6/6 | 1.00 | 6/6 | 6 | 6/6 | 6/6 | 6.19 |
| 12 | losing | 103 | 10 | 93 | 0/6 | 0.00 | 0/6 | 6 | 6/6 | 6/6 | 5.58 |
| 12 | model | 103 | 10 | 93 | 0/6 | 0.88 | 6/6 | 6 | 6/6 | 6/6 | 7571.83 |
| 30 | verbatim | 247 | 10 | 237 | 6/6 | 1.00 | 6/6 | 6 | 6/6 | 6/6 | 7.35 |
| 30 | losing | 247 | 10 | 237 | 0/6 | 0.00 | 0/6 | 6 | 6/6 | 6/6 | 6.63 |
| 30 | model | 247 | 10 | 237 | 0/6 | 0.92 | 6/6 | 6 | 6/6 | 6/6 | 9210.72 |

Reading it:

- **The machinery loses nothing.** Every fact that leaves the context is in the
  archive, and recall puts it first: 5/5 and 6/6 at every size. "Retrievable
  memory" means reachable, not forgotten.
- **A real summariser keeps the substance and not the sentences.** The `model`
  rows reproduce almost none of the planted facts *word for word* -- 1/6 at six
  turns, 0/6 at twelve and thirty -- which is what a summary is for. The
  paraphrase-tolerant reading is the one that matters: 88-100% of each fact's own
  words are still present and **all six facts survive in substance at every
  size**. A model does not have to quote to keep something.
- **So the fault line is the archive, not the summary.** With a summary that
  carries nothing, 0/6 facts are left in context and the archive is what makes
  them reachable (6/6); with a real summary, the context already carries them.
  The stub rows are the bounds, and the live row shows the mechanism does not sit
  near the bottom of them.
- **One pass can remove almost everything.** At 30 turns, 237 of 247 messages
  leave the context and 10 remain: after that, the summary is the only carrier of
  the rest, which is why `recall` exists as the second path.
- **The machinery is cheap and the summary is not.** A stub pass costs 5-14 ms;
  the same pass through the model costs **7.6-11.8 s**, three orders of magnitude
  more. The real cost of compaction is the summariser call, and that is the number
  to budget with -- the local work is free by comparison.
- **What is recorded, and what is not.** The live row carries the first 400
  characters of the summary it produced, so the claim can be read rather than
  believed. It does *not* measure whether the summary is true: a summary can keep
  every noun and invert the meaning, and nothing here would notice.

### Re-compaction

A fact the first pass removed, and whether it is still reachable after each later
one. A long run compacts repeatedly: the conversation shrinks while the archive
grows.

| Turns | Passes | Early facts | Reachable after each pass | Archive entries |
| ---: | ---: | ---: | --- | --- |
| 12 | 3 | 6 | 6/6 -> 6/6 -> 6/6 | 94, 113, 132 |

The earliest facts stay reachable through every later pass, so "retrievable
memory" does not quietly mean "the most recent loss" as the archive grows.

## 6. Vector retrieval

The same archives, searched through the pluggable backend with the offline
`hash` stand-in, at the embedder's batch size of 96. Cost first:

| Distractors | Archived | Texts embedded | Provider calls | Cold query (ms) | Repeat texts | Repeat calls | Warm query (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 55 | 56 | 2 | 21.99 | 0 | 0 | 4.27 |
| 200 | 205 | 206 | 4 | 152.29 | 0 | 0 | 16.08 |
| 800 | 805 | 806 | 10 | 232.82 | 0 | 0 | 65.96 |

Then quality and steady-state cost, with the archive already embedded:

| Distractors | Lexical (ms) | Vector (ms) | Hybrid (ms) | Top-1 lexical | Top-1 vector | Top-1 hybrid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 1.90 | 6.32 | 5.26 | 5/5 | 4/5 | 5/5 |
| 200 | 2.53 | 16.34 | 21.40 | 5/5 | 3/5 | 5/5 |
| 800 | 8.36 | 65.64 | 88.93 | 5/5 | 2/5 | 5/5 |

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
- **The vector scan is pure Python**, so it costs a few times a lexical scan at
  the same size and the gap widens with the archive (3.6x at 55 entries, 10x at
  805). A real deployment would put both behind a vector index rather than
  scanning a list.

## 7. Verification loop

A scripted agent that reads a file, edits it, and then stops. `verify_required`
is on by default; the nudge is bounded by `verify_nudges` (1 here). The criterion
is "a run touched what changed, or ran the suite": the verdict column says which
of those it was, and `none` means the run never ran anything at all.

| scenario | turns | mutations | verified | verdict | nudges | outcome |
| --- | ---: | ---: | --- | --- | ---: | --- |
| off, edit then answer | 3 | 1 | False | none | 0 | completed |
| on, edit then answer | 4 | 1 | False | none | 1 | completed |
| on, edit then run the changed file | 4 | 1 | True | targeted | 0 | completed |
| on, edit then run something unrelated | 5 | 1 | False | unrelated | 1 | completed |

Reading this:

- With verification off, the run stops unverified and **says so**: `verified` is
  False. The flag reports the truth whether or not the mechanism is enforcing it.
- With verification on, stopping unverified costs exactly one extra turn (3 to
  4) and one nudge.
- **Running the file that changed is accepted and costs nothing** (4 turns, no
  nudge, `verified` True). So does running the suite: `pytest`, `unittest`, `tox`,
  `cargo test` and a short list of others count on their own, because the most
  thorough check there is need not name the file.
- **Running something else is not accepted.** `echo verified` leaves the file
  unverified, so the run is nudged once and reports False — one turn more than
  the run that checked, which is the cost of the stricter rule. The nudge names
  the file it is asking about.
- The nudge is bounded. In the second and fourth scenarios the scripted agent
  ignores it, and the run still terminates after one nudge rather than looping.

What this does **not** measure is whether running something proves the change
works. "Touched the file" is a heuristic: `cat f.py` also satisfies it, and a
test that passes for an unrelated reason satisfies it too. It is strictly
stronger than the rule it replaced — which any command satisfied, including
`echo` — but it is still a mechanical check, which is why the result is reported
as a verdict (`none` / `unrelated` / `targeted` / `suite`) alongside `verified`
rather than being treated as proof.

## 8. Dispatch policy

Policy decisions for a representative set of calls, with
`policy_deny_tools=('run_sandbox',)` and `policy_deny_patterns=('rm -rf*',)`.

| Call | Allowed | Tag |
| --- | --- | --- |
| `run_bash` with `rm -rf /` | no | `policy_denied` |
| `run_bash` with `ls -la` | yes | success |
| `run_sandbox` with `ls` | no | `policy_denied` |
| `read_file` with `tree/f000.py` | yes | success |

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
| 24 | 0 | 36 | 0 | 18,557 | 4,639 | 1.00x |
| 24 | 8 | 8 | 28 | 6,149 | 1,537 | 0.33x |
| 24 | 12 | 12 | 24 | 8,803 | 2,201 | 0.47x |

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
- **The trace cost is measured per event, with a wide spread.** The generated
  line in section 3 carries the paired median and its spread; individual pairs
  there straddle zero, and the shorter run sits entirely inside them. Recorded
  runs put the per-event cost between 0.03 and 0.31 ms. Treat it as
  "of order 0.1 ms per event", not a constant.
- **The compaction experiment measures the machinery and one real summariser.**
  The `verbatim` and `losing` rows are stubs at the extremes, and they bracket the
  machinery rather than describe a model. The `model` row is a real summariser
  through the real call path, so the default prompt does have a reading now --
  with two limits: it is one model at one temperature on one synthetic
  conversation, and neither the exact-substring column nor the word-coverage
  column checks whether the summary is *true*. A summary that keeps every noun and
  inverts the meaning passes both.
- **Retrieval was measured on synthetic text.** Five planted facts per size,
  against filler that shares vocabulary with the queries. Real conversation
  queries are messier, and the useful message may share no rare token with them.
  The provider-backed measurement in [MODEL_EVAL.md](MODEL_EVAL.md) section 5 uses
  forty paraphrase queries, so one query is 2.5 points there -- enough to tell a
  ten-point gap from noise, not enough for a two-point one.
- **The exposure experiment measures a payload, not a request.** It counts the
  serialised schemas of the tools a budget selects. The token column is a
  `chars / 4` estimate, not a tokenizer count, and the bridged surface is
  synthetic filler rather than a real server. The recovery path is real: it runs
  the shipped `find_tools` and selection code.
- **Vector retrieval quality is measured elsewhere, against a provider.** The only
  embedder this suite can run offline hashes tokens, so it is a weaker lexical
  scorer, not a semantic one. The measured part here is the plumbing: batching,
  the cache, normalisation, the cosine scan and the fallback. A real embeddings
  endpoint is needed to say anything about quality, and the harness's default
  provider does not offer one. `bench/embed_quality.py` is that measurement, and
  it has now been run against a real model: the result is in
  [MODEL_EVAL.md](MODEL_EVAL.md) section 5, with the raw numbers in
  [EMBED_QUALITY.json](EMBED_QUALITY.json). It uses a provider when one is
  configured (the environment or an `embedding:` block in a local, gitignored
  `config.yaml`) and the stand-in otherwise, labelling which one it used in its
  output. Its corpus is paraphrase-shaped -- a query and its fact share no content
  word -- because that is the case lexical scoring cannot serve.

## Files

| Path | Contents |
| --- | --- |
| `bench/experiments.py` | the experiment runner; prints a report and writes the raw JSON |
| `EXPERIMENTS.json` | raw results of the recorded run |
| `tests/test_experiments_doc.py` | every number above has to be the number the JSON recorded |
| `tests/test_parallel.py` | overlap is proved with a barrier, not a stopwatch |
| `tests/test_policy.py` | policy decisions and dispatch attribution |
| `tests/test_compact.py` | cut-point selection, the summary prompt, and the ordering that makes the archive safe to replay |
| `tests/test_embed.py` | the embedders, the fusion, the cache and the fallback |
| `tests/test_selector.py` | the exposure budget, `find_tools`, and the hidden-call refusal |
| `tests/test_trace.py` | trace format, flushing and failure handling |
