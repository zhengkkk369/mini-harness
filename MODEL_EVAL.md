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

## 2. Mini benchmark: 8 tasks, 3 configurations

`bench/mini_bench.py` — every task is stdlib-only and every check is an
assertion run by the harness process, so a pass never depends on the model
saying it passed. Three configurations run the same eight tasks:

| config | what changes |
| --- | --- |
| `baseline` | repository defaults |
| `no_verify` | `verify_required=False` |
| `serial_tools` | `parallel_tools=False` |

### Result per task

| config | task | pass | turns | calls | tokens | verified | s |
| --- | --- | :-: | ---: | ---: | ---: | :-: | ---: |
| baseline | fix_syntax | yes | 5 | 4 | 18,507 | yes | 8.6 |
| baseline | implement_median | yes | 8 | 9 | 37,727 | yes | 12.1 |
| baseline | preserve_order | yes | 9 | 9 | 41,801 | yes | 11.5 |
| baseline | two_file_constant | yes | 5 | 7 | 19,860 | yes | 6.2 |
| baseline | add_validation | yes | 11 | 11 | 76,264 | yes | 28.2 |
| baseline | fix_import | yes | 5 | 5 | 19,132 | yes | 6.1 |
| baseline | read_and_report | yes | 2 | 1 | 6,859 | yes | 2.2 |
| baseline | fix_two_bugs | yes | 7 | 7 | 30,672 | yes | 13.0 |
| no_verify | fix_syntax | yes | 5 | 4 | 18,447 | yes | 5.9 |
| no_verify | implement_median | yes | 10 | 10 | 48,237 | yes | 18.2 |
| no_verify | preserve_order | yes | 7 | 7 | 29,346 | yes | 11.4 |
| no_verify | two_file_constant | yes | 5 | 8 | 19,683 | yes | 8.6 |
| no_verify | add_validation | yes | 10 | 10 | 66,328 | yes | 23.5 |
| no_verify | fix_import | yes | 6 | 6 | 24,500 | yes | 9.5 |
| no_verify | read_and_report | yes | 2 | 1 | 6,893 | yes | 1.2 |
| no_verify | fix_two_bugs | yes | 7 | 8 | 29,899 | yes | 10.0 |
| serial_tools | fix_syntax | yes | 6 | 5 | 23,132 | yes | 7.6 |
| serial_tools | implement_median | yes | 6 | 6 | 25,763 | yes | 11.1 |
| serial_tools | preserve_order | yes | 10 | 10 | 51,583 | yes | 20.5 |
| serial_tools | two_file_constant | yes | 4 | 6 | 15,458 | yes | 4.2 |
| serial_tools | add_validation | yes | 7 | 7 | 34,944 | yes | 13.3 |
| serial_tools | fix_import | yes | 6 | 7 | 27,193 | yes | 9.5 |
| serial_tools | read_and_report | yes | 2 | 1 | 6,870 | yes | 1.5 |
| serial_tools | fix_two_bugs | yes | 5 | 5 | 20,129 | yes | 6.7 |

### Aggregate

| config | passed | rate | median turns | total turns | total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8/8 | 100% | 7 | 52 | 250,822 |
| no_verify | 8/8 | 100% | 7 | 52 | 243,333 |
| serial_tools | 8/8 | 100% | 6 | 46 | 205,072 |

**The honest headline is that this suite is too easy.** 24/24 passed, so it
cannot separate the configurations, and the differences in turns and tokens are
within run-to-run noise at n=8 with a single repetition each — in particular the
`serial_tools` column looking cheapest is not something I would claim as a real
effect. See Limitations.

### What the traces showed

Every run wrote a trace, and the trace answers the question the scripted
experiments could not: does the verification loop actually fire?

| config | runs | turns | tool calls | **verify nudges** | parallel batches | retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 8 | 52 | 53 | **0** | 6 | 0 |
| no_verify | 8 | 52 | 54 | **0** | 8 | 0 |
| serial_tools | 8 | 46 | 47 | **0** | 0 | 0 |

Two findings worth keeping:

- **The nudge never fired, and that is the correct outcome.** In all 24 runs the
  model ran the code after editing it, so `verified` was already true when it
  stopped. For small, well-specified tasks with a capable model, the prompt's
  instruction is followed and the mechanism costs nothing. It also means these
  tasks cannot measure the mechanism's *benefit* — that needs a model or a task
  that stops without verifying.
- **Parallel execution engaged on its own.** The model emitted several
  side-effect-free calls in one turn in 6 of 8 baseline runs (8 of 8 with
  verification off), so the concurrent path is exercised in normal use rather
  than being a feature that only tests reach. `serial_tools` shows 0 batches, as
  it must.

No retries fired in any run, so the backoff path was not exercised.

## 3. Harbor and SWE-bench Verified: attempted, not completed

The repository's documented benchmark path is Harbor against SWE-bench Verified.
What actually happened:

| step | outcome |
| --- | --- |
| Install Harbor 0.20.0 | worked (`uv run --with harbor==0.20.0`) |
| CLI flags match the README | verified against `harbor run --help` |
| Registry query | initially failed with the same `InvalidURL`; worked after the `sitecustomize` fix |
| Dataset resolution | worked — selected `astropy__astropy-7606` |
| Trial created | worked |
| Environment build / agent run | **still in progress when this was written**; no container image present, `trial.log` empty, ~3.8 GB consumed |

So the job got past dependency install, the registry, and dataset resolution, and
stopped in the Docker environment phase. No SWE-bench score was produced, and
nothing here should be read as one. The recorded job directory is under `jobs/`,
which the repository ignores.

## Limitations

- **Ceiling effect.** The mini suite is 8 easy tasks and the model solved all of
  them under every configuration. It cannot rank configurations. Its value is
  that it exercises the harness end to end against a real model and produces
  token, turn and trace data.
- **n=1 per cell.** Each task ran once per configuration. The turn and token
  differences above should not be read as effects; only the trace counts
  (nudges, parallel batches) are structural rather than statistical.
- **Cost is not reported.** The runs record tokens but no prices were
  configured, so every cost figure is $0.0000 and is omitted rather than
  invented. Pass `--price-in` and `--price-out` to get real numbers.
- **The TUI run is one prompt**, driven through a test pilot rather than a real
  keyboard. Approval modals, session switching and cancellation were not driven.
- **Task choice is mine.** These are the tasks I wrote; they are not a
  standardised benchmark and are not comparable to SWE-bench or Terminal-Bench.
- **One model, one machine, one day.** No repetition across time, and no second
  model to compare against.

## Reproducing

```sh
uv run python -m bench.mini_bench                     # all three configs, writes MINI_BENCH.json
uv run python -m bench.mini_bench --only baseline --tasks fix_syntax
uv run python -m bench.mini_bench --price-in 0.28 --price-out 0.42
```

The TUI run used a small pilot driver that is not part of the repository; the
raw results of the recorded runs are in [MINI_BENCH.json](MINI_BENCH.json).
