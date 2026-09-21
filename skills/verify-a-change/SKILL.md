---
name: verify-a-change
description: Run what this repository actually checks before claiming a change works
tools: read_file, run_bash, grep_file
---

Do not report a change as working on the strength of reading the diff. Run the
things this repository runs, in this order, and report the output.

1. **The suite, exactly as CI runs it.** The console-script form is the one that
   matters: `python -m pytest` puts the working directory on `sys.path`, so a
   test that only passes under it is still broken in CI.

   ```sh
   uv run --locked pytest
   ```

   Expect `N passed` with no failures. If the count moved, say why.

2. **The linter.** `ruff` is the only style gate here:

   ```sh
   uv run --locked ruff check src tests bench tui.py
   ```

3. **The offline checks that are not tests.** The experiments run without a key
   and their output is recorded, so a change to a mechanism they cover has to
   leave them consistent:

   ```sh
   uv run --locked python -m bench.experiments --repeats 3
   ```

   `tests/test_experiments_doc.py` checks that `EXPERIMENTS.md` quotes the run
   `EXPERIMENTS.json` recorded. Re-recording moves those numbers, so run the test
   and update the document in the same change.

4. **Only then, the model-in-the-loop checks** if the change touches the loop,
   the tools or the prompt. These cost money and need an API key:

   ```sh
   uv run --locked python -m bench.mini_bench --tasks split_cents --only baseline --repeats 2
   ```

5. **Say what you did not verify.** A pass proves the code ran, not that the
   behaviour is right. Name the case that is still unchecked.
