"""Append-only JSONL event log for a run.

Off by default: with no path configured every `emit` is a cheap no-op, so the
agent can call it unconditionally. Set ``Config.trace_path`` (or the
``MINI_HARNESS_TRACE`` environment variable) to switch it on.

One JSON object per line, each with ``seq``, ``ts`` and ``event``. The event
names are stable and are what ``bench/atif.py`` or any external analysis would
key on:

    run_start, turn, usage, tool_call, tool_result, batch_parallel, retry,
    compact, subagent_start, subagent_end, verify_nudge, budget_stop, run_end

The file handle is opened once per run and flushed per event, so a crash keeps
everything written so far. Reopening the file for every event costs about 2 ms
on Windows, which is more than the event itself is worth.
"""

import json
import threading
import time

from pathlib import Path

class Trace:
    def __init__(self) -> None:
        self.path: Path|None = None
        self.enabled = False
        self._seq = 0
        self._lock = threading.Lock()
        self._handle = None
        return

    def configure(self, path: str|Path|None) -> 'Trace':
        with self._lock:
            self._close()
            self.path = Path(path) if path else None
            self.enabled = self.path is not None
            self._seq = 0
            if self.enabled:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._handle = open(self.path, 'a', errors='replace', encoding='utf-8')
                except OSError as e:
                    print(f'[trace]: disabled, cannot open {self.path}: {e}')
                    self.enabled = False
        return self

    def emit(self, event: str, **fields) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._handle is None:
                return
            self._seq += 1
            record = {'seq': self._seq, 'ts': time.time(), 'event': event, **fields}
            try:
                self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                self._handle.flush()
            except OSError as e:
                # A trace must never take down the run it is observing.
                print(f'[trace]: disabled, cannot write {self.path}: {e}')
                self._close()
                self.enabled = False
        return

    def close(self) -> None:
        with self._lock:
            self._close()
            self.enabled = False
        return

    def _close(self) -> None:
        """Caller holds the lock."""
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        return

TRACE = Trace()
