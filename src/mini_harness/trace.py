"""Append-only JSONL event log for a run.

Off by default: with no path configured every `emit` is a cheap no-op, so the
agent can call it unconditionally. Set ``Config.trace_path`` (or the
``MINI_HARNESS_TRACE`` environment variable) to switch it on.

One JSON object per line, each with ``seq``, ``ts`` and ``event``. The event
names are stable and are what ``bench/atif.py`` or any external analysis would
key on:

    run_start, turn, usage, tool_call, tool_result, retry, compact,
    subagent_start, subagent_end, run_end
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
        return

    def configure(self, path: str|Path|None) -> 'Trace':
        self.path = Path(path) if path else None
        self.enabled = self.path is not None
        self._seq = 0
        return self

    def emit(self, event: str, **fields) -> None:
        if not self.enabled or self.path is None:
            return
        with self._lock:
            self._seq += 1
            record = {'seq': self._seq, 'ts': time.time(), 'event': event, **fields}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, 'a', errors='replace', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            except OSError as e:
                # A trace must never take down the run it is observing.
                print(f'[trace]: disabled, cannot write {self.path}: {e}')
                self.enabled = False
        return

TRACE = Trace()
