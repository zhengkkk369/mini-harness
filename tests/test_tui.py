"""The TUI's worker protocol, driven the way the App drives it.

The worker is a separate process speaking JSON lines on stdin and stdout, so
this is the one place the interface can be tested for real without a terminal.
Demo mode is used throughout: it makes no model calls.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mini_harness.history import PLACEHOLDER

pytest.importorskip('textual', reason='the TUI needs textual to run')

ROOT = Path(__file__).resolve().parents[1]
TUI = ROOT / 'tui.py'


class Worker:
    """Start tui.py --worker, feed it one request, answer its prompts."""

    def __init__(self, session_path, prompt, messages=None, demo=True, timeout=120.0):
        self.timeout = timeout
        self.events = []
        self._queue = queue.Queue()
        self.proc = subprocess.Popen(
            [sys.executable, str(TUI), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', cwd=str(ROOT))
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.send({'prompt': prompt, 'messages': messages or [], 'tokens': 0,
                   'demo': demo, 'session_path': str(session_path)})

    def _pump(self):
        for line in self.proc.stdout:
            self._queue.put(line)
        self._queue.put(None)

    def send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + '\n')
        self.proc.stdin.flush()

    def run(self, approve=True):
        """Collect events, answering any approval request."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                line = self._queue.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                break
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            self.events.append(event)
            if event.get('kind') == 'confirm':
                self.send({'id': event['id'], 'allow': approve})
            if event.get('kind') in {'done', 'error'}:
                break
        return self.events

    def kinds(self):
        return [event.get('kind') for event in self.events]

    def close(self):
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


@pytest.fixture
def worker(session_dir):
    started = []

    def start(prompt, **kwargs):
        instance = Worker(session_dir / 'session.json', prompt, **kwargs)
        started.append(instance)
        return instance

    yield start
    for instance in started:
        instance.close()


# --------------------------------------------------------------------------- protocol


def test_a_turn_streams_text_and_ends_with_done(worker):
    instance = worker('what can mini-harness do?')

    instance.run()

    kinds = instance.kinds()
    assert kinds[-1] == 'done'
    assert 'text' in kinds
    assert 'history' in kinds
    assert 'confirm' not in kinds          # nothing risky on this path


def test_the_streamed_text_carries_the_answer(worker):
    instance = worker('what can mini-harness do?')

    instance.run()

    spoken = ''.join(event.get('text', '') for event in instance.events
                     if event.get('kind') == 'text')
    assert 'Ten tools' in spoken


def test_a_session_is_reported_so_the_app_can_save_it(worker):
    instance = worker('what can mini-harness do?')

    instance.run()

    history = next(event for event in instance.events if event.get('kind') == 'history')
    roles = [message['role'] for message in history['messages']]
    assert roles[0] == 'system'
    assert roles[-1] == 'user'
    assert history['messages'][-1]['content'] == 'what can mini-harness do?'


# --------------------------------------------------------------------------- approval


def test_a_tool_call_asks_the_app_for_approval(worker):
    instance = worker('list the agent files')

    instance.run(approve=True)

    kinds = instance.kinds()
    assert 'confirm' in kinds
    assert 'tool_start' in kinds and 'tool_end' in kinds
    assert kinds[-1] == 'done'
    confirm = next(event for event in instance.events if event['kind'] == 'confirm')
    assert confirm['name'] == 'run_bash'
    assert '"command"' in confirm['arguments']


def test_an_approved_tool_reports_its_result(worker):
    instance = worker('list the agent files')

    instance.run(approve=True)

    end = next(event for event in instance.events if event['kind'] == 'tool_end')
    assert end['ok'] is True
    assert 'agent.py' in end['content']
    assert end['tag'] == ''


def test_a_denied_tool_is_reported_as_denied(worker):
    instance = worker('list the agent files')

    instance.run(approve=False)

    end = next(event for event in instance.events if event['kind'] == 'tool_end')
    assert end['ok'] is False
    assert end['tag'] == 'denied'
    assert instance.kinds()[-1] == 'done'


def test_the_approval_reply_must_match_the_pending_call(worker):
    """A reply for a different id counts as a refusal, not an approval."""
    instance = worker('list the agent files')
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            line = instance._queue.get(timeout=30)
        except queue.Empty:
            break
        if line is None:
            break
        event = json.loads(line.strip() or '{}')
        instance.events.append(event)
        if event.get('kind') == 'confirm':
            instance.send({'id': 'not-this-id', 'allow': True})
        if event.get('kind') in {'done', 'error'}:
            break

    end = next(event for event in instance.events if event['kind'] == 'tool_end')
    assert end['ok'] is False


# --------------------------------------------------------------------------- repair


def test_a_history_with_an_unanswered_call_is_repaired_before_use(worker):
    """The worker fixes the session it was handed, not just the one it writes."""
    broken = [
        {'role': 'user', 'content': 'earlier'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'lost-call', 'type': 'function',
             'function': {'name': 'read_file', 'arguments': '{"file_path": "a.py"}'}}]},
    ]
    instance = worker('what can mini-harness do?', messages=broken)

    instance.run()

    history = next(event for event in instance.events if event['kind'] == 'history')
    placeholders = [message for message in history['messages']
                    if message.get('role') == 'tool' and message.get('content') == PLACEHOLDER]
    assert [message['tool_call_id'] for message in placeholders] == ['lost-call']


def test_a_healthy_history_passes_through_untouched(worker):
    healthy = [
        {'role': 'system', 'content': 'the system prompt'},
        {'role': 'user', 'content': 'earlier'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'type': 'function',
             'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'the result'},
    ]
    instance = worker('what can mini-harness do?', messages=healthy)

    instance.run()

    history = next(event for event in instance.events if event['kind'] == 'history')
    assert history['messages'][:4] == healthy


def test_a_resumed_history_owns_the_system_prompt(worker):
    """The worker does not re-seed it, so a session must carry its own.

    That is what the App always writes: the first message of a stored session is
    the system prompt produced by the run that created it.
    """
    instance = worker('what can mini-harness do?',
                      messages=[{'role': 'system', 'content': 'from the stored run'},
                                {'role': 'user', 'content': 'earlier'}])

    instance.run()

    history = next(event for event in instance.events if event['kind'] == 'history')
    assert history['messages'][0]['content'] == 'from the stored run'


# --------------------------------------------------------------------------- failures


def test_a_broken_request_is_reported_as_an_error(session_dir):
    instance = Worker.__new__(Worker)
    instance.timeout = 60
    instance.events = []
    instance._queue = queue.Queue()
    instance.proc = subprocess.Popen(
        [sys.executable, str(TUI), '--worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace', cwd=str(ROOT))
    instance._reader = threading.Thread(target=instance._pump, daemon=True)
    instance._reader.start()
    instance.proc.stdin.write('this is not json\n')
    instance.proc.stdin.flush()
    try:
        instance.run()
    finally:
        instance.close()

    assert instance.kinds()[-1] == 'error'
    assert 'JSONDecodeError' in instance.events[-1]['text']


def test_the_worker_does_not_touch_the_environment_of_the_app(worker):
    """It is a separate process; the parent keeps its own config."""
    before = os.environ.get('MINI_HARNESS_PROFILE')
    instance = worker('what can mini-harness do?')

    instance.run()

    assert os.environ.get('MINI_HARNESS_PROFILE') == before
