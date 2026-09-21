"""What a subagent hands back: the answer, and a report the harness can read.

A subagent used to return a bare string, so the parent could not tell "the
explore agent answered" from "the explore agent ran out of turns", and nothing
recorded which files it touched or what it cost. The model still gets prose --
it is the main reader -- with a one-line JSON trailer for everything else.
"""

import json
from types import SimpleNamespace

import pytest

from mini_harness.budget import ACCOUNT, Budget
from mini_harness.tool import box
from mini_harness.tool.box import TOOLS, subagent_report
from mini_harness.trace import TRACE
from tests.conftest import call, write
from tests.test_agent import model_message


def report_of(text: str) -> dict:
    """The trailer, as the harness would read it."""
    marker = '[subagent report] '
    assert marker in text, text
    return json.loads(text.split(marker, 1)[1])


class SubClient:
    """A subagent's model: a script of responses, then a final answer."""

    def __init__(self, script, usage=None):
        self.script = list(script)
        self.usage = usage or SimpleNamespace(prompt_tokens=120, completion_tokens=30)
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        message = self.script.pop(0) if self.script else model_message('out of script')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self.usage)

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def explore(prompt='look around'):
    return box.RunSubAgentInput(task_description='explore', prompt=prompt,
                                agent_type='explore_agent')


def write_call(path='sandbox/new.py', call_id='c1'):
    return call('write_file', call_id=call_id, file_path=path, content='value = 1\n')


# ------------------------------------------------------------------ the success path


def test_a_finished_subagent_reports_that_it_finished(cfg, monkeypatch):
    monkeypatch.setattr(box, 'OpenAI', lambda **_: SubClient([model_message('found it')]))

    text = box.run_subagent(explore(), cfg=cfg)
    report = report_of(text)

    assert 'found it' in text
    assert report['agent'] == 'explore_agent'
    assert report['ok'] is True
    assert report['reason'] == ''
    assert report['turns'] == 1
    assert report['tools'] == 0
    assert report['prompt'] == 120 and report['completion'] == 30
    assert report['seconds'] >= 0


def test_the_prose_comes_before_the_trailer(cfg, monkeypatch):
    monkeypatch.setattr(box, 'OpenAI', lambda **_: SubClient([model_message('the answer')]))

    text = box.run_subagent(explore(), cfg=cfg)

    assert text.index('the answer') < text.index('[subagent report]')


def test_an_empty_answer_still_produces_a_readable_report(cfg, monkeypatch):
    monkeypatch.setattr(box, 'OpenAI', lambda **_: SubClient([model_message('')]))

    text = box.run_subagent(explore(), cfg=cfg)

    assert 'finished in 1 turn(s)' in text
    assert report_of(text)['ok'] is True


# ------------------------------------------------------------------ what it did


def test_a_coding_subagent_reports_the_files_it_edited(cfg, workspace, monkeypatch):
    from tests.conftest import executor
    client = SubClient([model_message('', [write_call('sandbox/new.py')]), model_message('done')])
    monkeypatch.setattr(box, 'OpenAI', lambda **_: client)

    report = report_of(box.run_subagent(
        box.RunSubAgentInput(task_description='write it', prompt='create the file',
                             agent_type='coding_agent'), cfg=cfg))

    assert report['edited'] == ['sandbox/new.py']
    assert report['tools'] == 1
    assert (workspace / 'sandbox' / 'new.py').exists()


def test_a_refused_write_is_not_reported_as_an_edit(cfg, monkeypatch):
    """The report must describe what happened, not what was attempted."""
    client = SubClient([model_message('', [call('write_file', call_id='c1',
                                                 file_path='../escape.py', content='x')]),
                        model_message('done')])
    monkeypatch.setattr(box, 'OpenAI', lambda **_: client)

    report = report_of(box.run_subagent(
        box.RunSubAgentInput(task_description='write it', prompt='escape',
                             agent_type='coding_agent'), cfg=cfg))

    assert report['edited'] == []
    assert report['tools'] == 1


def test_explore_cannot_write_even_if_it_tries(cfg, workspace, monkeypatch):
    """The role's tool list is the boundary, and the report says so."""
    client = SubClient([model_message('', [write_call('sandbox/nope.py')]), model_message('ok')])
    monkeypatch.setattr(box, 'OpenAI', lambda **_: client)

    report = report_of(box.run_subagent(explore(), cfg=cfg))

    assert report['edited'] == []
    assert not (workspace / 'sandbox' / 'nope.py').exists()


# ------------------------------------------------------------------ not finishing


def test_running_out_of_turns_is_reported_as_unfinished(cfg, monkeypatch):
    always_calls = [model_message('', [call('glob_file', call_id=f'c{i}', pattern='*.py')])
                    for i in range(cfg.max_turns_sub)]
    client = SubClient(always_calls)
    monkeypatch.setattr(box, 'OpenAI', lambda **_: client)

    text = box.run_subagent(explore(), cfg=cfg)
    report = report_of(text)

    assert report['ok'] is False
    assert report['reason'] == 'exhausted'
    assert 'did not finish' in text
    assert report['turns'] == cfg.max_turns_sub
    assert client.calls == cfg.max_turns_sub


def test_a_spent_budget_is_reported_and_nothing_is_sent(cfg, monkeypatch):
    client = SubClient([model_message('never sent')])
    monkeypatch.setattr(box, 'OpenAI', lambda **_: client)
    ACCOUNT.reset().attach(Budget(cost_budget=0.01, price_in=1.0, price_out=1.0))
    ACCOUNT.record(SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0))

    text = box.run_subagent(explore(), cfg=cfg)
    report = report_of(text)

    assert report['ok'] is False
    assert report['reason'] == 'budget-cost'
    assert 'the cost budget was spent' in text
    assert client.calls == 0


# ------------------------------------------------------------------ the trace agrees


def trace_events(cfg, session_dir, action):
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        action()
    finally:
        TRACE.configure(None)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_the_trace_carries_the_same_facts_as_the_report(cfg, session_dir, monkeypatch):
    monkeypatch.setattr(box, 'OpenAI', lambda **_: SubClient([model_message('done')]))

    events = trace_events(cfg, session_dir, lambda: box.run_subagent(explore(), cfg=cfg))
    end = [event for event in events if event['event'] == 'subagent_end'][0]

    assert end['ok'] is True
    assert end['reason'] == ''
    assert end['turns'] == 1
    assert end['prompt'] == 120 and end['completion'] == 30
    assert end['edited'] == []
    assert events[0]['event'] == 'subagent_start'


def test_the_trace_marks_an_unfinished_subagent(cfg, session_dir, monkeypatch):
    always_calls = [model_message('', [call('glob_file', call_id=f'c{i}', pattern='*.py')])
                    for i in range(cfg.max_turns_sub)]
    monkeypatch.setattr(box, 'OpenAI', lambda **_: SubClient(always_calls))

    events = trace_events(cfg, session_dir, lambda: box.run_subagent(explore(), cfg=cfg))
    end = [event for event in events if event['event'] == 'subagent_end'][0]

    assert end['ok'] is False
    assert end['reason'] == 'exhausted'


# ------------------------------------------------------------------ the renderer


def test_the_report_renders_a_write_count_when_it_edited_something():
    text = subagent_report(agent='coding_agent', ok=True, reason='', turns=3, tools=4,
                           edited=['b.py', 'a.py', 'a.py'], prompt=10, completion=2,
                           seconds=1.234, answer='finished the job')

    assert 'edited 2 file(s): a.py, b.py' in text
    assert report_of(text)['edited'] == ['a.py', 'b.py']


def test_the_report_keeps_the_reason_for_the_model_to_read():
    text = subagent_report(agent='explore_agent', ok=False, reason='exhausted', turns=20,
                           tools=20, edited=[], prompt=1, completion=1, seconds=2.0,
                           answer='Where I got to')

    assert 'did not finish: exhausted' in text
    assert 'Where I got to' in text
    assert report_of(text)['seconds'] == 2.0
