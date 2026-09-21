"""The trajectory builder: rebuilding a conversation from the files it left.

`bench/atif.py` is the replay layer -- it turns a session, the compaction journal,
the telemetry file and the console log into one ATIF trajectory. It had no
offline tests, because it imports Harbor's trajectory models at module level.
Those are stubbed here when Harbor is absent, so the logic that decides what the
recorded run actually looked like is covered without the dependency.
"""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _plain(value):
    if isinstance(value, Record):
        return value.to_json_dict()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class Record:
    """Stands in for a Harbor model: keeps whatever it was constructed with.

    Harbor declares its optional fields with defaults, and `_fold` relies on
    that (`prev.observation is None`). An attribute that was never set therefore
    reads as None here too, and `to_json_dict` recurses the way pydantic's would.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, name):
        return None

    def to_json_dict(self):
        return _plain(self.__dict__)


def _install_harbor_stubs():
    """atif imports seven classes from harbor.models.trajectories.*."""
    for name in ('harbor', 'harbor.models', 'harbor.models.trajectories'):
        sys.modules.setdefault(name, ModuleType(name))
    classes = {
        'agent': 'Agent', 'final_metrics': 'FinalMetrics', 'observation': 'Observation',
        'observation_result': 'ObservationResult', 'step': 'Step', 'tool_call': 'ToolCall',
        'trajectory': 'Trajectory',
    }
    for module_name, class_name in classes.items():
        full = f'harbor.models.trajectories.{module_name}'
        module = ModuleType(full)
        setattr(module, class_name, type(class_name, (Record,), {}))
        sys.modules.setdefault(full, module)


def load_atif():
    try:
        return importlib.import_module('bench.atif')
    except ImportError:
        _install_harbor_stubs()
        return importlib.import_module('bench.atif')


atif = load_atif()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def journal_lines(*events) -> str:
    return ''.join(json.dumps(event) + '\n' for event in events)


# ------------------------------------------------------------------ loading the journal


def test_an_absent_journal_is_empty(tmp_path):
    events, notes = atif._load_history(tmp_path)

    assert events == []
    assert notes == []


def test_malformed_journal_lines_are_dropped_and_reported(tmp_path):
    write(tmp_path / atif.HISTORY,
          'not json\n'
          '\n'
          '{"ts": 1, "removed": "not a list"}\n'
          '{"ts": 2, "removed": [{"role": "user", "content": "kept"}]}\n')

    events, notes = atif._load_history(tmp_path)

    assert len(events) == 1
    assert events[0]['removed'][0]['content'] == 'kept'
    assert notes == ['dropped 2 malformed history line(s)']


# ------------------------------------------------------------------ splicing


SESSION = [{'role': 'system', 'content': 's'},
           {'role': 'user', 'content': 'live question'},
           {'role': 'assistant', 'content': 'live answer'}]


def test_a_committed_removal_is_spliced_in_front_of_the_session():
    removed = [{'role': 'user', 'content': 'old question'},
               {'role': 'assistant', 'content': 'old answer'}]
    events = [{'ts': 10.0, 'removed': removed, 'committed': True}]

    stream, summary_ts, notes = atif._splice(events, SESSION)

    assert [m['content'] for m in stream] == ['old question', 'old answer', 'live question',
                                              'live answer']
    assert summary_ts == {2: 10.0}
    assert notes == []


def test_an_uncommitted_removal_is_dropped():
    """committed: false is the interrupted compaction the writer can leave."""
    removed = [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}]
    events = [{'ts': 10.0, 'removed': removed, 'committed': False}]

    stream, _summary_ts, notes = atif._splice(events, SESSION)

    assert [m['content'] for m in stream] == ['live question', 'live answer']
    assert notes == ['dropped uncommitted trailing history event']


def test_a_legacy_entry_still_at_the_front_of_the_session_is_dropped():
    """Files written before the flag existed are judged by content.

    An interrupted compaction leaves the removal inside the session, where the
    entry would otherwise splice it in a second time. Dropping the entry keeps
    the conversation whole rather than doubled.
    """
    removed = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}]
    session = [{'role': 'system', 'content': 's'}, *removed, {'role': 'user', 'content': 'live'}]
    events = [{'ts': 1.0, 'removed': removed}]

    stream, _summary_ts, notes = atif._splice(events, session)

    assert [m['content'] for m in stream] == ['old', 'answer', 'live']
    assert notes == ['dropped uncommitted trailing history event']


def test_a_legacy_entry_that_is_not_at_the_front_is_kept():
    removed = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}]
    events = [{'ts': 1.0, 'removed': removed}]

    stream, _summary_ts, notes = atif._splice(events, SESSION)

    assert [m['content'] for m in stream] == ['old', 'answer', 'live question', 'live answer']
    assert notes == []


def test_several_compactions_are_spliced_in_order_with_their_timestamps():
    first = [{'role': 'user', 'content': 'first'}]
    second = [{'role': 'user', 'content': 'second'}]
    events = [{'ts': 1.0, 'removed': first, 'committed': True},
              {'ts': 2.0, 'removed': second, 'committed': True}]

    stream, summary_ts, _notes = atif._splice(events, SESSION)

    assert [m['content'] for m in stream] == ['first', 'second', 'live question', 'live answer']
    assert summary_ts == {1: 1.0, 2: 2.0}


# ------------------------------------------------------------------ folding into steps


def fold(stream, summary_ts=None):
    return atif._fold(stream, summary_ts or {})


def test_roles_become_steps():
    steps, notes = fold([{'role': 'user', 'content': 'q'},
                         {'role': 'assistant', 'content': 'a'}])

    assert [(step.source, step.message) for step in steps] == [('user', 'q'), ('agent', 'a')]
    assert notes == []
    assert [step.step_id for step in steps] == [1, 2]


def test_a_tool_call_and_its_result_become_one_step():
    stream = [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'function': {'name': 'read_file', 'arguments': '{"file_path": "a.py"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'file contents'},
        {'role': 'assistant', 'content': 'done'},
    ]

    steps, notes = fold(stream)

    assert len(steps) == 2
    call = steps[0]
    assert call.source == 'agent'
    assert call.message == '[tool call]'
    assert call.tool_calls[0].function_name == 'read_file'
    assert call.tool_calls[0].arguments == {'file_path': 'a.py'}
    assert call.observation.results[0].content == 'file contents'
    assert call.observation.results[0].source_call_id == 'c1'
    assert notes == []


def test_a_summary_becomes_a_system_step_with_a_compaction_marker():
    stream = [{'role': 'assistant', 'content': f'{atif.SUMMARY_PREFIX} the story so far'}]

    steps, notes = fold(stream, {0: 42.0})

    assert steps[0].source == 'system'
    assert steps[0].timestamp is not None
    assert steps[0].extra == atif.COMPACTION_EXTRA
    assert notes == []


def test_a_summary_without_the_prefix_is_a_normal_step_and_noted():
    """The positional mark disagrees with the text; the text wins."""
    stream = [{'role': 'assistant', 'content': 'no prefix here'}]

    steps, notes = fold(stream, {0: 42.0})

    assert steps[0].source == 'agent'
    assert len(notes) == 1
    assert 'positional summary mark' in notes[0]


def test_unparseable_tool_arguments_are_kept_raw():
    stream = [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c1', 'function': {'name': 'run_bash', 'arguments': '{not json'}}]}]

    steps, _notes = fold(stream)

    assert steps[0].tool_calls[0].arguments == {'raw': '{not json'}


def test_a_tool_call_without_an_id_gets_one():
    stream = [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'function': {'name': 'read_file', 'arguments': '{}'}}]}]

    steps, _notes = fold(stream)

    assert steps[0].tool_calls[0].tool_call_id


def test_an_orphan_tool_message_is_reattached_and_noted():
    stream = [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'assistant', 'content': 'thinking'},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'late result'},
    ]

    steps, notes = fold(stream)

    assert steps[-1].observation.results[0].content == 'late result'
    assert steps[-1].observation.results[0].source_call_id is None
    assert any('orphan tool message' in note for note in notes)


def test_an_unexpected_role_is_skipped_and_noted():
    steps, notes = fold([{'role': 'user', 'content': 'q'}, {'role': 'system', 'content': 'x'}])

    assert len(steps) == 1
    assert any("unexpected role 'system'" in note for note in notes)


# ------------------------------------------------------------------ token accounting


def test_telemetry_tokens_win_when_present(tmp_path):
    write(tmp_path / atif.TELE, json.dumps({'prompt_total': 1234, 'completion_total': 56}))

    assert atif._tokens(tmp_path, json.loads((tmp_path / atif.TELE).read_text())) == (
        1234, 56, 'telemetry')


def test_tokens_are_reconstructed_from_the_console_log(tmp_path):
    write(tmp_path / atif.LOG,
          '[ctx]: 100 / 300000 tokens, out 20\n'
          'noise\n'
          '[ctx]: 250 / 300000 tokens, out 30 [TRUNCATED]\n')

    assert atif._tokens(tmp_path, None) == (350, 50, 'reconstructed')


def test_tokens_are_unknown_without_either_source(tmp_path):
    assert atif._tokens(tmp_path, None) == (None, None, None)


# ------------------------------------------------------------------ building


def test_a_missing_session_builds_nothing(tmp_path):
    assert atif.build(tmp_path) is None


def test_a_session_with_only_a_system_message_builds_nothing(tmp_path):
    write(tmp_path / atif.SESSION, json.dumps([{'role': 'system', 'content': 's'}]))

    assert atif.build(tmp_path) is None


def test_a_full_round_trip_produces_a_trajectory(tmp_path):
    removed = [{'role': 'user', 'content': 'old question'}]
    write(tmp_path / atif.HISTORY, journal_lines({'ts': 5.0, 'removed': removed, 'committed': True}))
    write(tmp_path / atif.SESSION, json.dumps([
        {'role': 'system', 'content': 's'},
        {'role': 'assistant', 'content': f'{atif.SUMMARY_PREFIX} summary'},
        {'role': 'user', 'content': 'live question'},
    ]))
    write(tmp_path / atif.TELE, json.dumps({'prompt_total': 10, 'completion_total': 2,
                                            'model': 'deepseek-flash', 'outcome': 'completed'}))

    trajectory = atif.build(tmp_path, version='abc123', model_name='fallback')

    assert trajectory.agent.name == 'mini-harness'
    assert trajectory.agent.version == 'abc123'
    assert trajectory.agent.model_name == 'deepseek-flash'
    assert [step.source for step in trajectory.steps] == ['user', 'system', 'user']
    assert trajectory.final_metrics.total_prompt_tokens == 10
    assert trajectory.final_metrics.extra == {'token_source': 'telemetry'}
    assert trajectory.extra == {'outcome': 'completed'}


def test_the_trajectory_is_written_as_json(tmp_path):
    write(tmp_path / atif.SESSION, json.dumps(SESSION))
    trajectory = atif.build(tmp_path)

    path = atif.write_trajectory(tmp_path, trajectory)

    assert path == tmp_path / 'trajectory.json'
    written = json.loads(path.read_text(encoding='utf-8'))
    assert written['agent']['name'] == 'mini-harness'
    assert len(written['steps']) == 2
