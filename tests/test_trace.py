"""The JSONL event trace: off by default, append-only when configured."""

import json

import httpx
import pytest
from openai import RateLimitError

from mini_harness.retry_request import retry_call
from mini_harness.trace import TRACE, Trace


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def rate_limit_error():
    request = httpx.Request('POST', 'https://example.test/v1/chat/completions')
    response = httpx.Response(429, request=request)
    return RateLimitError('slow down', response=response, body=None)


# --------------------------------------------------------------------------- unit


def test_off_by_default_and_writes_nothing(tmp_path):
    trace = Trace()

    trace.emit('anything', value=1)

    assert trace.enabled is False
    assert list(tmp_path.iterdir()) == []


def test_configuring_without_a_path_stays_off():
    trace = Trace().configure(None)
    trace.emit('event')
    assert trace.enabled is False


def test_events_are_appended_as_json_lines(tmp_path):
    path = tmp_path / 'trace.jsonl'
    trace = Trace().configure(path)

    trace.emit('run_start', task='hi')
    trace.emit('run_end', outcome='completed')

    events = read_events(path)
    assert [e['event'] for e in events] == ['run_start', 'run_end']
    assert events[0]['task'] == 'hi'
    assert [e['seq'] for e in events] == [1, 2]
    assert all(e['ts'] > 0 for e in events)


def test_reconfiguring_restarts_the_sequence(tmp_path):
    path = tmp_path / 'trace.jsonl'
    trace = Trace().configure(path)
    trace.emit('a')
    trace.configure(path)
    trace.emit('b')
    assert [e['seq'] for e in read_events(path)] == [1, 1]


def test_missing_parent_directories_are_created(tmp_path):
    path = tmp_path / 'deep' / 'nested' / 'trace.jsonl'
    Trace().configure(path).emit('run_start')
    assert path.is_file()


def test_values_that_are_not_json_are_stringified(tmp_path):
    path = tmp_path / 'trace.jsonl'
    Trace().configure(path).emit('weird', value=object())
    assert 'object object' in read_events(path)[0]['value']


def test_a_write_failure_disables_the_trace(tmp_path, capsys):
    blocked = tmp_path / 'a-directory'
    blocked.mkdir()
    trace = Trace().configure(blocked)

    trace.emit('event')

    assert trace.enabled is False
    assert 'disabled' in capsys.readouterr().out


def test_events_are_flushed_as_they_are_written(tmp_path):
    """A crash must not lose the events already emitted."""
    path = tmp_path / 'trace.jsonl'
    trace = Trace().configure(path)

    trace.emit('run_start')

    assert len(read_events(path)) == 1       # visible without close()


def test_close_is_idempotent_and_ends_the_run(tmp_path):
    path = tmp_path / 'trace.jsonl'
    trace = Trace().configure(path)
    trace.emit('kept')

    trace.close()
    trace.close()
    trace.emit('ignored')

    assert [e['event'] for e in read_events(path)] == ['kept']
    assert trace.enabled is False


def test_reconfiguring_closes_the_previous_file(tmp_path):
    first = tmp_path / 'first.jsonl'
    second = tmp_path / 'second.jsonl'
    trace = Trace().configure(first)
    trace.emit('a')

    trace.configure(second)
    trace.emit('b')
    trace.close()

    assert [e['event'] for e in read_events(first)] == ['a']
    assert [e['event'] for e in read_events(second)] == ['b']


# --------------------------------------------------------------------------- retry


def test_retry_emits_an_event_before_backing_off(tmp_path, cfg_factory):
    path = tmp_path / 'trace.jsonl'
    TRACE.configure(path)
    # zero backoff keeps the test instant; the emission is what is under test
    cfg = cfg_factory(rate_retry=2, rate_base=0.0, rate_cap=0.0)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise rate_limit_error()
        return 'ok'

    assert retry_call(flaky, cfg=cfg) == 'ok'

    events = read_events(path)
    assert len(events) == 1
    assert events[0]['event'] == 'retry'
    assert events[0]['error'] == 'RateLimitError'
    assert events[0]['limited'] is True


def test_retry_without_a_trace_still_works(cfg_factory):
    cfg = cfg_factory(rate_retry=2, rate_base=0.0, rate_cap=0.0)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise rate_limit_error()
        return 'ok'

    assert retry_call(flaky, cfg=cfg) == 'ok'
    assert TRACE.enabled is False


def test_exhausted_retries_raise(cfg_factory):
    cfg = cfg_factory(rate_retry=1, rate_base=0.0, rate_cap=0.0)

    with pytest.raises(RateLimitError):
        retry_call(lambda: (_ for _ in ()).throw(rate_limit_error()), cfg=cfg)
