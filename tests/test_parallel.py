"""Parallel execution of side-effect-free tool batches.

Overlap is proved with a barrier rather than a stopwatch: if the calls did not
run concurrently, the barrier times out and the test fails. No timing
assertions, so nothing here is flaky on a loaded machine.
"""

import json
import threading
import time

import pytest

from mini_harness.tool import box
from mini_harness.tool.box import ToolDefinition
from mini_harness.trace import TRACE
from tests.conftest import REGISTRY, call, write


def runner(cfg, **replacements):
    registry = dict(REGISTRY)
    registry.update(replacements)
    return box.ToolExecution(registry, box._always_allow, cfg=cfg)


def read_tool(function, risky=False):
    return ToolDefinition('read_file', 'stub reader', box.ReadFileInput, function, risky)


def reads(count):
    return [call('read_file', file_path=f'f{i}.py') for i in range(count)]


def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


# --------------------------------------------------------------------------- overlap


def test_a_read_batch_really_overlaps(cfg):
    gate = threading.Barrier(2, timeout=10)
    seen = []

    def slow_read(args, cfg=None):
        seen.append(args.file_path)
        gate.wait()
        return 'ok'

    execu = runner(cfg, read_file=read_tool(slow_read))

    results = execu.execute_batch(reads(2), cfg=cfg)

    assert [r.content for r in results] == ['ok', 'ok']
    assert sorted(seen) == ['f0.py', 'f1.py']


def test_results_keep_the_order_the_model_asked_for(cfg):
    def echo(args, cfg=None):
        return args.file_path

    execu = runner(cfg, read_file=read_tool(echo))

    results = execu.execute_batch(reads(8), cfg=cfg)

    assert [r.content for r in results] == [f'f{i}.py' for i in range(8)]


def test_a_batch_with_a_write_stays_serial(cfg, workspace):
    order = []

    def slow_read(args, cfg=None):
        order.append(('read', args.file_path))
        time.sleep(0.01)
        return 'r'

    def slow_write(args, cfg=None):
        order.append(('write', args.file_path))
        time.sleep(0.01)
        target = cfg.work_space / args.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding='utf-8')
        return 'w'

    execu = runner(
        cfg,
        read_file=read_tool(slow_read),
        write_file=ToolDefinition('write_file', 'stub writer', box.WriteFileInput, slow_write, False),
    )
    calls = [call('read_file', file_path='a.py'),
             call('write_file', file_path='sandbox/b.py', content='x')]

    execu.execute_batch(calls, cfg=cfg)

    assert order == [('read', 'a.py'), ('write', 'sandbox/b.py')]


# --------------------------------------------------------------------------- eligibility


def test_two_reads_are_parallelizable(cfg):
    assert runner(cfg)._parallelizable(reads(2), cfg) is True


def test_one_call_is_not_worth_a_pool(cfg):
    assert runner(cfg)._parallelizable(reads(1), cfg) is False


def test_parallelism_can_be_switched_off(cfg_factory):
    cfg = cfg_factory(parallel_tools=False)
    assert runner(cfg)._parallelizable(reads(2), cfg) is False


def test_a_risky_call_makes_the_batch_serial(cfg):
    calls = [call('read_file', file_path='a.py'), call('run_bash', command='ls')]
    assert runner(cfg)._parallelizable(calls, cfg) is False


def test_an_unknown_tool_makes_the_batch_serial(cfg):
    calls = [call('read_file', file_path='a.py'), call('nope')]
    assert runner(cfg)._parallelizable(calls, cfg) is False


def test_stateful_tools_are_not_overlapped(cfg):
    """run_todo mutates shared state, so it never joins a batch."""
    calls = [call('run_todo', items=[{'content': 'a', 'activeForm': 'a', 'status': 'pending'}])] * 2
    assert runner(cfg)._parallelizable(calls, cfg) is False


def test_glob_and_grep_are_side_effect_free(cfg):
    calls = [call('glob_file', pattern='*.py'), call('grep_file', pattern='x')]
    assert runner(cfg)._parallelizable(calls, cfg) is True


def test_the_pool_is_bounded_by_the_config(cfg_factory, monkeypatch):
    cfg = cfg_factory(max_parallel_tools=2)
    seen = {}
    real = box.ThreadPoolExecutor

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(box, 'ThreadPoolExecutor', spy)

    runner(cfg).execute_batch(reads(5), cfg=cfg)

    assert seen['max_workers'] == 2


# --------------------------------------------------------------------------- real files


def test_a_real_read_batch_returns_every_file(cfg, workspace):
    for i in range(6):
        write(workspace / 'sandbox' / f'f{i}.py', f'value = {i}\n')

    results = runner(cfg).execute_batch(
        [call('read_file', file_path=f'sandbox/f{i}.py') for i in range(6)], cfg=cfg)

    assert all(r.ok for r in results)
    assert all(f'value = {i}' in results[i].content for i in range(6))


def test_concurrent_reads_share_file_state_safely(cfg, workspace):
    """The bookkeeping is shared mutable state; all six reads must register."""
    keys = []
    for i in range(6):
        target = write(workspace / 'sandbox' / f'f{i}.py', f'value = {i}\n')
        keys.append(str(target.resolve()))

    execu = runner(cfg)
    execu.execute_batch([call('read_file', file_path=f'sandbox/f{i}.py') for i in range(6)], cfg=cfg)

    assert sorted(execu.files) == sorted(keys)
    assert all(record.level == 'full' for record in execu.files.values())


def test_a_write_tool_that_leaves_no_file_does_not_crash(cfg):
    """Regression: _record stats the path after the tool runs; it may be gone."""
    def vanishing_write(args, cfg=None):
        return 'wrote it and removed it'

    execu = runner(cfg, write_file=ToolDefinition(
        'write_file', 'stub writer', box.WriteFileInput, vanishing_write, False))

    result = execu.execute_tool(call('write_file', file_path='sandbox/ghost.py', content='x'), cfg=cfg)

    assert result.ok is True
    assert execu.files == {}


# --------------------------------------------------------------------------- trace

def test_a_parallel_batch_is_traced(cfg, workspace, session_dir):
    path = session_dir / 'trace.jsonl'
    for i in range(3):
        write(workspace / 'sandbox' / f'f{i}.py', 'x\n')
    TRACE.configure(path)
    try:
        runner(cfg).execute_batch(
            [call('read_file', file_path=f'sandbox/f{i}.py') for i in range(3)], cfg=cfg)
    finally:
        TRACE.configure(None)

    batch = next(e for e in events(path) if e['event'] == 'batch_parallel')
    assert batch['workers'] == 3
    assert batch['tools'] == ['read_file'] * 3


def test_a_serial_batch_emits_no_batch_event(cfg, workspace, session_dir):
    path = session_dir / 'trace.jsonl'
    write(workspace / 'sandbox' / 'f.py', 'x\n')
    TRACE.configure(path)
    try:
        runner(cfg).execute_batch([call('read_file', file_path='sandbox/f.py')], cfg=cfg)
    finally:
        TRACE.configure(None)

    assert not [e for e in events(path) if e['event'] == 'batch_parallel']
