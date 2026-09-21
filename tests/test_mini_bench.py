"""Integrity checks for the model-in-the-loop mini benchmark.

The benchmark itself needs an API key, but its task definitions do not: every
task is a setup function plus a checker, and both run offline. These tests hold
that pair to its promises, because a benchmark whose tasks are already solved --
or whose checker accepts a weakened suite -- silently reports the wrong thing.

The checkers run the sandbox with a subprocess and read its output, so the whole
module skips in an environment that forbids capturing a child process.
"""

import subprocess
import sys

from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import mini_bench  # noqa: E402
from mini_harness.config import Config  # noqa: E402


def child_output_is_available() -> bool:
    try:
        done = subprocess.run([sys.executable, '-c', 'print(1)'], capture_output=True,
                              text=True, timeout=60)
    except (OSError, PermissionError):
        return False
    return done.returncode == 0 and done.stdout.strip() == '1'


pytestmark = pytest.mark.skipif(
    not child_output_is_available(),
    reason='this environment does not allow capturing a child process\'s output')


def prepare(task, workspace: Path) -> Path:
    """A fresh sandbox with the task's starting state in it."""
    sandbox = workspace / task.name
    sandbox.mkdir(parents=True, exist_ok=True)
    task.setup(sandbox)
    return sandbox


# ------------------------------------------------------------------ the task list


def test_task_names_are_unique():
    names = [task.name for task in mini_bench.TASKS]

    assert len(names) == len(set(names))


def test_every_task_states_a_prompt_and_a_reason_to_exist():
    for task in mini_bench.TASKS:
        assert task.prompt.strip(), task.name
        assert callable(task.setup) and callable(task.check), task.name


def test_every_configuration_override_is_a_real_config_field():
    known = {entry.name for entry in fields(Config)}

    for name, overrides in mini_bench.CONFIGS.items():
        unknown = sorted(set(overrides) - known)
        assert not unknown, f'{name} sets fields that do not exist: {unknown}'


# ------------------------------------------------------------------ the tasks are tasks


@pytest.mark.parametrize('task', mini_bench.TASKS, ids=lambda task: task.name)
def test_every_task_starts_unsolved(task, tmp_path):
    """A task the starting state already passes would measure nothing."""
    sandbox = prepare(task, tmp_path)

    ok, detail = task.check(sandbox, '')

    assert ok is False, f'{task.name} is already solved: {detail}'


# ------------------------------------------------------------------ the harder tier


REFERENCE = {
    'rolling_window': ('window.py',
                       'def rolling_max(values, k):\n'
                       '    """The maximum of every window of size k."""\n'
                       '    if k <= 0:\n'
                       '        raise ValueError("k must be positive")\n'
                       '    if k > len(values):\n'
                       '        return []\n'
                       '    return [max(values[i:i + k]) for i in range(len(values) - k + 1)]\n'),
    'round_half_up': ('rounding.py',
                      'from decimal import ROUND_HALF_UP, Decimal\n'
                      '\n'
                      'def round_half_up(value, places=0):\n'
                      '    """Round to the nearest value, ties away from zero."""\n'
                      '    quantum = Decimal(1).scaleb(-places)\n'
                      '    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)\n'
                      '    return int(rounded) if places == 0 else float(rounded)\n'),
    'sample_variance': ('spread.py',
                        'def variance(values):\n'
                        '    """The spread of the values, as a sample variance."""\n'
                        '    if len(values) < 2:\n'
                        '        raise ValueError("at least two values are needed")\n'
                        '    mean = sum(values) / len(values)\n'
                        '    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)\n'),
    'split_cents': ('split.py',
                    'def split_cents(total, parts):\n'
                    '    """Divide an amount in whole cents into that many parts."""\n'
                    '    if parts <= 0:\n'
                    '        raise ValueError("parts must be positive")\n'
                    '    base, extra = divmod(total, parts)\n'
                    '    return [base + 1] * extra + [base] * (parts - extra)\n'),
}

# These three ship their specification as a test file, so the checker also has
# to refuse a weakened one. split_cents has no suite to weaken.
SUITE_TASKS = ['rolling_window', 'round_half_up', 'sample_variance']


def tasks_in(names):
    return [task for task in mini_bench.TASKS if task.name in names]


def test_the_tiers_are_wired_up():
    assert sorted(REFERENCE) == ['rolling_window', 'round_half_up', 'sample_variance', 'split_cents']
    assert set(SUITE_TASKS) < set(REFERENCE)
    assert {task.name for task in tasks_in(REFERENCE)} == set(REFERENCE)


@pytest.mark.parametrize('task', tasks_in(REFERENCE), ids=lambda task: task.name)
def test_a_reference_solution_passes(task, tmp_path):
    """The checker accepts a correct fix, which is the one thing it must do."""
    sandbox = prepare(task, tmp_path)
    name, body = REFERENCE[task.name]
    (sandbox / name).write_text(body, encoding='utf-8')

    ok, detail = task.check(sandbox, '')

    assert ok is True, f'{task.name} rejects a correct fix: {detail}'


@pytest.mark.parametrize('task', tasks_in(SUITE_TASKS), ids=lambda task: task.name)
def test_the_specification_suite_cannot_be_weakened(task, tmp_path):
    """Gutting the test is not a solution, even when the code is fixed."""
    sandbox = prepare(task, tmp_path)
    name, body = REFERENCE[task.name]
    (sandbox / name).write_text(body, encoding='utf-8')
    suite = next(path for path in sorted(sandbox.glob('test_*.py')))
    suite.write_text('import unittest\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
                     encoding='utf-8')

    ok, detail = task.check(sandbox, '')

    assert ok is False
    assert 'modified' in detail


@pytest.mark.parametrize('task', tasks_in(SUITE_TASKS), ids=lambda task: task.name)
def test_a_deleted_specification_suite_is_not_a_pass(task, tmp_path):
    sandbox = prepare(task, tmp_path)
    name, body = REFERENCE[task.name]
    (sandbox / name).write_text(body, encoding='utf-8')
    next(path for path in sorted(sandbox.glob('test_*.py'))).unlink()

    ok, detail = task.check(sandbox, '')

    assert ok is False
    assert 'deleted' in detail


def test_the_specification_tasks_are_the_ones_that_ship_a_suite(tmp_path):
    """The note and the sandbox have to agree about which tier a task is in."""
    for task in mini_bench.TASKS:
        sandbox = prepare(task, tmp_path / task.name)
        suites = sorted(path.name for path in sandbox.glob('test_*.py'))
        if task.name in SUITE_TASKS:
            assert len(suites) == 1, f'{task.name}: {suites}'
            assert 'test file in the sandbox' in task.note, task.name
        else:
            assert not suites, f'{task.name} ships {suites}'
            assert 'test file in the sandbox' not in task.note, task.name


def test_the_split_task_rejects_the_obvious_even_split(tmp_path):
    """The whole point of the task: the naive implementation must fail it."""
    task = tasks_in(['split_cents'])[0]
    sandbox = prepare(task, tmp_path)
    (sandbox / 'split.py').write_text(
        'def split_cents(total, parts):\n'
        '    """Divide evenly."""\n'
        '    return [total // parts] * parts\n', encoding='utf-8')

    ok, detail = task.check(sandbox, '')

    assert ok is False
    assert '[3, 3, 3]' in detail, detail


def test_the_reference_suites_import_from_the_sandbox_root():
    """A bare `python test_x.py` has to work, so the imports must be local."""
    for suite in (mini_bench.WINDOW_SUITE, mini_bench.ROUNDING_SUITE, mini_bench.VARIANCE_SUITE):
        assert 'import unittest' in suite
        assert 'sys.path' not in suite
