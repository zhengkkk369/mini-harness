"""Integrity checks for the model-in-the-loop mini benchmark.

The benchmark itself needs an API key, but its task definitions do not: every
task is a setup function plus a checker, and both run offline. These tests hold
that pair to its promises, because a benchmark whose tasks are already solved --
or whose checker accepts a weakened suite -- silently reports the wrong thing.

The checkers run the sandbox with a subprocess and read its output, so the whole
module skips in an environment that forbids capturing a child process.
"""

import json
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
    'rolling_window': {'window.py':
                       'def rolling_max(values, k):\n'
                       '    """The maximum of every window of size k."""\n'
                       '    if k <= 0:\n'
                       '        raise ValueError("k must be positive")\n'
                       '    if k > len(values):\n'
                       '        return []\n'
                       '    return [max(values[i:i + k]) for i in range(len(values) - k + 1)]\n'},
    'round_half_up': {'rounding.py':
                      'from decimal import ROUND_HALF_UP, Decimal\n'
                      '\n'
                      'def round_half_up(value, places=0):\n'
                      '    """Round to the nearest value, ties away from zero."""\n'
                      '    quantum = Decimal(1).scaleb(-places)\n'
                      '    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)\n'
                      '    return int(rounded) if places == 0 else float(rounded)\n'},
    'sample_variance': {'spread.py':
                        'def variance(values):\n'
                        '    """The spread of the values, as a sample variance."""\n'
                        '    if len(values) < 2:\n'
                        '        raise ValueError("at least two values are needed")\n'
                        '    mean = sum(values) / len(values)\n'
                        '    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)\n'},
    'split_cents': {'split.py':
                    'def split_cents(total, parts):\n'
                    '    """Divide an amount in whole cents into that many parts."""\n'
                    '    if parts <= 0:\n'
                    '        raise ValueError("parts must be positive")\n'
                    '    base, extra = divmod(total, parts)\n'
                    '    return [base + 1] * extra + [base] * (parts - extra)\n'},
    # Three files, and none of them is sufficient alone.
    'calc_package': {
        'calc/ops.py':
            'def add(a, b):\n'
            '    """The sum of two numbers."""\n'
            '    return a + b\n'
            '\n'
            '\n'
            'def multiply(a, b):\n'
            '    """The product of two numbers."""\n'
            '    return a * b\n',
        'calc/text.py':
            'def percent(value, total):\n'
            '    """The value as a percentage of the total, rounded to a whole number."""\n'
            '    if total == 0:\n'
            '        return "0%"\n'
            '    return f"{round(value / total * 100)}%"\n',
        'calc/__init__.py':
            'from calc.ops import add, multiply\n'
            'from calc.text import percent\n',
    },
    # Reading the constant at call time, from the module, is the whole task:
    # `from limits import LIMIT` binds once and fails the suite.
    'single_source_of_truth': {
        'reader.py':
            'import limits\n'
            '\n'
            '\n'
            'def read(page):\n'
            '    """How many results one page holds."""\n'
            '    return min(page, limits.LIMIT)\n',
        'writer.py':
            'import limits\n'
            '\n'
            '\n'
            'def write(items):\n'
            '    """The items that fit on one page."""\n'
            '    return items[:limits.LIMIT]\n',
    },
}

# The three with a shipped suite in the sandbox, which the checker also has to
# refuse to see weakened. split_cents has no suite to weaken.
SUITE_TASKS = ['rolling_window', 'round_half_up', 'sample_variance',
               'calc_package', 'single_source_of_truth']


def tasks_in(names):
    return [task for task in mini_bench.TASKS if task.name in names]


def test_the_tiers_are_wired_up():
    assert sorted(REFERENCE) == ['calc_package', 'rolling_window', 'round_half_up',
                                 'sample_variance', 'single_source_of_truth', 'split_cents']
    assert set(SUITE_TASKS) < set(REFERENCE)
    assert {task.name for task in tasks_in(REFERENCE)} == set(REFERENCE)


@pytest.mark.parametrize('task', tasks_in(REFERENCE), ids=lambda task: task.name)
def test_a_reference_solution_passes(task, tmp_path):
    """The checker accepts a correct fix, which is the one thing it must do."""
    sandbox = prepare(task, tmp_path)
    for name, body in REFERENCE[task.name].items():
        (sandbox / name).write_text(body, encoding='utf-8')

    ok, detail = task.check(sandbox, '')

    assert ok is True, f'{task.name} rejects a correct fix: {detail}'


@pytest.mark.parametrize('task', tasks_in(SUITE_TASKS), ids=lambda task: task.name)
def test_the_specification_suite_cannot_be_weakened(task, tmp_path):
    """Gutting the test is not a solution, even when the code is fixed."""
    sandbox = prepare(task, tmp_path)
    for name, body in REFERENCE[task.name].items():
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
    for name, body in REFERENCE[task.name].items():
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


# ------------------------------------------------------------------ reading the trace back


def trace_file(tmp_path, *events):
    path = tmp_path / 'trace.jsonl'
    path.write_text(''.join(json.dumps(event) + '\n' for event in events), encoding='utf-8')
    return path


def test_trace_counts_still_counts(tmp_path):
    path = trace_file(tmp_path, {'event': 'turn'}, {'event': 'tool_call'}, {'event': 'turn'})

    assert mini_bench.trace_counts(path) == {'turn': 2, 'tool_call': 1}


def test_trace_events_keeps_the_fields_a_count_cannot_carry(tmp_path):
    path = trace_file(tmp_path,
                      {'event': 'subagent_end', 'agent': 'explore_agent', 'ok': True,
                       'reason': '', 'turns': 2},
                      {'event': 'turn'},
                      {'event': 'subagent_end', 'agent': 'coding_agent', 'ok': False,
                       'reason': 'exhausted', 'turns': 20})

    ends = mini_bench.trace_events(path, 'subagent_end')

    assert [event['agent'] for event in ends] == ['explore_agent', 'coding_agent']
    assert ends[1]['reason'] == 'exhausted'


def test_a_missing_or_broken_trace_is_not_an_error(tmp_path):
    assert mini_bench.trace_events(tmp_path / 'absent.jsonl', 'subagent_end') == []
    broken = tmp_path / 'broken.jsonl'
    broken.write_text('not json\n{"event": "turn"}\n', encoding='utf-8')

    assert mini_bench.trace_events(broken, 'turn') == [{'event': 'turn'}]


def test_the_report_shows_subagent_outcomes(tmp_path):
    rows = [
        {'task': 't', 'config': 'baseline', 'repeat': 0, 'passed': True, 'detail': '',
         'outcome': 'completed', 'turns': 4, 'calls': 4, 'errors_by_tag': {},
         'prompt_tokens': 1, 'completion_tokens': 1, 'cost': 0.0, 'verified': True,
         'mutations': 1, 'nudges': 0, 'parallel_batches': 0, 'seconds': 1.0,
         'subagents': 2, 'subagent_failures': ['coding_agent:exhausted']},
    ]

    report = mini_bench.render(rows)

    assert '| subagents |' in report
    assert '| 1/2 |' in report


def test_a_run_without_subagents_shows_a_dash(tmp_path):
    rows = [
        {'task': 't', 'config': 'baseline', 'repeat': 0, 'passed': True, 'detail': '',
         'outcome': 'completed', 'turns': 4, 'calls': 4, 'errors_by_tag': {},
         'prompt_tokens': 1, 'completion_tokens': 1, 'cost': 0.0, 'verified': True,
         'mutations': 1, 'nudges': 0, 'parallel_batches': 0, 'seconds': 1.0},
    ]

    assert '| - |' in mini_bench.render(rows)


# ------------------------------------------------------------------ command-line overrides


def test_the_model_override_reaches_every_configuration():
    """A weaker model is the other way to look for headroom."""
    configs = {'baseline': {}, 'no_verify': {'verify_required': False}}

    mini_bench.apply_overrides(configs, model='deepseek-chat', sub_model='deepseek-chat')

    assert configs == {'baseline': {'model_main': 'deepseek-chat', 'model_sub': 'deepseek-chat'},
                       'no_verify': {'verify_required': False, 'model_main': 'deepseek-chat',
                                     'model_sub': 'deepseek-chat'}}


def test_prices_reach_every_configuration():
    configs = {'baseline': {}}

    mini_bench.apply_overrides(configs, prices={'price_in': 0.15, 'price_out': 0.6,
                                                'price_cache_in': None})

    assert configs['baseline'] == {'price_in': 0.15, 'price_out': 0.6}


def test_no_override_leaves_the_configurations_alone():
    configs = {'baseline': {'verify_required': False}}

    assert mini_bench.apply_overrides(configs) == {'baseline': {'verify_required': False}}


# ------------------------------------------------------------------ the recorded rows


def test_tool_latency_reads_the_durations_the_trace_already_carries(tmp_path):
    """The run-level wall clock cannot say which call was slow.

    One run in this repository finished in 283 s because a single tool call waited
    out its own timeout; a p95 over runs reports that as "a slow run". Every
    `tool_result` carries its own `seconds`, so the distribution is free.
    """
    path = tmp_path / 'trace.jsonl'
    events = [{'event': 'tool_result', 'tool': f't{i}', 'seconds': value}
              for i, value in enumerate([0.01, 0.02, 0.03, 0.04, 2.0])]
    events.append({'event': 'turn', 'seconds': 99.0})
    path.write_text('\n'.join(json.dumps(event) for event in events) + '\n', encoding='utf-8')

    latency = mini_bench.tool_latency(path)

    assert latency['tool_calls'] == 5
    assert latency['tool_p50_ms'] == 30.0
    assert latency['tool_p95_ms'] == 2000.0
    assert latency['tool_max_ms'] == 2000.0


def test_tool_latency_of_a_missing_trace_is_zero_not_an_error(tmp_path):
    latency = mini_bench.tool_latency(tmp_path / 'absent.jsonl')

    assert latency == {'tool_calls': 0, 'tool_p50_ms': 0.0, 'tool_p95_ms': 0.0, 'tool_max_ms': 0.0}


def test_the_extra_tools_are_the_same_surface_the_exposure_experiment_counts():
    """The A/B and the payload measurement have to describe one tool surface.

    Otherwise "schema tokens fell by 67%" and "the tasks still pass" would be
    about two different things, which is exactly the confusion the tier exists to
    avoid.
    """
    from bench.experiments import experiment_exposure

    surface = list(mini_bench.TOOLS) + mini_bench.external_surface(24)
    exposure = experiment_exposure(24, (0, 8), 'fix the failing test in the parser')

    assert len(surface) == exposure[0]['exposed'] == 36
    assert len({tool.name for tool in surface}) == 36


def test_the_budgeted_configuration_asks_for_a_real_field():
    assert mini_bench.CONFIGS['budgeted_tools'] == {'tool_budget': 8}
    assert 'tool_budget' in {field.name for field in fields(Config)}


def test_a_recorded_run_carries_the_accounting_its_cost_is_quoted_with():
    """`cost` alone hides a 4x difference, and hides any subtask that was billed.

    The artifact is committed, so this also pins the shape of the rows the
    documents summarise: a run whose cost is quoted has to say how much of its
    input was cached, what the same usage would cost without that discount, and
    which part of the run spent it.
    """
    path = ROOT / 'MINI_BENCH_PRICED.json'
    if not path.exists():
        pytest.skip('no priced run recorded here')
    rows = json.loads(path.read_text(encoding='utf-8'))

    for row in rows:
        assert 'cached_tokens' in row and 'cost_naive' in row and 'usage_by_source' in row
        assert 0 <= row['cached_tokens'] <= row['prompt_tokens']
        assert row['cost_naive'] >= row['cost']
        assert row['usage_by_source'], 'a run always bills at least its own turns'
        assert sum(entry['prompt'] for entry in row['usage_by_source'].values()) == \
            row['prompt_tokens']
        assert sum(entry['completion'] for entry in row['usage_by_source'].values()) == \
            row['completion_tokens']
