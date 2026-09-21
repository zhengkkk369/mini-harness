"""A small model-in-the-loop evaluation with deterministic checks.

Unlike bench/experiments.py, a real model drives the agent here, so this can
answer questions a scripted client cannot: does the task actually get solved,
and does the verification loop change the outcome?

Every task is stdlib-only and every check is an assertion run by this process,
so pass/fail never depends on the model's own reporting. Tasks are deliberately
small: the point is to compare harness configurations, not to compete with
SWE-bench.

    uv run python -m bench.mini_bench                     # all configs
    uv run python -m bench.mini_bench --only baseline     # one config
    uv run python -m bench.mini_bench --tasks fix_logic   # one task
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / '.env')

from mini_harness.agent import DeepSeekAgent  # noqa: E402
from mini_harness.config import Config  # noqa: E402
from mini_harness.tool.box import TOOLS  # noqa: E402

WORK = ROOT / '.experiments' / 'mini-bench'


# --------------------------------------------------------------------------- helpers


def write(sandbox: Path, name: str, body: str) -> None:
    path = sandbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding='utf-8')


def run_python(sandbox: Path, code: str, timeout: int = 60):
    try:
        done = subprocess.run([sys.executable, '-c', code], cwd=sandbox, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, 'timeout'
    return done.returncode, (done.stdout + done.stderr).strip()


def run_file(sandbox: Path, name: str, timeout: int = 60):
    try:
        done = subprocess.run([sys.executable, name], cwd=sandbox, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, 'timeout'
    return done.returncode, (done.stdout + done.stderr).strip()


def tail(text: str, limit: int = 240) -> str:
    return ' '.join(text.split())[:limit]


def trace_counts(path: Path) -> dict:
    """Count the events a run emitted, so the mechanism is visible per run."""
    return {name: len(events) for name, events in _by_kind(path).items()}


def trace_events(path: Path, kind: str) -> list:
    """The events of one kind, in order, for the fields a count cannot carry."""
    return _by_kind(path).get(kind, [])


def _by_kind(path: Path) -> dict:
    grouped = {}
    if not path.exists():
        return grouped
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        grouped.setdefault(event.get('event'), []).append(event)
    return grouped


# --------------------------------------------------------------------------- tasks


@dataclass
class Task:
    name: str
    prompt: str
    setup: object
    check: object
    note: str = field(default='')


def setup_report(sandbox: Path) -> None:
    write(sandbox, 'report.py', 'def total(values)\n    return sum(values)\n\n'
                                'if __name__ == "__main__":\n    print(total([1, 2, 3]))\n')


def check_report(sandbox: Path, answer: str):
    code, out = run_file(sandbox, 'report.py')
    return (code == 0 and out == '6'), f'exit={code} out={tail(out)!r}'


def setup_stats(sandbox: Path) -> None:
    write(sandbox, 'stats.py', 'def median(values):\n    raise NotImplementedError\n')


def check_stats(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from stats import median\n'
        'assert median([3, 1, 2]) == 2, "odd"\n'
        'assert median([4, 1, 3, 2]) == 2.5, "even"\n'
        'assert median([7]) == 7, "single"\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


def setup_dedupe(sandbox: Path) -> None:
    write(sandbox, 'dedupe.py', 'def dedupe(items):\n    return list(set(items))\n')


def check_dedupe(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from dedupe import dedupe\n'
        'assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2], dedupe([3, 1, 3, 2, 1])\n'
        'assert dedupe([]) == []\n'
        'assert dedupe(["b", "a", "b"]) == ["b", "a"]\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


def setup_limit(sandbox: Path) -> None:
    write(sandbox, 'config_a.py', 'LIMIT = 10\nLABEL = "alpha"\n')
    write(sandbox, 'config_b.py', 'LIMIT = 10\nLABEL = "beta"\n')
    write(sandbox, 'check_limit.py',
          'from config_a import LIMIT as a\nfrom config_b import LIMIT as b\n'
          'assert a == 25, f"config_a.LIMIT is {a}"\n'
          'assert b == 25, f"config_b.LIMIT is {b}"\nprint("ok")\n')


def check_limit(sandbox: Path, answer: str):
    code, out = run_file(sandbox, 'check_limit.py')
    labels = run_python(sandbox, 'import config_a, config_b;'
                                 'print(config_a.LABEL, config_b.LABEL)')[1]
    ok = code == 0 and out.endswith('ok') and labels == 'alpha beta'
    return ok, f'exit={code} out={tail(out)!r} labels={labels!r}'


def setup_age(sandbox: Path) -> None:
    write(sandbox, 'parse_age.py', 'def parse_age(text):\n    return int(text)\n')


def check_age(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from parse_age import parse_age\n'
        'assert parse_age("42") == 42\n'
        'for bad in ("abc", "-5", "3.5", ""):\n'
        '    try:\n        parse_age(bad)\n'
        '    except ValueError:\n        pass\n'
        '    else:\n        raise AssertionError(f"{bad!r} did not raise ValueError")\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


def setup_import(sandbox: Path) -> None:
    write(sandbox, 'helpers.py', 'def help_me():\n    return "helped"\n')
    write(sandbox, 'app.py', 'from helpers import helper\n\n'
                             'if __name__ == "__main__":\n    print(helper())\n')


def check_import(sandbox: Path, answer: str):
    code, out = run_file(sandbox, 'app.py')
    return (code == 0 and out == 'helped'), f'exit={code} out={tail(out)!r}'


def setup_notes(sandbox: Path) -> None:
    write(sandbox, 'NOTES.md',
          '# Operations notes\n\n'
          '- The deploy window is 02:00-04:00 UTC on weekdays.\n'
          '- Rollbacks use the previous image tag.\n'
          '- On-call rotation is weekly.\n')


def check_notes(sandbox: Path, answer: str):
    ok = '02:00' in answer and '04:00' in answer
    return ok, f'answer={tail(answer, 160)!r}'


def setup_queue(sandbox: Path) -> None:
    write(sandbox, 'queue.py',
          'class Queue:\n'
          '    def __init__(self):\n'
          '        self.items = []\n'
          '        self.seen = set()\n\n'
          '    def push(self, item):\n'
          '        if item in self.seen:\n'
          '            return False\n'
          '        self.seen.add(item)\n'
          '        self.items.append(item)\n'
          '        return True\n\n'
          '    def pop(self):\n'
          '        return self.items.pop()\n')
    write(sandbox, 'check_queue.py',
          'from queue import Queue\n'
          'q = Queue()\n'
          'assert q.push("a") is True\n'
          'assert q.push("b") is True\n'
          'assert q.push("a") is False\n'
          'assert q.pop() == "a", "first in, first out"\n'
          'assert q.pop() == "b"\n'
          'assert q.pop() is None, "empty pop returns None"\n'
          'print("ok")\n')


def check_queue(sandbox: Path, answer: str):
    code, out = run_file(sandbox, 'check_queue.py')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


def setup_normalize(sandbox: Path) -> None:
    """The contract is prose only: the checker is not in the sandbox."""
    write(sandbox, 'normalize.py',
          'def normalize(text):\n'
          '    """Return the text lowercased, without leading or trailing space."""\n'
          '    return text.lower()\n')


def check_normalize(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from normalize import normalize\n'
        'assert normalize("  Hello   World  ") == "hello world", normalize("  Hello   World  ")\n'
        'assert normalize("A\\t\\tB") == "a b", repr(normalize("A\\t\\tB"))\n'
        'assert normalize("already fine") == "already fine"\n'
        'assert normalize("") == ""\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


def setup_bounds(sandbox: Path) -> None:
    write(sandbox, 'bounds.py',
          'def page_bounds(page, size):\n'
          '    """Bounds of one page of results."""\n'
          '    return (page * size, page * size + size)\n')


def check_bounds(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from bounds import page_bounds\n'
        'assert page_bounds(1, 10) == (1, 10), page_bounds(1, 10)\n'
        'assert page_bounds(2, 10) == (11, 20), page_bounds(2, 10)\n'
        'assert page_bounds(3, 5) == (11, 15), page_bounds(3, 5)\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


def setup_pure_add(sandbox: Path) -> None:
    write(sandbox, 'basket.py',
          'def add_item(items, item):\n'
          '    """Return a basket containing item as well."""\n'
          '    items.append(item)\n'
          '    return items\n')


def check_pure_add(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from basket import add_item\n'
        'original = ["apple"]\n'
        'result = add_item(original, "pear")\n'
        'assert result == ["apple", "pear"], result\n'
        'assert original == ["apple"], f"the argument was mutated: {original}"\n'
        'assert result is not original, "the same list came back"\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


def setup_id_sort(sandbox: Path) -> None:
    write(sandbox, 'ids.py',
          'def sort_ids(ids):\n'
          '    """Sort identifiers."""\n'
          '    return sorted(ids)\n')


def check_id_sort(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from ids import sort_ids\n'
        'assert sort_ids(["10", "9", "2"]) == ["2", "9", "10"], sort_ids(["10", "9", "2"])\n'
        'assert sort_ids(["b", "a"]) == ["a", "b"]\n'
        # the mixed case must be settled by the spec, because both readings of\n
        # "2 before 9 before 10" agree on the two cases above\n
        'assert sort_ids(["10", "2", "a"]) == ["10", "2", "a"], sort_ids(["10", "2", "a"])\n'
        'assert sort_ids([]) == []\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


# --------------------------------------------------------------------------- the harder tier
#
# These three ship the specification as a test file inside the sandbox. The
# contract is therefore never ambiguous -- it is written down and runnable -- but
# the prompt does not restate it, so a run has to go and read it, and the
# behaviour is only confirmed by running something. That is the shape a task
# needs if turning verification on is ever going to change the outcome.
#
# The check refuses a modified suite: weakening the test is not a solution. That
# is the same rule the system prompt states, enforced here rather than trusted.

def check_test_suite(sandbox: Path, body: str, name: str):
    """Pass only if the shipped suite is intact and green."""
    path = sandbox / name
    if not path.exists():
        return False, f'{name} was deleted; it is the specification'
    if path.read_text(encoding='utf-8') != body:
        return False, f'{name} was modified; it is the specification'
    code, out = run_file(sandbox, name)
    return code == 0, f'exit={code} out={tail(out)!r}'


def check_suite_files(sandbox: Path, files: dict, runner: str):
    """The same rule for a task whose specification spans more than one file."""
    for name, body in files.items():
        path = sandbox / name
        if not path.exists():
            return False, f'{name} was deleted; it is the specification'
        if path.read_text(encoding='utf-8') != body:
            return False, f'{name} was modified; it is the specification'
    code, out = run_file(sandbox, runner)
    return code == 0, f'exit={code} out={tail(out)!r}'


# The two tasks below are the long-horizon ones: the change spans three files, the
# failure is not visible in the file the model edits first, and the only way to
# know it is done is to run the suite. They exist because everything else in this
# suite is solved on the first attempt, which leaves nothing for a configuration
# difference to show.
CALC_SUITE = '''import unittest

from calc import percent as exported
from calc.ops import add, multiply
from calc.text import percent


class Calc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(multiply(3, 4), 12)

    def test_multiply_by_zero(self):
        self.assertEqual(multiply(3, 0), 0)

    def test_a_quarter(self):
        self.assertEqual(percent(1, 4), '25%')

    def test_rounding(self):
        self.assertEqual(percent(1, 3), '33%')

    def test_no_total(self):
        self.assertEqual(percent(3, 0), '0%')

    def test_the_package_exports_it(self):
        self.assertEqual(exported(1, 2), '50%')


if __name__ == '__main__':
    unittest.main()
'''


def setup_calc_package(sandbox: Path) -> None:
    write(sandbox, 'calc/__init__.py', 'from calc.ops import add, multiply\n')
    write(sandbox, 'calc/ops.py',
          'def add(a, b):\n'
          '    """The sum of two numbers."""\n'
          '    return a + b\n'
          '\n'
          '\n'
          'def multiply(a, b):\n'
          '    """The product of two numbers."""\n'
          '    total = 0\n'
          '    for _ in range(b - 1):\n'
          '        total += a\n'
          '    return total\n')
    write(sandbox, 'calc/text.py',
          'def percent(value, total):\n'
          '    """The value as a percentage of the total, rounded to a whole number."""\n'
          '    if total == 0:\n'
          '        return "0%"\n'
          '    return f"{value / total}"\n')
    write(sandbox, 'test_calc.py', CALC_SUITE)


def check_calc_package(sandbox: Path, answer: str):
    return check_suite_files(sandbox, {'test_calc.py': CALC_SUITE}, 'test_calc.py')


LIMITS_SUITE = '''import unittest

import limits
import reader
import writer


class SingleSource(unittest.TestCase):
    def setUp(self):
        self.original = limits.LIMIT

    def tearDown(self):
        limits.LIMIT = self.original

    def test_the_default(self):
        self.assertEqual(limits.LIMIT, 10)

    def test_the_reader_follows_the_constant(self):
        limits.LIMIT = 3
        self.assertEqual(reader.read(5), 3)

    def test_the_writer_follows_the_constant(self):
        limits.LIMIT = 3
        self.assertEqual(len(writer.write(list(range(10)))), 3)


if __name__ == '__main__':
    unittest.main()
'''


def setup_limit_modules(sandbox: Path) -> None:
    write(sandbox, 'limits.py', 'LIMIT = 10\n')
    write(sandbox, 'reader.py',
          'LIMIT = 10\n'
          '\n'
          '\n'
          'def read(page):\n'
          '    """How many results one page holds."""\n'
          '    return min(page, LIMIT)\n')
    write(sandbox, 'writer.py',
          'LIMIT = 10\n'
          '\n'
          '\n'
          'def write(items):\n'
          '    """The items that fit on one page."""\n'
          '    return items[:LIMIT]\n')
    write(sandbox, 'test_limits.py', LIMITS_SUITE)


def check_limit_modules(sandbox: Path, answer: str):
    return check_suite_files(sandbox, {'test_limits.py': LIMITS_SUITE}, 'test_limits.py')


# This one carries no checker at all, on purpose. The contract is stated in
# prose and is fully determined, but the obvious implementation -- an even
# division -- is wrong, and being wrong is invisible without running something.
# It is the only task here shaped to let the verification nudge change the
# outcome, so it is the only one that can measure the loop's benefit rather
# than just its firing.
def setup_split_cents(sandbox: Path) -> None:
    write(sandbox, 'split.py',
          'def split_cents(total, parts):\n'
          '    """Divide an amount in whole cents into that many parts."""\n'
          '    return [total // parts] * parts\n')


def check_split_cents(sandbox: Path, answer: str):
    code, out = run_python(
        sandbox,
        'from split import split_cents\n'
        'assert split_cents(10, 3) == [4, 3, 3], split_cents(10, 3)\n'
        'assert split_cents(3, 5) == [1, 1, 1, 0, 0], split_cents(3, 5)\n'
        'assert split_cents(5, 1) == [5], split_cents(5, 1)\n'
        'assert split_cents(0, 4) == [0, 0, 0, 0], split_cents(0, 4)\n'
        'shares = split_cents(100, 7)\n'
        'assert sum(shares) == 100, f"the parts sum to {sum(shares)}, not 100: {shares}"\n'
        'assert max(shares) - min(shares) <= 1, shares\n'
        'for bad in (0, -1):\n'
        '    try:\n'
        '        split_cents(10, bad)\n'
        '    except ValueError:\n'
        '        pass\n'
        '    else:\n'
        '        raise AssertionError(f"parts={bad} should raise ValueError")\n'
        'print("ok")\n')
    return (code == 0 and out.endswith('ok')), f'exit={code} out={tail(out)!r}'


WINDOW_SUITE = '''import unittest

from window import rolling_max


class RollingMax(unittest.TestCase):
    def test_the_stated_example(self):
        self.assertEqual(rolling_max([1, 3, 2, 5, 4], 2), [3, 3, 5, 5])

    def test_windows_of_one(self):
        self.assertEqual(rolling_max([4, 1, 7], 1), [4, 1, 7])

    def test_one_window_covering_everything(self):
        self.assertEqual(rolling_max([2, 9, 3], 3), [9])

    def test_a_window_longer_than_the_input(self):
        self.assertEqual(rolling_max([2, 9, 3], 5), [])

    def test_an_empty_input(self):
        self.assertEqual(rolling_max([], 2), [])


if __name__ == '__main__':
    unittest.main()
'''


def setup_rolling_window(sandbox: Path) -> None:
    write(sandbox, 'window.py',
          'def rolling_max(values, k):\n'
          '    """The maximum of every window of size k."""\n'
          '    return [max(values[i:i + k]) for i in range(len(values))]\n')
    write(sandbox, 'test_window.py', WINDOW_SUITE)


def check_rolling_window(sandbox: Path, answer: str):
    return check_test_suite(sandbox, WINDOW_SUITE, 'test_window.py')


ROUNDING_SUITE = '''import unittest

from rounding import round_half_up


class RoundHalfUp(unittest.TestCase):
    def test_half_goes_up(self):
        self.assertEqual(round_half_up(2.5), 3)

    def test_a_negative_half_goes_away_from_zero(self):
        self.assertEqual(round_half_up(-2.5), -3)

    def test_zero_point_five(self):
        self.assertEqual(round_half_up(0.5), 1)

    def test_below_the_half_stays_down(self):
        self.assertEqual(round_half_up(2.4), 2)

    def test_places_round_the_same_way(self):
        self.assertAlmostEqual(round_half_up(1.25, 1), 1.3)
        self.assertAlmostEqual(round_half_up(-1.25, 1), -1.3)

    def test_places_below_the_half_stay_down(self):
        self.assertAlmostEqual(round_half_up(1.24, 1), 1.2)


if __name__ == '__main__':
    unittest.main()
'''


def setup_round_half_up(sandbox: Path) -> None:
    write(sandbox, 'rounding.py',
          'def round_half_up(value, places=0):\n'
          '    """Round to the nearest value, at the given number of places."""\n'
          '    return round(value, places)\n')
    write(sandbox, 'test_rounding.py', ROUNDING_SUITE)


def check_round_half_up(sandbox: Path, answer: str):
    return check_test_suite(sandbox, ROUNDING_SUITE, 'test_rounding.py')


VARIANCE_SUITE = '''import unittest

from spread import variance


class Variance(unittest.TestCase):
    def test_two_values(self):
        self.assertAlmostEqual(variance([1, 3]), 2.0)

    def test_three_values(self):
        self.assertAlmostEqual(variance([2, 4, 6]), 4.0)

    def test_a_constant_list(self):
        self.assertAlmostEqual(variance([5, 5, 5]), 0.0)

    def test_a_single_value_has_no_spread(self):
        with self.assertRaises(ValueError):
            variance([7])

    def test_an_empty_list_has_no_spread(self):
        with self.assertRaises(ValueError):
            variance([])


if __name__ == '__main__':
    unittest.main()
'''


def setup_sample_variance(sandbox: Path) -> None:
    write(sandbox, 'spread.py',
          'def variance(values):\n'
          '    """The spread of the values."""\n'
          '    mean = sum(values) / len(values)\n'
          '    return sum((value - mean) ** 2 for value in values) / len(values)\n')
    write(sandbox, 'test_spread.py', VARIANCE_SUITE)


def check_sample_variance(sandbox: Path, answer: str):
    return check_test_suite(sandbox, VARIANCE_SUITE, 'test_spread.py')


TASKS = [
    Task('fix_syntax',
         'sandbox/report.py does not even run because of a syntax error on the `total` '
         'function. Fix it so that running `python report.py` prints 6.',
         setup_report, check_report),
    Task('implement_median',
         'sandbox/stats.py declares `median(values)` but raises NotImplementedError. '
         'Implement it so it returns the middle value for odd-length lists and the mean '
         'of the two middle values for even-length lists.',
         setup_stats, check_stats),
    Task('preserve_order',
         'sandbox/dedupe.py returns a list of unique items but the order is lost. Make '
         '`dedupe` keep the order in which items were first seen.',
         setup_dedupe, check_dedupe),
    Task('two_file_constant',
         'The limit is 10 in sandbox/config_a.py and sandbox/config_b.py, but it should '
         'be 25 in both. Change it in both files so that `python check_limit.py` passes, '
         'and leave every other value alone.',
         setup_limit, check_limit),
    Task('add_validation',
         'sandbox/parse_age.py takes a string and returns an int. Make it raise ValueError '
         'when the text is not a non-negative whole number (for example "abc", "-5", '
         '"3.5" or ""). Age 0 must stay valid.',
         setup_age, check_age),
    Task('fix_import',
         'sandbox/app.py imports `helper` from helpers.py, but that module defines '
         '`help_me`. Make `python app.py` print helped.',
         setup_import, check_import),
    Task('read_and_report',
         'Read sandbox/NOTES.md and tell me the deploy window in one short sentence.',
         setup_notes, check_notes),
    Task('fix_two_bugs',
         'sandbox/queue.py has two bugs: pop() takes from the wrong end, and popping an '
         'empty queue raises instead of returning None. Fix both so that '
         '`python check_queue.py` prints ok.',
         setup_queue, check_queue),
    # The four below state their contract in prose and ship no checker, so
    # whether the code ever runs is entirely up to the model.
    Task('normalize_whitespace',
         'sandbox/normalize.py should return its input lowercased with leading and '
         'trailing whitespace removed, and with any run of whitespace inside replaced by '
         'a single space. Implement that.',
         setup_normalize, check_normalize,
         note='contract in prose only; no checker in the sandbox'),
    Task('inclusive_bounds',
         'sandbox/bounds.py computes the bounds of one page of results. The caller wants '
         '1-based inclusive bounds: page 1 with size 10 is (1, 10), and page 2 with size '
         '10 is (11, 20). Make it do that.',
         setup_bounds, check_bounds,
         note='contract in prose only; no checker in the sandbox'),
    Task('pure_add_item',
         'sandbox/basket.py add_item should return a new basket containing the item, and '
         'must leave the list it was given untouched.',
         setup_pure_add, check_pure_add,
         note='contract in prose only; no checker in the sandbox'),
    Task('numeric_ids',
         'sandbox/ids.py sort_ids should order identifiers numerically when every one of '
         'them is all digits, so 2 comes before 9 and 9 before 10. When any identifier is '
         'not all digits, sort the whole list lexically instead. An empty list stays empty.',
         setup_id_sort, check_id_sort,
         note='contract in prose only; no checker in the sandbox'),
    # The harder tier: the specification is a runnable test file in the sandbox.
    # The prompt says where it is and forbids editing it, and deliberately does
    # not restate the cases, so getting it right means reading and running it.
    Task('rolling_window',
         'sandbox/window.py implements rolling_max(values, k), the maximum of every '
         'consecutive window of size k. It is wrong. sandbox/test_window.py is the '
         'specification of what it should do: make that suite pass by changing window.py '
         'only, and do not edit the test file. Run it to see where you stand.',
         setup_rolling_window, check_rolling_window,
         note='specification is a test file in the sandbox'),
    Task('round_half_up',
         'sandbox/rounding.py implements round_half_up(value, places), and it does not round '
         'the way its caller needs. sandbox/test_rounding.py is the specification: make that '
         'suite pass by changing rounding.py only, and do not edit the test file. Run it to '
         'see where you stand.',
         setup_round_half_up, check_round_half_up,
         note='specification is a test file in the sandbox'),
    Task('sample_variance',
         'sandbox/spread.py implements variance(values), and its definition of spread is not '
         'the one its caller needs. sandbox/test_spread.py is the specification: make that '
         'suite pass by changing spread.py only, and do not edit the test file. Run it to '
         'see where you stand.',
         setup_sample_variance, check_sample_variance,
         note='specification is a test file in the sandbox'),
    Task('split_cents',
         'sandbox/split.py implements split_cents(total, parts), which divides an amount of '
         'money in whole cents into that many parts. It is wrong. Every part must be a whole '
         'number of cents, the parts must sum to total exactly, no two parts may differ by '
         'more than one cent, and the extra cents go to the leftmost parts: split_cents(10, 3) '
         'must be [4, 3, 3] and split_cents(3, 5) must be [1, 1, 1, 0, 0]. A parts value of '
         'zero or less must raise ValueError. Fix split_cents.',
         setup_split_cents, check_split_cents,
         note='prose only; the obvious even split does not sum to the total'),
    # The long-horizon tier: three files, a failure the first edit does not fix,
    # and a suite that has to be run to know.
    Task('calc_package',
         'sandbox/calc/ is a small package with three problems. sandbox/test_calc.py is the '
         'specification of what it should do: make that suite pass, changing only the files '
         'under sandbox/calc/ and never the test file. More than one file needs changing, and '
         'the first one you look at is not the last one.',
         setup_calc_package, check_calc_package,
         note='specification is a test file in the sandbox; the fix spans three files'),
    Task('single_source_of_truth',
         'sandbox/reader.py and sandbox/writer.py each carry their own copy of the page limit '
         'that sandbox/limits.py defines. Make limits.py the single source of truth: setting '
         'limits.LIMIT at run time must change what both modules do. sandbox/test_limits.py is '
         'the specification; make it pass without editing it.',
         setup_limit_modules, check_limit_modules,
         note='specification is a test file in the sandbox; importing the name is not enough'),
]

CONFIGS = {
    'baseline': {},
    'no_verify': {'verify_required': False},
    'serial_tools': {'parallel_tools': False},
    # The model cannot satisfy the nudge, so this measures what the mechanism
    # does when its precondition holds and its remedy is unavailable.
    'unverifiable': {'policy_deny_tools': ('run_bash', 'run_sandbox')},
}


# --------------------------------------------------------------------------- runner


def run_one(task: Task, name: str, overrides: dict, turn_limit: int, timeout: float,
            verbose: bool = False, repeat: int = 0) -> dict:
    run_dir = WORK / name / f'{task.name}-r{repeat}'
    sandbox = run_dir / 'sandbox'
    shutil.rmtree(run_dir, ignore_errors=True)
    sandbox.mkdir(parents=True, exist_ok=True)
    task.setup(sandbox)

    cfg = Config(
        work_space=run_dir,
        session_path=str(run_dir / 'session.json'),
        trace_path=str(run_dir / 'trace.jsonl'),
        max_turns_main=turn_limit,
        wall_budget=timeout,
        **overrides)
    agent = DeepSeekAgent(TOOLS, cfg=cfg)

    start = time.time()
    if verbose:
        result = agent.run_task(task.prompt, cfg=cfg)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.run_task(task.prompt, cfg=cfg)
    elapsed = time.time() - start
    answer = ''
    if agent.message and agent.message[-1].get('role') == 'assistant':
        answer = str(agent.message[-1].get('content') or '')

    try:
        ok, detail = task.check(sandbox, answer)
    except Exception as error:                       # a broken submission is a fail
        ok, detail = False, f'check raised {type(error).__name__}: {error}'

    counts = trace_counts(run_dir / 'trace.jsonl')
    subagents = trace_events(run_dir / 'trace.jsonl', 'subagent_end')
    return {
        'task': task.name,
        'config': name,
        'repeat': repeat,
        'model': cfg.model_main,
        'passed': bool(ok),
        'detail': detail,
        'outcome': result.outcome,
        'turns': result.turns,
        'calls': result.calls,
        'errors_by_tag': result.failed_by_tag,
        'prompt_tokens': result.prompt_total,
        'completion_tokens': result.completion_total,
        'cost': round(result.cost, 6),
        'verified': result.verified,
        'mutations': result.mutations,
        'nudges': counts.get('verify_nudge', 0),
        'parallel_batches': counts.get('batch_parallel', 0),
        # A subagent that ran out of turns is not a failed tool call, so the
        # task can pass while a subtask did not. Recording both keeps that
        # visible instead of folding it into the pass/fail column.
        'subagents': len(subagents),
        'subagent_failures': [f"{event.get('agent')}:{event.get('reason')}"
                              for event in subagents if not event.get('ok')],
        'seconds': round(elapsed, 1),
    }


def render(rows: list) -> str:
    by_config = {}
    for row in rows:
        by_config.setdefault(row['config'], []).append(row)

    lines = ['# mini-bench', '',
             '| config | task | pass | turns (median) | verified | nudges | parallel | subagents |',
             '| --- | --- | :-: | ---: | :-: | ---: | ---: | :-: |']
    for name, group in by_config.items():
        tasks = {}
        for row in group:
            tasks.setdefault(row['task'], []).append(row)
        for task, runs in tasks.items():
            passed = sum(1 for r in runs if r['passed'])
            turns = sorted(r['turns'] for r in runs)
            # verified means "no edit is left unverified", which is vacuously
            # true for a run that edited nothing. Say so rather than implying
            # the work was checked.
            edited = [r for r in runs if r['mutations'] > 0]
            verified = ('n/a' if not edited
                        else f"{sum(1 for r in edited if r['verified'])}/{len(edited)}")
            nudges = sum(r['nudges'] for r in runs)
            parallel = sum(r['parallel_batches'] for r in runs)
            total = sum(r.get('subagents', 0) for r in runs)
            failures = sum(len(r.get('subagent_failures') or []) for r in runs)
            subagents = f"{total - failures}/{total}" if total else '-'
            lines.append(f"| {name} | {task} | {passed}/{len(runs)} | "
                         f"{turns[len(turns) // 2]} | {verified} | "
                         f"{nudges} | {parallel} | {subagents} |")

    lines += ['', '| config | passed | rate | runs | nudges | total tokens | total cost |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, group in by_config.items():
        passed = sum(1 for r in group if r['passed'])
        tokens = sum(r['prompt_tokens'] + r['completion_tokens'] for r in group)
        cost = sum(r['cost'] for r in group)
        nudges = sum(r['nudges'] for r in group)
        lines.append(f"| {name} | {passed}/{len(group)} | {passed / len(group):.0%} | "
                     f"{len(group)} | {nudges} | {tokens} | ${cost:.4f} |")

    lines += ['', '## failures', '']
    failures = [r for r in rows if not r['passed']]
    if failures:
        for row in failures:
            lines.append(f"- **{row['config']}/{row['task']}** r{row['repeat']}: "
                         f"{row['detail']} (outcome {row['outcome']}, {row['turns']} turns)")
    else:
        lines.append('none')
    return '\n'.join(lines) + '\n'


def aggregate(rows: list, config: str) -> dict:
    """The per-configuration summary these documents quote.

    Median turns over every row in the configuration, totals for tokens and cost,
    and `passed` as ``n/m``. One definition, so a table in a document cannot mean
    something different from the artifact it claims to summarise -- which it did:
    one table's "median tokens" column was a total, another's medians were taken
    over an even number of rows and rounded to a whole turn.
    """
    group = [row for row in rows if row['config'] == config]
    if not group:
        raise ValueError(f'no rows for configuration {config!r}')
    turns = sorted(row['turns'] for row in group)
    return {
        'config': config,
        'passed': f"{sum(1 for row in group if row['passed'])}/{len(group)}",
        'median_turns': turns[len(turns) // 2] if len(turns) % 2
                        else (turns[len(turns) // 2 - 1] + turns[len(turns) // 2]) / 2,
        'total_tokens': sum(row['prompt_tokens'] + row['completion_tokens'] for row in group),
        'total_cost': sum(row['cost'] for row in group),
        'nudges': sum(row['nudges'] for row in group),
        'subagent_failures': sum(len(row.get('subagent_failures') or []) for row in group),
        'runs': len(group),
    }


def summary_row(summary: dict) -> str:
    """The table row the documents carry: one line per configuration."""
    return (f"| `{summary['config']}` | {summary['passed']} | "
            f"{summary['median_turns']:g} | {summary['total_tokens']:,} | "
            f"${summary['total_cost']:.4f} |")


# The census table's rows, in the order MODEL_EVAL.md presents them: which
# mechanisms could engage in a run of that configuration.
CENSUS_ORDER = ((True, True), (True, False), (False, True), (False, False))


def exposure(config: str) -> tuple:
    """``(verify_required, shell available)`` for a configuration.

    Read off the configuration's own overrides rather than restated, so a
    configuration that starts denying the shell cannot leave the census table
    describing the previous set.
    """
    overrides = CONFIGS.get(config)
    if overrides is None:
        raise ValueError(f'unknown configuration {config!r}')
    denied = tuple(overrides.get('policy_deny_tools', ()))
    return (bool(overrides.get('verify_required', True)),
            not ('run_bash' in denied or 'run_sandbox' in denied))


def census_rows(paths) -> list:
    """MODEL_EVAL.md's "every run this repository has recorded" table.

    The document claimed to count every recorded run while its numbers described
    an earlier repository: two more tiers were recorded and the table still read
    45/8/69 against 146. It is derived from the artifacts here for the same
    reason the per-configuration summaries are.
    """
    groups = {}
    for path in paths:
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
        for row in rows:
            bucket = groups.setdefault(exposure(row['config']), {'runs': 0, 'nudges': 0})
            bucket['runs'] += 1
            bucket['nudges'] += row.get('nudges', 0) or 0
    return [_census_row(key, groups[key]) for key in CENSUS_ORDER if key in groups]


def _census_row(key: tuple, bucket: dict) -> str:
    verify, shell = key
    runs, nudges = bucket['runs'], bucket['nudges']
    if not verify:
        shown = 'n/a'
    elif nudges and nudges == runs:
        shown = f'{nudges} — one per run'
    else:
        shown = f'**{nudges}**'
    return (f"| {runs} | {'on' if verify else 'off'} | "
            f"{'yes' if shell else 'no (denied by policy)'} | {shown} |")


def apply_overrides(configs: dict, model: str = None, sub_model: str = None,
                    prices: dict = None) -> dict:
    """Fold the command line's overrides into every configuration.

    ``--model`` is the headroom lever: a suite every configuration solves says
    nothing about either, so a weaker model is the other way to look for a
    difference. Prices apply to all configurations for the same reason.
    """
    for overrides in configs.values():
        if model:
            overrides['model_main'] = model
        if sub_model:
            overrides['model_sub'] = sub_model
        for name, value in (prices or {}).items():
            if value is not None:
                overrides[name] = value
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='MINI_BENCH.json')
    parser.add_argument('--only', action='append', default=None, help='config name (repeatable)')
    parser.add_argument('--tasks', action='append', default=None, help='task name (repeatable)')
    parser.add_argument('--turns', type=int, default=30)
    parser.add_argument('--timeout', type=float, default=240.0, help='wall budget per run, seconds')
    parser.add_argument('--repeats', type=int, default=1, help='runs per task and configuration')
    parser.add_argument('--model', default=None,
                        help='override the model for every configuration, to look for headroom')
    parser.add_argument('--sub-model', default=None, help='override the sub-model (compaction, subagents)')
    parser.add_argument('--price-in', type=float, default=None, help='dollars per million prompt tokens')
    parser.add_argument('--price-out', type=float, default=None, help='dollars per million output tokens')
    parser.add_argument('--price-cache-in', type=float, default=None,
                        help='dollars per million cached prompt tokens; unset bills them at --price-in')
    parser.add_argument('--verbose', action='store_true', help='do not silence the agent')
    args = parser.parse_args()

    configs = {k: dict(v) for k, v in CONFIGS.items() if not args.only or k in args.only}
    tasks = [t for t in TASKS if not args.tasks or t.name in args.tasks]
    WORK.mkdir(parents=True, exist_ok=True)

    apply_overrides(configs, model = args.model, sub_model = args.sub_model,
                    prices = {'price_in': args.price_in, 'price_out': args.price_out,
                              'price_cache_in': args.price_cache_in})

    rows = []
    for name, overrides in configs.items():
        for task in tasks:
            for repeat in range(args.repeats):
                print(f'running {name}/{task.name} r{repeat + 1}/{args.repeats} ...', flush=True)
                row = run_one(task, name, overrides, args.turns, args.timeout, args.verbose, repeat)
                rows.append(row)
                print(f'  -> {"pass" if row["passed"] else "FAIL"} in {row["seconds"]}s, '
                      f'{row["turns"]} turns, {row["calls"]} calls, {row["nudges"]} nudges '
                      f'| {row["detail"]}', flush=True)
                Path(args.out).write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')

    report = render(rows)
    print()
    print(report)
    print(f'raw results written to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
