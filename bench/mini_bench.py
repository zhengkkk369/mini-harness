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
]

CONFIGS = {
    'baseline': {},
    'no_verify': {'verify_required': False},
    'serial_tools': {'parallel_tools': False},
}


# --------------------------------------------------------------------------- runner


def run_one(task: Task, name: str, overrides: dict, turn_limit: int, timeout: float,
            verbose: bool = False) -> dict:
    run_dir = WORK / name / task.name
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

    return {
        'task': task.name,
        'config': name,
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
        'seconds': round(elapsed, 1),
    }


def render(rows: list) -> str:
    by_config = {}
    for row in rows:
        by_config.setdefault(row['config'], []).append(row)

    lines = ['# mini-bench', '', '| config | task | pass | turns | calls | tokens | cost | verified | s |',
             '| --- | --- | :-: | ---: | ---: | ---: | ---: | :-: | ---: |']
    for name, group in by_config.items():
        for row in group:
            tokens = row['prompt_tokens'] + row['completion_tokens']
            lines.append(f"| {name} | {row['task']} | {'yes' if row['passed'] else 'NO'} | "
                         f"{row['turns']} | {row['calls']} | {tokens} | ${row['cost']:.4f} | "
                         f"{'yes' if row['verified'] else 'no'} | {row['seconds']} |")
    lines += ['', '| config | passed | rate | median turns | total tokens | total cost |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, group in by_config.items():
        passed = sum(1 for r in group if r['passed'])
        turns = sorted(r['turns'] for r in group)
        median = turns[len(turns) // 2] if turns else 0
        tokens = sum(r['prompt_tokens'] + r['completion_tokens'] for r in group)
        cost = sum(r['cost'] for r in group)
        lines.append(f"| {name} | {passed}/{len(group)} | {passed / len(group):.0%} | {median} | "
                     f"{tokens} | ${cost:.4f} |")
    lines += ['', '## failures', '']
    failures = [r for r in rows if not r['passed']]
    if failures:
        for row in failures:
            lines.append(f"- **{row['config']}/{row['task']}**: {row['detail']} "
                         f"(outcome {row['outcome']}, {row['turns']} turns)")
    else:
        lines.append('none')
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='MINI_BENCH.json')
    parser.add_argument('--only', action='append', default=None, help='config name (repeatable)')
    parser.add_argument('--tasks', action='append', default=None, help='task name (repeatable)')
    parser.add_argument('--turns', type=int, default=30)
    parser.add_argument('--timeout', type=float, default=240.0, help='wall budget per run, seconds')
    parser.add_argument('--price-in', type=float, default=None, help='dollars per million prompt tokens')
    parser.add_argument('--price-out', type=float, default=None, help='dollars per million output tokens')
    parser.add_argument('--verbose', action='store_true', help='do not silence the agent')
    args = parser.parse_args()

    configs = {k: dict(v) for k, v in CONFIGS.items() if not args.only or k in args.only}
    tasks = [t for t in TASKS if not args.tasks or t.name in args.tasks]
    WORK.mkdir(parents=True, exist_ok=True)

    if args.price_in is not None or args.price_out is not None:
        for overrides in configs.values():
            overrides['price_in'] = args.price_in
            overrides['price_out'] = args.price_out

    rows = []
    for name, overrides in configs.items():
        for task in tasks:
            print(f'running {name}/{task.name} ...', flush=True)
            row = run_one(task, name, overrides, args.turns, args.timeout, args.verbose)
            rows.append(row)
            print(f'  -> {"pass" if row["passed"] else "FAIL"} in {row["seconds"]}s, '
                  f'{row["turns"]} turns, {row["calls"]} calls | {row["detail"]}', flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')

    report = render(rows)
    print()
    print(report)
    print(f'raw results written to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
