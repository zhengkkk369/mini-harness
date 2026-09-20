"""Offline, reproducible experiments for the harness mechanisms.

There is no model in the loop and no network: every experiment drives the real
agent and tool code with a scripted client, so the numbers measure the harness
itself. That is a deliberate scope choice, not a substitute for an end-to-end
agent evaluation, and the results are reported as such.

    uv run python -m bench.experiments            # prints a report, writes EXPERIMENTS.json
    uv run python -m bench.experiments --repeats 15

Experiments:

    parallel   wall time of a read-only tool batch, serial vs concurrent
    trace      wall time and bytes written with the event trace off vs on
    verify     turns, nudges and the verified flag for the verification loop
    policy     which calls a deny policy refuses, and under which tag
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('DEEPSEEK_API_KEY', 'experiments-offline')
os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mini_harness import agent as agent_module  # noqa: E402
from mini_harness.agent import DeepSeekAgent  # noqa: E402
from mini_harness.config import Config  # noqa: E402
from mini_harness.tool import box  # noqa: E402
from mini_harness.tool.box import TOOLS, ToolDefinition  # noqa: E402
from mini_harness.trace import TRACE  # noqa: E402

WORK = Path('.experiments').resolve()
READ_TOOL = next(tool for tool in TOOLS if tool.name == 'read_file')
BASH_TOOL = next(tool for tool in TOOLS if tool.name == 'run_bash')


# --------------------------------------------------------------------------- helpers


def call(name, **arguments):
    return SimpleNamespace(id=f'call_{name}', function=SimpleNamespace(
        name=name, arguments=json.dumps(arguments)))


def message(content='', tool_calls=None):
    return SimpleNamespace(
        role='assistant', content=content, tool_calls=tool_calls, reasoning_content='',
        model_dump=lambda **_: {'role': 'assistant', 'content': content})


def completion(msg, prompt_tokens=100, completion_tokens=20):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                                 completion_tokens=completion_tokens))


class Stream:
    """Stands in for openai's streaming context manager."""

    def __init__(self, complete):
        self.complete = complete

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def get_final_completion(self):
        return self.complete


class ScriptedClient:
    """Replays prepared responses through the real agent code."""

    def __init__(self, script):
        self.script = list(script)

    def _stream(self, **kwargs):
        item = self.script.pop(0)
        # accept either a bare completion or an already wrapped stream
        return item if hasattr(item, '__enter__') else Stream(item)

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(stream=self._stream))


def config_for(workdir, **overrides):
    base = dict(work_space=workdir, session_path=str(workdir / 'session.json'))
    base.update(overrides)
    return Config(**base)


def write_tree(root, count, size=4096):
    """A directory of readable files, so read_file has real work to do."""
    target = root / 'tree'
    target.mkdir(parents=True, exist_ok=True)
    body = ('x' * (size - 12)) + '\n'
    paths = []
    for i in range(count):
        path = target / f'f{i:03d}.py'
        path.write_text(f'value = {i}\n{body}', encoding='utf-8')
        paths.append(path)
    return paths


def read_batch(count):
    return [call('read_file', file_path=f'tree/f{i:03d}.py') for i in range(count)]


def median_seconds(repeats, action):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples), max(samples)


def executor_with(cfg, tool):
    registry = {t.name: t for t in TOOLS}
    registry[tool.name] = tool
    return box.ToolExecution(registry, box._always_allow, cfg=cfg)


# --------------------------------------------------------------------------- experiments


def experiment_parallel(repeats, calls, latencies):
    """Serial vs concurrent wall time for a batch of read-only calls."""
    workdir = WORK / 'parallel'
    write_tree(workdir, calls)
    rows = []
    for latency in latencies:
        real_read = READ_TOOL.function

        def delayed(args, cfg=None, _delay=latency, _real=real_read):
            if _delay:
                time.sleep(_delay)
            return _real(args, cfg=cfg)

        tool = ToolDefinition('read_file', 'timed reader', box.ReadFileInput, delayed, False)
        row = {'latency_ms': latency * 1000, 'calls': calls}
        for label, parallel in (('serial', False), ('parallel', True)):
            cfg = config_for(workdir, parallel_tools=parallel, max_parallel_tools=calls)
            execu = executor_with(cfg, tool)
            batch = read_batch(calls)
            med, low, high = median_seconds(
                repeats, lambda e=execu, c=cfg, b=batch: e.execute_batch(b, cfg=c))
            row[label] = {'median_ms': med * 1000, 'min_ms': low * 1000, 'max_ms': high * 1000}
        row['speedup'] = row['serial']['median_ms'] / row['parallel']['median_ms']
        rows.append(row)
    return rows


def run_scripted(cfg, script, quiet=True):
    """Drive the real agent with a scripted model; return (result, client, agent)."""
    client = ScriptedClient(script)
    agent_module.OpenAI = lambda **_: client
    agent = DeepSeekAgent(TOOLS, cfg=cfg)
    agent.session_memory = Path(cfg.session_path)
    if quiet:
        # the agent narrates every turn; a measurement report does not need it
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.run_task('solve the task', cfg=cfg)
    else:
        result = agent.run_task('solve the task', cfg=cfg)
    return result, client, agent


def experiment_verify(repeats):
    """What the verification loop costs and what it notices."""
    workdir = WORK / 'verify'

    def seed():
        """The editable file lives under sandbox/, where writes are allowed."""
        target = workdir / 'sandbox'
        target.mkdir(parents=True, exist_ok=True)
        (target / 'f.py').write_text('value = 0\n', encoding='utf-8')

    seed()
    edit = call('edit_file', file_path='sandbox/f.py', old_string='value = 0', new_string='value = 1')
    read = call('read_file', file_path='sandbox/f.py')
    # a shell tool that always succeeds: what matters is that a run happened
    run = call('run_bash', command='echo verified')

    scenarios = {
        'off, edit then answer': (dict(verify_required=False),
                                  [completion(message('', [read])),
                                   completion(message('', [edit])),
                                   completion(message('done'))]),
        'on, edit then answer': (dict(verify_required=True),
                                 [completion(message('', [read])),
                                  completion(message('', [edit])),
                                  completion(message('done')),
                                  completion(message('done again'))]),
        'on, edit then run then answer': (dict(verify_required=True),
                                          [completion(message('', [read])),
                                           completion(message('', [edit])),
                                           completion(message('', [run])),
                                           completion(message('done'))]),
    }
    rows = []
    for label, (overrides, script) in scenarios.items():
        profile = []
        for _ in range(repeats):
            seed()
            cfg = config_for(workdir, **overrides)
            result, _client, agent = run_scripted(cfg, list(script))
            profile.append({
                'turns': result.turns,
                'mutations': result.mutations,
                'verified': result.verified,
                'nudges': sum(1 for m in agent.message
                              if m.get('role') == 'user'
                              and 'have not run anything since' in str(m.get('content'))),
                'outcome': result.outcome,
            })
        rows.append({'scenario': label, **profile[-1]})
    return rows


def experiment_trace(repeats, turns):
    """The cost of recording every event."""
    workdir = WORK / 'trace'
    write_tree(workdir, 1)
    read = call('read_file', file_path='tree/f000.py')
    script = [completion(message('', [read])) for _ in range(turns)]
    script.append(completion(message('done')))
    rows = []
    for label, enabled in (('off', False), ('on', True)):
        sizes = []

        def one_run():
            trace_file = workdir / f'trace-{label}.jsonl'
            trace_file.unlink(missing_ok=True)
            cfg = config_for(workdir, trace_path=str(trace_file) if enabled else None)
            TRACE.configure(None)
            run_scripted(cfg, list(script))
            sizes.append(trace_file.stat().st_size if trace_file.exists() else 0)

        med, low, high = median_seconds(repeats, one_run)
        TRACE.configure(None)
        rows.append({'trace': label, 'median_ms': med * 1000, 'min_ms': low * 1000,
                     'max_ms': high * 1000, 'bytes': sizes[-1]})
    rows.append({'trace': 'overhead', 'delta_ms': rows[1]['median_ms'] - rows[0]['median_ms'],
                 'bytes': rows[1]['bytes']})
    return rows


def experiment_policy():
    """Which calls a deny policy refuses, and under which tag."""
    workdir = WORK / 'policy'
    write_tree(workdir, 1)
    cfg = config_for(workdir, policy_deny_tools=('run_sandbox',),
                     policy_deny_patterns=('rm -rf*',))
    execu = box.ToolExecution({t.name: t for t in TOOLS}, box._always_allow, cfg=cfg)
    calls = {
        'run_bash: rm -rf /': call('run_bash', command='rm -rf /'),
        'run_bash: ls -la': call('run_bash', command='ls -la'),
        'run_sandbox: ls': call('run_sandbox', command='ls'),
        'read_file: tree/f000.py': call('read_file', file_path='tree/f000.py'),
    }
    rows = []
    for label, tool_call in calls.items():
        item = execu.execute_tool(tool_call, cfg=cfg)
        rows.append({'call': label, 'ok': item.ok, 'tag': item.tag or 'success'})
    return rows


# --------------------------------------------------------------------------- report


def render(results):
    lines = ['# mini-harness harness experiments', '']
    lines += ['## Parallel tool execution', '',
              'Batch of read-only calls through ToolExecution.execute_batch.',
              '', '| injected latency | calls | serial (ms) | parallel (ms) | speedup |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for row in results['parallel']:
        lines.append(f"| {row['latency_ms']:.0f} ms | {row['calls']} | "
                     f"{row['serial']['median_ms']:.2f} | {row['parallel']['median_ms']:.2f} | "
                     f"{row['speedup']:.2f}x |")
    lines += ['', '## Event trace', '', '| trace | median (ms) | min (ms) | max (ms) | bytes |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in results['trace']:
        if 'median_ms' in row:
            lines.append(f"| {row['trace']} | {row['median_ms']:.2f} | {row['min_ms']:.2f} | "
                         f"{row['max_ms']:.2f} | {row['bytes']} |")
        else:
            lines.append(f"| {row['trace']} | {row['delta_ms']:+.2f} (delta) | | | {row['bytes']} |")
    lines += ['', '## Verification loop', '',
              '| scenario | turns | mutations | nudges | verified | outcome |',
              '| --- | ---: | ---: | ---: | --- | --- |']
    for row in results['verify']:
        lines.append(f"| {row['scenario']} | {row['turns']} | {row['mutations']} | "
                     f"{row['nudges']} | {row['verified']} | {row['outcome']} |")
    lines += ['', '## Dispatch policy', '', '| call | ok | tag |', '| --- | --- | --- |']
    for row in results['policy']:
        lines.append(f"| {row['call']} | {row['ok']} | {row['tag']} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='EXPERIMENTS.json', help='where to write the raw results')
    parser.add_argument('--repeats', type=int, default=7, help='timed runs per configuration')
    parser.add_argument('--calls', type=int, default=6, help='calls per read batch')
    parser.add_argument('--turns', type=int, default=20, help='tool turns per trace run')
    args = parser.parse_args()

    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    payload = {
        'python': sys.version.split()[0],
        'platform': sys.platform,
        'repeats': args.repeats,
        'parallel': experiment_parallel(args.repeats, args.calls, (0.0, 0.005, 0.02)),
        'trace': experiment_trace(args.repeats, args.turns),
        'verify': experiment_verify(args.repeats),
        'policy': experiment_policy(),
    }
    report = render(payload)
    print(report)
    Path(args.out).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'raw results written to {args.out}')
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
