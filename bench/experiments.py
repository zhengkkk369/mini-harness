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
    recall     whether recall ranks a planted detail first as the archive grows
    vector     what the vector backend embeds and what it costs to search
    exposure   request schema payload of a tool surface, with and without a budget
"""

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

os.environ.setdefault('DEEPSEEK_API_KEY', 'experiments-offline')
os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mini_harness import agent as agent_module  # noqa: E402
from mini_harness.agent import CORE_TOOLS, DeepSeekAgent  # noqa: E402
from mini_harness.config import Config  # noqa: E402
from mini_harness.embed import HashEmbedder  # noqa: E402
from mini_harness.memory import Memory  # noqa: E402
from mini_harness.selector import SELECTION, select  # noqa: E402
from mini_harness.tool import box  # noqa: E402
from mini_harness.tool.box import TOOLS, ToolDefinition, find_tools  # noqa: E402
from mini_harness.trace import TRACE  # noqa: E402

WORK = Path('.experiments').resolve()
READ_TOOL = next(tool for tool in TOOLS if tool.name == 'read_file')
BASH_TOOL = next(tool for tool in TOOLS if tool.name == 'run_bash')


# --------------------------------------------------------------------------- helpers


def call(name, call_id=None, **arguments):
    return SimpleNamespace(id=call_id or f'call_{name}', function=SimpleNamespace(
        name=name, arguments=json.dumps(arguments)))


def subagent_call(index):
    return call('run_subagent', call_id=f'sub{index}', task_description=f'task {index}',
                prompt='explore the workspace', agent_type='explore_agent')


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


def experiment_subagents(repeats, count, latencies):
    """Serial vs concurrent wall time for a batch of subagent calls.

    A subagent spends its time blocked on its own model calls, which is what the
    injected latency models here.
    """
    workdir = WORK / 'subagents'
    workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for latency in latencies:
        def delayed_subagent(args, cfg=None, _delay=latency):
            if _delay:
                time.sleep(_delay)
            return 'summary'

        tool = ToolDefinition('run_subagent', 'timed subagent', box.RunSubAgentInput,
                              delayed_subagent, True)
        row = {'latency_ms': latency * 1000, 'calls': count}
        for label, parallel in (('serial', False), ('parallel', True)):
            cfg = config_for(workdir, parallel_tools=parallel, max_parallel_tools=count)
            execu = executor_with(cfg, tool)
            batch = [subagent_call(i) for i in range(count)]
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
    """The cost of recording every event, measured in pairs.

    The first version of this experiment timed one condition and then the other
    and subtracted the medians, which measured whatever the machine was doing in
    between: at 804 events it reported a *negative* overhead of 252 ms, because
    the second block ran while the first block's work was still being written
    back. Recording the two conditions alternately, in the same repeat, and
    taking the median of the paired differences removes that drift.
    """
    workdir = WORK / 'trace'
    write_tree(workdir, 1)
    rows = []
    for length in turns:
        read = call('read_file', file_path='tree/f000.py')
        script = [completion(message('', [read])) for _ in range(length)]
        script.append(completion(message('done')))
        off_times, on_times, deltas = [], [], []
        events = 0
        size = 0

        def one(label, enabled):
            nonlocal events, size
            trace_file = workdir / f'trace-{label}.jsonl'
            trace_file.unlink(missing_ok=True)
            # The turn limit has to allow the whole script, or the long run
            # silently stops at the default and measures the wrong length.
            cfg = config_for(workdir, max_turns_main = length + 2,
                             trace_path=str(trace_file) if enabled else None)
            TRACE.configure(None)
            start = time.perf_counter()
            run_scripted(cfg, list(script))
            elapsed = time.perf_counter() - start
            if trace_file.exists():
                size = trace_file.stat().st_size
                events = len([line for line in trace_file.read_text(encoding='utf-8').splitlines()
                              if line.strip()])
            return elapsed

        for index in range(repeats):
            # Alternate which condition goes first, so a drift that favours one
            # order cancels instead of accumulating.
            order = (('off', False), ('on', True)) if index % 2 == 0 else (('on', True), ('off', False))
            pair = {label: one(label, enabled) for label, enabled in order}
            off_times.append(pair['off'])
            on_times.append(pair['on'])
            deltas.append(pair['on'] - pair['off'])
        TRACE.configure(None)
        for label, samples in (('off', off_times), ('on', on_times)):
            rows.append({'trace': label, 'turns': length, 'events': events if label == 'on' else 0,
                         'median_ms': statistics.median(samples) * 1000,
                         'min_ms': min(samples) * 1000, 'max_ms': max(samples) * 1000,
                         'bytes': size if label == 'on' else 0})
        delta = statistics.median(deltas) * 1000
        rows.append({'trace': 'overhead', 'turns': length, 'events': events, 'bytes': size,
                     'delta_ms': delta, 'min_delta_ms': min(deltas) * 1000,
                     'max_delta_ms': max(deltas) * 1000, 'pairs': len(deltas),
                     'per_event_us': (delta * 1000 / events) if events else 0.0})
    return rows


RECALL_FACTS = [
    ('deploy window', 'the deploy window is 02:00 to 04:00 UTC on weekdays'),
    ('service token name', 'the service token is stored in SERVICE_TOKEN'),
    ('listening port', 'the service listens on port 8443'),
    ('retry budget', 'the retry budget is five attempts per request'),
    ('database host', 'the database host is db.internal'),
]
FILLER = ('config cache worker queue schema index buffer handler parser session timeout '
          'retry logger metric deploy rollout cluster shard replica').split()


def _distractors(count):
    """Deterministic filler text, so the measurement is reproducible.

    Each line carries its index: without it the filler repeats every 15 lines
    and the "800 distractor" row would really be a 15-document archive.
    """
    return [f'{FILLER[i % len(FILLER)]} {FILLER[(i * 7) % len(FILLER)]} '
            f'{FILLER[(i * 13) % len(FILLER)]} entry {i}' for i in range(count)]


class CountingHash(HashEmbedder):
    """The offline embedder, recording how many texts each call carried."""

    def __init__(self):
        super().__init__()
        self.sizes = []
        return

    def _embed_many(self, texts):
        self.sizes.append(len(texts))
        return super()._embed_many(texts)


def _provider_calls(sizes, batch):
    """Provider round trips for those batch sizes, at the embedder's batch size."""
    return sum(math.ceil(size / batch) for size in sizes if size)


def experiment_recall(sizes):
    """Does recall rank a planted detail first as the archive grows?"""
    workdir = WORK / 'recall'
    workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in sizes:
        path = workdir / f'journal-{size}.jsonl'
        removed = [{'role': 'tool', 'content': text} for text in _distractors(size)]
        removed.extend({'role': 'user', 'content': fact} for _, fact in RECALL_FACTS)
        path.write_text(json.dumps({'ts': 1.0, 'removed': removed}) + '\n', encoding='utf-8')
        memory = Memory(path)

        top1 = top3 = 0
        samples = []
        for query, fact in RECALL_FACTS:
            start = time.perf_counter()
            hits = memory.search(query, limit=3)
            samples.append(time.perf_counter() - start)
            if hits and hits[0].text == fact:
                top1 += 1
            if any(hit.text == fact for hit in hits):
                top3 += 1
        rows.append({
            'distractors': size,
            'entries': len(removed),
            'queries': len(RECALL_FACTS),
            'top1': top1,
            'top3': top3,
            'search_ms': statistics.median(samples) * 1000,
        })
    return rows


# A dozen plausible bridged tools, then deterministic filler. The point is the
# shape of an external surface -- many tools with unrelated descriptions -- not
# any particular server.
EXTERNAL_TOOLS = [
    ('web_search', 'search the web and return the top results for a query'),
    ('sql_query', 'run a SQL query against the postgres database'),
    ('browser_open', 'open a web page in a headless browser and read the text'),
    ('email_send', 'send an email message to a recipient'),
    ('slack_post', 'post a message to a slack channel'),
    ('jira_create', 'create a jira issue in a project'),
    ('pdf_render', 'render a PDF document to images'),
    ('chart_plot', 'draw a chart from numeric data'),
    ('vector_search', 'search a vector index for similar documents'),
    ('s3_upload', 'upload a file to an s3 bucket'),
    ('k8s_apply', 'apply a kubernetes manifest to a cluster'),
    ('stripe_charge', 'charge a credit card through stripe'),
]


class ExternalInput(BaseModel):
    """Stand-in for the arguments model an MCP tool is given."""

    model_config = ConfigDict(extra='forbid')
    text: str = ''


def _external_tool(name, description):
    def run(args: ExternalInput, cfg=None) -> str:
        return 'ok'

    return ToolDefinition(name, description, ExternalInput, run, False)


def external_surface(count):
    tools = [_external_tool(name, description) for name, description in EXTERNAL_TOOLS]
    while len(tools) < count:
        index = len(tools)
        tools.append(_external_tool(f'plugin_tool_{index:02d}',
                                    f'plugin capability number {index} for {FILLER[index % len(FILLER)]}'))
    return tools[:count]


def _payload_chars(definitions):
    """What the request carries: the schemas, serialised the way they are sent."""
    return len(json.dumps(box._to_api_tool(definitions)))


def experiment_exposure(external, budgets, task):
    """Schema payload of the tool surface, with the budget off and on.

    ``chars / 4`` is the usual rough token estimate; the file reports characters
    so the arithmetic stays checkable rather than presented as a measurement.
    """
    SELECTION.reset()
    definitions = list(TOOLS) + external_surface(external)
    pinned = tuple(tool.name for tool in definitions if tool.name in CORE_TOOLS)
    rows = []
    for budget in budgets:
        chosen = select(task, definitions, always=pinned, budget=budget)
        rows.append({
            'budget': budget,
            'external': external,
            'exposed': len(chosen),
            'hidden': len(definitions) - len(chosen),
            'schema_chars': _payload_chars(chosen),
            'est_tokens': round(_payload_chars(chosen) / 4),
        })
    baseline = rows[0]['schema_chars']
    for row in rows:
        row['vs_no_budget'] = round(row['schema_chars'] / baseline, 4)
    return rows


def experiment_recovery(external, budget, task, want):
    """A hidden tool has to be reachable again through find_tools.

    Returns one row, in a list, so it renders like the other experiments.
    """
    SELECTION.reset()
    definitions = list(TOOLS) + external_surface(external)
    SELECTION.register(definitions)
    pinned = tuple(tool.name for tool in definitions if tool.name in CORE_TOOLS)
    before = [tool.name for tool in select(task, definitions, always=pinned, budget=budget)]
    query = next(tool.description for tool in definitions if tool.name == want)
    find_tools(box.FindToolsInput(query=query, limit=1))
    after = [tool.name for tool in select(task, definitions, always=pinned, budget=budget)]
    return [{
        'task': task,
        'budget': budget,
        'external': external,
        'wanted': want,
        'query': query,
        'exposed_before': before,
        'exposed_after': after,
        'hidden_before': want not in before,
        'recovered': want in after and want not in before,
    }]


def experiment_vector_recall(sizes, batch = 96):
    """What the vector backend costs, and what the offline stand-in cannot say.

    The embedder here is the deterministic hash one, so the ranking *quality* of
    a real embeddings model is not measured by this at all. What is measured is
    the plumbing: how many texts the archive costs on the first query, that a
    repeated query costs nothing, and what the cosine scan adds to a search.
    """
    workdir = WORK / 'vector'
    workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in sizes:
        path = workdir / f'journal-{size}.jsonl'
        removed = [{'role': 'tool', 'content': text} for text in _distractors(size)]
        removed.extend({'role': 'user', 'content': fact} for _, fact in RECALL_FACTS)
        path.write_text(json.dumps({'ts': 1.0, 'removed': removed}) + '\n', encoding='utf-8')
        memory = Memory(path)

        embedder = CountingHash()
        query = RECALL_FACTS[0][0]
        start = time.perf_counter()
        memory.search(query, limit=3, embedder=embedder, mode='vector')
        cold_ms = (time.perf_counter() - start) * 1000
        archive_texts = sum(embedder.sizes)
        archive_calls = _provider_calls(embedder.sizes, batch)

        embedder.sizes.clear()
        start = time.perf_counter()
        memory.search(query, limit=3, embedder=embedder, mode='vector')
        warm_ms = (time.perf_counter() - start) * 1000
        repeat_texts = sum(embedder.sizes)
        repeat_calls = _provider_calls(embedder.sizes, batch)

        vector_ms, lexical_ms, hybrid_ms = [], [], []
        top1 = {'lexical': 0, 'vector': 0, 'hybrid': 0}
        for probe, fact in RECALL_FACTS:
            for mode, samples in (('vector', vector_ms), ('hybrid', hybrid_ms)):
                start = time.perf_counter()
                hits = memory.search(probe, limit=3, embedder=embedder, mode=mode)
                samples.append((time.perf_counter() - start) * 1000)
                if hits and hits[0].text == fact:
                    top1[mode] += 1
            start = time.perf_counter()
            hits = memory.search(probe, limit=3, mode='lexical')
            lexical_ms.append((time.perf_counter() - start) * 1000)
            if hits and hits[0].text == fact:
                top1['lexical'] += 1

        rows.append({
            'distractors': size,
            'entries': len(removed),
            'archive_texts': archive_texts,
            'archive_provider_calls': archive_calls,
            'cold_query_ms': cold_ms,
            'repeat_texts': repeat_texts,
            'repeat_provider_calls': repeat_calls,
            'warm_query_ms': warm_ms,
            'lexical_ms': statistics.median(lexical_ms),
            'vector_ms': statistics.median(vector_ms),
            'hybrid_ms': statistics.median(hybrid_ms),
            'queries': len(RECALL_FACTS),
            'top1_lexical': top1['lexical'],
            'top1_vector': top1['vector'],
            'top1_hybrid': top1['hybrid'],
        })
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
    lines += ['', '## Subagent concurrency', '',
              'Batch of run_subagent calls through ToolExecution.execute_batch.',
              '', '| injected latency | calls | serial (ms) | concurrent (ms) | speedup |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for row in results['subagents']:
        lines.append(f"| {row['latency_ms']:.0f} ms | {row['calls']} | "
                     f"{row['serial']['median_ms']:.2f} | {row['parallel']['median_ms']:.2f} | "
                     f"{row['speedup']:.2f}x |")
    lines += ['', '## Event trace', '',
              '| trace | turns | events | median (ms) | min (ms) | max (ms) | bytes |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results['trace']:
        if 'median_ms' in row:
            lines.append(f"| {row['trace']} | {row['turns']} | {row['events']} | "
                         f"{row['median_ms']:.2f} | {row['min_ms']:.2f} | "
                         f"{row['max_ms']:.2f} | {row['bytes']} |")
        else:
            lines.append(f"| {row['trace']} | {row['turns']} | {row['events']} | "
                         f"{row['delta_ms']:+.2f} (delta) | | | {row['bytes']} |")
    lines += ['', 'Per-event cost, from the paired difference within each repeat:', '',
              '| turns | events | overhead (ms) | min | max | per event (us) |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results['trace']:
        if 'per_event_us' in row:
            lines.append(f"| {row['turns']} | {row['events']} | {row['delta_ms']:+.2f} | "
                         f"{row['min_delta_ms']:+.2f} | {row['max_delta_ms']:+.2f} | "
                         f"{row['per_event_us']:.1f} |")
    lines += ['', '## Verification loop', '',
              '| scenario | turns | mutations | nudges | verified | outcome |',
              '| --- | ---: | ---: | ---: | --- | --- |']
    for row in results['verify']:
        lines.append(f"| {row['scenario']} | {row['turns']} | {row['mutations']} | "
                     f"{row['nudges']} | {row['verified']} | {row['outcome']} |")
    lines += ['', '## Retrievable memory', '',
              'Planted facts searched for among deterministic distractors.',
              '', '| distractors | archived | top-1 | top-3 | search (median) |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for row in results['recall']:
        lines.append(f"| {row['distractors']} | {row['entries']} | "
                     f"{row['top1']}/{row['queries']} | {row['top3']}/{row['queries']} | "
                     f"{row['search_ms']:.2f} ms |")
    lines += ['', '## Vector retrieval cost', '',
              'The archive embedded with the offline stand-in, at a batch size of 96. '
              'Ranking quality is not measured here: the stand-in has no semantics.',
              '', '| distractors | archived | archive texts | provider calls | cold query (ms) | '
              'repeat texts | repeat calls | warm query (ms) |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results['vector']:
        lines.append(f"| {row['distractors']} | {row['entries']} | {row['archive_texts']} | "
                     f"{row['archive_provider_calls']} | {row['cold_query_ms']:.2f} | "
                     f"{row['repeat_texts']} | {row['repeat_provider_calls']} | {row['warm_query_ms']:.2f} |")
    lines += ['', '| distractors | lexical (ms) | vector (ms) | hybrid (ms) | top-1 lexical | '
              'top-1 vector | top-1 hybrid |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results['vector']:
        lines.append(f"| {row['distractors']} | {row['lexical_ms']:.2f} | {row['vector_ms']:.2f} | "
                     f"{row['hybrid_ms']:.2f} | {row['top1_lexical']}/{row['queries']} | "
                     f"{row['top1_vector']}/{row['queries']} | {row['top1_hybrid']}/{row['queries']} |")
    lines += ['', '## Dispatch policy', '', '| call | ok | tag |', '| --- | --- | --- |']
    for row in results['policy']:
        lines.append(f"| {row['call']} | {row['ok']} | {row['tag']} |")
    lines += ['', '## Tool exposure', '',
              'Serialised schemas for the built-in tools plus a bridged surface of N '
              'tools, budget 0 meaning "expose everything".',
              '', '| bridged tools | budget | exposed | hidden | schema chars | est. tokens | vs no budget |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results['exposure']:
        lines.append(f"| {row['external']} | {row['budget']} | {row['exposed']} | {row['hidden']} | "
                     f"{row['schema_chars']} | {row['est_tokens']} | {row['vs_no_budget']:.2f}x |")
    lines += ['', '## Hidden tool recovery', '',
              'A tool the budget hid, asked for by its own description through find_tools.', '',
              '| wanted | hidden before | budget | exposed before | recovered |',
              '| --- | --- | ---: | ---: | --- |']
    for row in results['recovery']:
        lines.append(f"| {row['wanted']} | {'yes' if row['hidden_before'] else 'no'} | {row['budget']} | "
                     f"{len(row['exposed_before'])} | {'yes' if row['recovered'] else 'no'} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='EXPERIMENTS.json', help='where to write the raw results')
    parser.add_argument('--repeats', type=int, default=7, help='timed runs per configuration')
    parser.add_argument('--calls', type=int, default=6, help='calls per read batch')
    parser.add_argument('--subagents', type=int, default=4, help='subagents per batch')
    parser.add_argument('--turns', type=int, default=20,
                        help='tool turns in the short trace run; a second run is ten times longer')
    args = parser.parse_args()

    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    payload = {
        'python': sys.version.split()[0],
        'platform': sys.platform,
        'repeats': args.repeats,
        'parallel': experiment_parallel(args.repeats, args.calls, (0.0, 0.005, 0.02)),
        'subagents': experiment_subagents(args.repeats, args.subagents, (0.02, 0.1)),
        'trace': experiment_trace(args.repeats, (args.turns, args.turns * 10)),
        'verify': experiment_verify(args.repeats),
        'recall': experiment_recall((50, 200, 800)),
        'vector': experiment_vector_recall((50, 200, 800)),
        'policy': experiment_policy(),
        'exposure': experiment_exposure(24, (0, 8, 12), 'fix the failing test in the parser'),
        'recovery': experiment_recovery(24, 8, 'fix the failing test in the parser', 'vector_search'),
    }
    report = render(payload)
    print(report)
    Path(args.out).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'raw results written to {args.out}')
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
