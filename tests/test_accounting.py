"""One ledger per run: every model call is billed, whoever made it.

Before this existed the token totals, the cost and the token budget only ever
saw the main loop's turns. A run that used subagents or that compacted its
context spent real money that no number in the report accounted for, and a
token budget could not stop it. These tests hold the ledger to that promise.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from mini_harness.agent import DeepSeekAgent
from mini_harness.budget import ACCOUNT, Budget, SOURCE_COMPACT, SOURCE_MAIN, SOURCE_SUBAGENT
from mini_harness.compact import COMPACT
from mini_harness.tool import box
from mini_harness.tool.box import TOOLS
from tests.conftest import REGISTRY, call, executor, write
from tests.test_agent import FakeClient, Stream, completion, make_agent, model_message

PRICES = dict(price_in = 1.0, price_out = 2.0, price_cache_in = 0.5)


def usage(prompt = 100, completion = 20, cached = 0):
    return SimpleNamespace(prompt_tokens = prompt, completion_tokens = completion,
                           prompt_cache_hit_tokens = cached)


# ------------------------------------------------------------------ the ledger itself


def test_usage_without_a_response_is_ignored():
    assert ACCOUNT.record(None) == 0
    assert ACCOUNT.calls == 0
    assert ACCOUNT.prompt_total == 0


def test_a_call_is_counted_and_attributed_to_its_source():
    ACCOUNT.record(usage(prompt = 100, completion = 20), source = SOURCE_MAIN)
    ACCOUNT.record(usage(prompt = 300, completion = 60), source = f'{SOURCE_SUBAGENT}:explore_agent')

    assert ACCOUNT.prompt_total == 400
    assert ACCOUNT.completion_total == 80
    assert ACCOUNT.calls == 2
    assert ACCOUNT.by_source[SOURCE_MAIN] == {'calls': 1, 'prompt': 100, 'completion': 20}
    assert ACCOUNT.by_source[f'{SOURCE_SUBAGENT}:explore_agent'] == {
        'calls': 1, 'prompt': 300, 'completion': 60}


def test_the_recorded_cached_count_is_returned_for_the_turn_event():
    assert ACCOUNT.record(usage(prompt = 100, cached = 80)) == 80


def test_an_attached_budget_receives_every_source():
    budget = Budget(**PRICES)
    ACCOUNT.attach(budget)

    ACCOUNT.record(usage(prompt = 1_000_000, completion = 1_000_000))
    ACCOUNT.record(usage(prompt = 1_000_000, completion = 0, cached = 1_000_000),
                   source = SOURCE_COMPACT)

    # 1M uncached in + 1M out = $1 + $2; 1M cached in = $0.50
    assert budget.cost == pytest.approx(3.5)
    assert budget.tokens == 3_000_000


def test_exceeded_reports_the_attached_budget():
    assert ACCOUNT.exceeded() is None

    ACCOUNT.attach(Budget(token_budget = 10))
    ACCOUNT.record(usage(prompt = 20))

    assert ACCOUNT.exceeded() == 'tokens'


def test_detaching_stops_the_billing():
    budget = Budget(token_budget = 1)
    ACCOUNT.attach(budget)
    ACCOUNT.detach()
    ACCOUNT.record(usage(prompt = 100))

    assert budget.tokens == 0
    assert ACCOUNT.prompt_total == 100


def test_concurrent_records_are_not_lost():
    """A parallel subagent batch records from several threads at once."""
    budget = Budget()
    ACCOUNT.attach(budget)
    threads = 8
    per_thread = 250

    def work():
        for _ in range(per_thread):
            ACCOUNT.record(usage(prompt = 1, completion = 1))

    workers = [threading.Thread(target = work) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert ACCOUNT.calls == threads * per_thread
    assert ACCOUNT.prompt_total == threads * per_thread
    assert budget.prompt_tokens == threads * per_thread


# ------------------------------------------------------------------ a run bills what it spends

SUBAGENT_USAGE = SimpleNamespace(prompt_tokens = 700, completion_tokens = 90)


class SubagentClient:
    """Stands in for the client run_subagent builds for itself."""

    def __init__(self, usage = SUBAGENT_USAGE):
        self.usage = usage
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        message = model_message('the subagent answer')
        return SimpleNamespace(choices = [SimpleNamespace(message = message)], usage = self.usage)

    @property
    def chat(self):
        return SimpleNamespace(completions = SimpleNamespace(create = self._create))


def subagent_call(call_id = 'sub1'):
    return call('run_subagent', call_id = call_id, task_description = 'explore',
                prompt = 'look around', agent_type = 'explore_agent')


def test_a_subagents_tokens_reach_the_run_totals(cfg, workspace, monkeypatch):
    write(workspace / 'sandbox' / 'f.py', 'alpha\n')
    sub = SubagentClient()
    monkeypatch.setattr(box, 'OpenAI', lambda **_: sub)
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message('', [subagent_call()]))),
        Stream(completion(model_message('all done'))),
    )
    agent.session_memory = workspace / 'session.json'

    result = agent._run_turn(client, executor(cfg), cfg = cfg)

    # two main-loop responses (100/20 each) plus the subagent's (700/90)
    assert result.prompt_total == 100 + 100 + 700
    assert result.completion_total == 20 + 20 + 90
    assert result.model_calls == 3
    assert result.usage_by_source[SOURCE_MAIN]['calls'] == 2
    assert result.usage_by_source[f'{SOURCE_SUBAGENT}:explore_agent'] == {
        'calls': 1, 'prompt': 700, 'completion': 90}


def test_a_subagents_cost_reaches_the_run_cost(cfg_factory, workspace, monkeypatch):
    cfg = cfg_factory(**PRICES)
    write(workspace / 'sandbox' / 'f.py', 'alpha\n')
    monkeypatch.setattr(box, 'OpenAI', lambda **_: SubagentClient())
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message('', [subagent_call()]))),
        Stream(completion(model_message('all done'))),
    )
    agent.session_memory = workspace / 'session.json'

    result = agent._run_turn(client, executor(cfg), cfg = cfg)

    # 700 prompt at $1/M plus 90 output at $2/M = $0.00088 from the subagent alone
    assert result.cost >= 700 / 1_000_000 * 1.0 + 90 / 1_000_000 * 2.0


def test_the_run_end_trace_carries_the_breakdown(cfg, workspace, session_dir, monkeypatch):
    from mini_harness.trace import TRACE
    import json
    write(workspace / 'sandbox' / 'f.py', 'alpha\n')
    monkeypatch.setattr(box, 'OpenAI', lambda **_: SubagentClient())
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message('', [subagent_call()]))),
        Stream(completion(model_message('all done'))),
    )
    agent.session_memory = workspace / 'session.json'
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        agent._run_turn(client, executor(cfg), cfg = cfg)
    finally:
        TRACE.configure(None)

    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    end = [event for event in events if event['event'] == 'run_end'][0]

    assert end['model_calls'] == 3
    assert end['usage_by_source'][f'{SOURCE_SUBAGENT}:explore_agent']['prompt'] == 700
    assert end['prompt_tokens'] == 900


def test_a_subagent_stops_when_the_run_budget_is_spent(cfg, monkeypatch):
    """The parent's budget is the run's budget."""
    sub = SubagentClient()
    monkeypatch.setattr(box, 'OpenAI', lambda **_: sub)
    ACCOUNT.reset().attach(Budget(token_budget = 1))
    ACCOUNT.record(usage(prompt = 10))          # the main loop already spent it

    answer = box.run_subagent(box.RunSubAgentInput(task_description = 'explore',
                                                   prompt = 'look', agent_type = 'explore_agent'),
                              cfg = cfg)

    assert 'stopped' in answer
    assert 'tokens budget' in answer
    assert sub.calls == 0, 'the subagent should not have called the model at all'


def test_a_subagent_without_a_budget_still_runs(cfg, monkeypatch):
    """No attached run (a tool called outside the loop) must not break it."""
    sub = SubagentClient()
    monkeypatch.setattr(box, 'OpenAI', lambda **_: sub)

    answer = box.run_subagent(box.RunSubAgentInput(task_description = 'explore',
                                                   prompt = 'look', agent_type = 'explore_agent'),
                              cfg = cfg)

    assert answer == 'the subagent answer'
    assert sub.calls == 1
    assert ACCOUNT.prompt_total == 700


# ------------------------------------------------------------------ compaction is billed


class SummariserClient:
    """A compaction client whose response carries usage."""

    def __init__(self, usage = None):
        self.usage = usage or SimpleNamespace(prompt_tokens = 250, completion_tokens = 30)
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content = 'the summary')
        return SimpleNamespace(choices = [SimpleNamespace(message = message)], usage = self.usage)

    @property
    def chat(self):
        return SimpleNamespace(completions = SimpleNamespace(create = self._create))


def test_the_summariser_is_billed(cfg_factory):
    cfg = cfg_factory(recent_keep = 1)
    messages = [{'role': 'system', 'content': 's'}]
    for index in range(6):
        messages.append({'role': 'user', 'content': f'q{index}'})
        messages.append({'role': 'assistant', 'content': f'a{index}'})
    client = SummariserClient()

    COMPACT.compact_content(client, messages, cfg = cfg)

    assert client.calls == 1
    assert ACCOUNT.by_source[SOURCE_COMPACT] == {'calls': 1, 'prompt': 250, 'completion': 30}


def test_a_short_conversation_is_not_compacted(cfg_factory):
    """The gate that made this test's first version fail is real behaviour."""
    cfg = cfg_factory(compact_limit = 1, recent_keep = 20)
    messages = [{'role': 'system', 'content': 's'},
                {'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': 'a'}]
    client = SummariserClient()

    assert COMPACT.compact_content(client, messages, cfg = cfg) == messages
    assert client.calls == 0
    assert SOURCE_COMPACT not in ACCOUNT.by_source


def test_compaction_inside_a_run_increases_the_reported_totals(cfg_factory, workspace):
    """A two-turn run whose context crosses compact_limit between turns.

    recent_keep has to be smaller than the conversation, or the cut-point guard
    leaves the message list alone.
    """
    write(workspace / 'sandbox' / 'f.py', 'alpha\n')
    cfg = cfg_factory(compact_limit = 1, recent_keep = 1)
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message('', [call('read_file', file_path = 'sandbox/f.py')]))),
        Stream(completion(model_message('all done'))),
    )
    agent.session_memory = workspace / 'session.json'
    agent.message.append({'role': 'user', 'content': 'read the file'})

    result = agent._run_turn(client, executor(cfg), cfg = cfg)

    # two main responses (100/20 each) plus the summariser's own (100/20)
    assert result.usage_by_source[SOURCE_COMPACT]['calls'] == 1
    assert result.prompt_total == 100 + 100 + 100
    assert result.model_calls == 3
    assert result.outcome == 'completed'


# ------------------------------------------------------------------ the intra-turn gate


class SlowClient(FakeClient):
    """A client whose first response takes longer than the wall budget."""

    def __init__(self, script, delay):
        super().__init__(script)
        self.delay = delay

    def _stream(self, **kwargs):
        time.sleep(self.delay)
        return super()._stream(**kwargs)


def test_a_spent_wall_budget_skips_the_batch_instead_of_running_it(cfg_factory, workspace):
    write(workspace / 'sandbox' / 'f.py', 'alpha\n')
    cfg = cfg_factory(wall_budget = 0.05)
    agent = DeepSeekAgent(TOOLS, cfg = cfg)
    agent.session_memory = workspace / 'session.json'
    client = SlowClient([Stream(completion(model_message('', [call('read_file', file_path='sandbox/f.py')]))),
                         Stream(completion(model_message('never reached')))], delay = 0.2)
    agent.message.append({'role': 'user', 'content': 'read the file'})

    result = agent._run_turn(client, executor(cfg), cfg = cfg)

    assert result.stopped_by == 'wall'
    assert result.outcome == 'timeout'
    assert result.calls == 0, 'the batch must not have run'
    skipped = [m for m in agent.message if m.get('role') == 'tool']
    assert len(skipped) == 1
    assert 'skipped' in skipped[0]['content']


def test_the_skipped_call_keeps_the_conversation_valid(cfg_factory, workspace):
    """Every tool call needs a result, or the next request is rejected."""
    from mini_harness.history import paired_history, unanswered
    write(workspace / 'sandbox' / 'f.py', 'alpha\n')
    cfg = cfg_factory(wall_budget = 0.05)
    agent = DeepSeekAgent(TOOLS, cfg = cfg)
    agent.session_memory = workspace / 'session.json'
    client = SlowClient([Stream(completion(model_message('', [call('read_file', file_path='sandbox/f.py')]))),
                         Stream(completion(model_message('never reached')))], delay = 0.2)
    agent.message.append({'role': 'user', 'content': 'read the file'})

    agent._run_turn(client, executor(cfg), cfg = cfg)

    assert unanswered(agent.message) == []
    assert paired_history(agent.message) == agent.message


def test_an_interrupted_batch_still_reads_as_before(cfg, workspace):
    """The old placeholder wording is kept for the interrupt path."""
    agent = DeepSeekAgent(TOOLS, cfg = cfg)
    agent.message.append({'role': 'user', 'content': 'x'})

    agent._fill_interrupted([call('read_file', call_id = 'c1', file_path = 'a.py')], cfg = cfg)

    assert 'the run was interrupted before this call ran' in agent.message[-1]['content']
    assert agent.message[-1]['tool_call_id'] == 'c1'


def test_the_generic_helper_names_the_tool_and_the_reason(cfg):
    agent = DeepSeekAgent(TOOLS, cfg = cfg)

    agent._fill_skipped([call('run_bash', call_id = 'c2', command = 'ls')], 'because', cfg = cfg)

    assert agent.message[-1]['content'] == '[run_bash skipped]: because'


def test_a_result_without_the_new_fields_still_builds():
    """Result is constructed positionally elsewhere; new fields must default."""
    from mini_harness.agent import Result

    result = Result('completed', 1, 1, 1, {}, {}, 5, 5, 5, 0.5)

    assert result.model_calls == 0
    assert result.usage_by_source == {}


def test_the_registry_helper_is_unchanged():
    """Guard the import surface the new tests rely on."""
    assert 'run_subagent' in REGISTRY
