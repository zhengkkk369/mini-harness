import os
import json
import re
import time

from openai import OpenAI, LengthFinishReasonError
from pathlib import Path
from dataclasses import dataclass, asdict, field

from mini_harness.config import CONFIG
from mini_harness.budget import ACCOUNT, Budget, STOP_WALL, cached_tokens, SOURCE_MAIN
from mini_harness.trace import TRACE
from mini_harness.retry_request import retry_call
from mini_harness.compact import COMPACT
from mini_harness.tool.box import _ask_human, _always_allow, _edited_path, ToolExecution, _to_api_tool, log_tool, WRITE_TOOLS
from mini_harness.history import atomic_write
from mini_harness.selector import SELECTION, select
from mini_harness.skills import load as load_skills
from mini_harness.tool.tag import OUTCOME, MARK
from mini_harness.tool.block import CLIP

# Always exposed, whatever the budget: the escape hatch itself, the tools the
# locate/understand workflow starts from, and the one the skills index in the
# prompt tells the agent to call. Everything else -- the write, shell and
# delegation tools, and anything bridged in from MCP -- has to earn its place or
# be pulled back in by find_tools.
CORE_TOOLS = frozenset({'find_tools', 'read_file', 'grep_file', 'glob_file', 'skills'})

VERIFY_TOOLS = {'run_bash', 'run_sandbox'}
VERIFY_NUDGE = (
    'You changed files but have not run anything since. Before finishing, run the code or the '
    'tests that cover your change and show the result. If you cannot verify it, say plainly what '
    'remains unverified.'
)
# Commands that check a project rather than a file. Running the suite is the most
# thorough verification there is, and it need not name the file that changed, so
# it counts on its own. The list is a heuristic and deliberately short.
VERIFY_RUNNERS = ('pytest', 'unittest', 'tox', 'nox', 'make test', 'npm test', 'npm run test',
                  'cargo test', 'go test', 'gradle test', 'mvn test')

def normalise_path(text) -> str:
    return str(text).replace('\\', '/').strip().lower()

def _command_of(tool_call) -> str:
    """The shell command a verification call ran, when it has one."""
    try:
        arguments = json.loads(tool_call.function.arguments or '{}')
    except (TypeError, ValueError):
        return ''
    command = arguments.get('command') if isinstance(arguments, dict) else None
    return command if isinstance(command, str) else ''

def verification_touches(command: str, output: str, changed) -> str:
    """What one verification run had to do with the files that changed.

    ``targeted`` when the command or its output names a changed file, its file
    name, or its stem (so ``python test_parse.py`` counts for ``parse.py``).
    ``suite`` when it invokes a test runner, which covers everything by
    definition. ``unrelated`` otherwise -- running *something* is not the same as
    running something that could fail because of the change.

    This is a heuristic, not a proof: ``cat f.py`` also "touches" the file. It is
    strictly stronger than the rule it replaces, which any command satisfied.
    """
    haystack = normalise_path(f'{command}\n{output}')
    for path in changed or ():
        full = normalise_path(path)
        name = full.rsplit('/', 1)[-1]
        stem = name.rsplit('.', 1)[0]
        if full and full in haystack:
            return 'targeted'
        if name and name in haystack:
            return 'targeted'
        # A stem has to be long enough to be a word, and has to start one, so
        # "f" does not match half the output and "test_parse" still matches.
        if len(stem) >= 4 and re.search(rf'(?<![a-z0-9]){re.escape(stem)}', haystack):
            return 'targeted'
    if any(runner in haystack for runner in VERIFY_RUNNERS):
        return 'suite'
    return 'unrelated'

def verify_nudge(changed) -> str:
    """The nudge, naming the files when the run knows them."""
    files = sorted({normalise_path(path) for path in changed or () if normalise_path(path)})
    if not files:
        return VERIFY_NUDGE
    shown = ', '.join(files[:4])
    if len(files) > 4:
        shown += f' (and {len(files) - 4} more)'
    return (
        f'You changed {shown} but have not run anything since. Before finishing, run the code or '
        f'the tests that cover the change and show the result. If you cannot verify it, say '
        f'plainly what remains unverified.'
    )

@dataclass(frozen = True)
class Result:
    outcome: str
    calls: int
    turns: int
    ok: int
    failed_by_tag: dict
    calls_by_tool: dict
    last_prompt: int
    prompt_total: int
    completion_total: int
    wall: float
    err: str = ''
    cost: float = 0.0
    stopped_by: str = ''
    verified: bool = True
    mutations: int = 0
    model_calls: int = 0
    usage_by_source: dict = field(default_factory = dict)
    # Prompt tokens the provider served from its cache, and what the same usage
    # would have cost billed without that discount. Both are needed to quote a
    # cost honestly: the raw `cost` alone hides a 4x difference on a cache-heavy
    # run, and this is the number an unconfigured price would have produced.
    cached_total: int = 0
    cost_naive: float = 0.0
    # What the last verification had to do with the change, and what is still
    # unaccounted for. `verified` is the summary of these two.
    verification: str = 'none'
    unverified_files: tuple = ()


class DeepSeekAgent:
    def __init__(self, tools: list, cfg = CONFIG) -> None:
        self.definitions = list(tools)
        self.tools = _to_api_tool(tools)
        self.regis = {t.name: t for t in tools}
        self.pinned = tuple(t.name for t in tools if t.name in CORE_TOOLS)
        self.task = ''
        self._exposed = tuple(t.name for t in tools)
        self.last_prompt_tokens = 0
        self.printed = ''
        self.skills = load_skills(cfg.skills_path, enabled = cfg.skills_enabled)
        self.system = [
            {
                'role': 'system',
                'content': cfg.system_prompt + self._skills_block()
            }
        ]
        self.message = list(self.system)
        self.session_memory = cfg.session_path if cfg.session_path else cfg.work_space/'session.json'
        self.last_usage = None
        self.last_reasoning = ''
        return

    def _skills_block(self) -> str:
        """The skills index, appended to the system prompt.

        Names and descriptions only: enough for the agent to know what exists,
        without paying for the procedures whether or not they are used. An empty
        directory, or skills turned off, appends nothing at all.
        """
        index = self.skills.index_text()
        if not index:
            return ''
        return (
            '\n\n--- Skills ---\n\n'
            'Written procedures for kinds of work in this workspace. The name and description\n'
            'are here; call skills(action="load", name=...) to read the steps before starting\n'
            'that kind of task.\n\n'
            f'{index}\n'
        )

    def _expose(self, cfg = CONFIG) -> list:
        """The tool list for the next request, honouring the exposure budget.

        Rebuilt only when the exposed set changes, because generating a JSON
        Schema for every tool on every turn is not free.
        """
        chosen = select(self.task, self.definitions, always = self.pinned,
                        budget = cfg.tool_budget)
        names = tuple(definition.name for definition in chosen)
        if names != self._exposed:
            self._exposed = names
            self.tools = _to_api_tool(chosen)
        return self.tools

    def _load_memory(self, cfg = CONFIG) -> list:
        if Path(self.session_memory).exists():
            ans = input('\nFound the memory, want to continue the conversation?(yes/no) -> ').strip().lower()
            if ans in cfg.AGREE:
                try:
                    data = json.loads(Path(self.session_memory).read_text(errors = 'replace', encoding = 'utf-8'))
                except (OSError, json.JSONDecodeError) as e:
                    print(f'[load error]: cannot load memory: {e}, start a new conversation')
                    return self.message
                self.message = data
                return self.message
        return self.message

    def _save_memory(self, quiet: bool = False,  cfg = CONFIG) -> None:
        try:
            data = json.dumps(self.message, indent = 2, ensure_ascii=False)
            atomic_write(Path(self.session_memory), data)
        except (TypeError, ValueError, OSError) as e:
            print(f'[save failed]: cannot save the memory: {e}')
            return

        if not quiet:
            print(f'[save successfully!]')
        return

    def _request_agent(self, client: OpenAI, cfg = CONFIG):
        buffer = ''
        reasoning = ''
        streaming = False
        truncated = None
        with client.chat.completions.stream(
            tools = self.tools,
            messages = cfg.request_messages(self.message),
            **cfg.request_options(),
            stream_options = {'include_usage': True}
        ) as e:
            for d in e:
                if d.type == 'content.delta':
                    if streaming:
                        print(d.delta, end = '', flush = True)
                        self.printed += d.delta
                        continue

                    buffer += d.delta
                    n = min(len(buffer), len(self.printed))
                    if buffer[:n] != self.printed[:n]:
                        print('\n---- [connection lost, the text above is useless] ----')
                        print(buffer, end = '', flush = True)
                        self.printed = buffer
                        streaming = True
                    elif len(buffer) > len(self.printed):
                        print(buffer[len(self.printed):], end = '', flush = True)
                        self.printed = buffer
                        streaming = True
                if d.type == 'chunk':
                    u = getattr(d.chunk, 'usage', None)
                    if u is not None:
                        self.last_usage = u
                    if d.chunk.choices:
                        res = getattr(d.chunk.choices[0].delta, 'reasoning_content', None)
                        if res:
                            reasoning += res
                            print(res, end = '', flush = True)
                if d.type == 'tool_calls.function.arguments.delta':
                    pass

            try:
                response = e.get_final_completion()
                response.choices[0].message.reasoning_content = reasoning
            except LengthFinishReasonError as ex:
                truncated = buffer
                comp = getattr(ex, 'completion', None)
                u = getattr(comp, 'usage', None) if comp is not None else None
                if u is not None:
                    self.last_usage = u
                response = None
        if not streaming and self.printed:
            print(f'\n--- [reconnection lost, the context above is useless] ----')
            if buffer:
                print(buffer, end = '', flush = True)
            self.printed = buffer
        self.last_reasoning = reasoning
        return response, truncated

    def _fill_interrupted(self, tool_calls, cfg = CONFIG) -> None:
        self._fill_skipped(tool_calls, 'the run was interrupted before this call ran', cfg = cfg)
        return

    def _fill_skipped(self, tool_calls, reason: str, cfg = CONFIG) -> None:
        """Answer every call that will never run, so the conversation stays valid.

        The chat API rejects an assistant message whose tool calls have no
        results, so a run stopped mid-batch has to close the pairing before it
        can end -- and before the session can be resumed.
        """
        done = {t['tool_call_id'] for t in self.message if t['role'] == 'tool'}
        for tool_call in tool_calls:
            if tool_call.id not in done:
                self.message.append(
                    {'role': 'tool', 'tool_call_id': tool_call.id,
                     'content': f'[{tool_call.function.name} skipped]: {reason}'}
                )
        return

    def _run_turn(self, client: OpenAI, executer: ToolExecution, cfg = CONFIG) -> Result:
        start = time.time()
        outcome = OUTCOME.ERROR
        turns = calls = ok = last_prompt = 0
        err = ''
        stopped_by = ''
        mutated = unverified = nudges = unrelated = 0
        verification = 'none'
        changed = set()
        verified = True
        by_tag = {}
        by_tool = {}
        budget = Budget.from_config(cfg, started = start)
        # One run, one ledger. Everything that calls a model reports here, so the
        # token totals and the budget cover the summariser and the subagents too.
        ACCOUNT.reset().attach(budget)

        def record_usage(usage) -> int:
            """Fold one response's usage into the totals, the budget and the trace.

            Returns this response's cached input count, which is what a per-turn
            event should carry; only run_end reports the cumulative figure.
            """
            nonlocal last_prompt
            prompt = getattr(usage, 'prompt_tokens', 0) or 0
            self.last_prompt_tokens = prompt
            last_prompt = prompt
            return ACCOUNT.record(usage, source = SOURCE_MAIN)

        try:
            for turn in range(cfg.max_turns_main):
                spent = budget.exceeded()
                if spent is not None:
                    stopped_by = spent
                    outcome = OUTCOME.TIMEOUT if spent == STOP_WALL else OUTCOME.BUDGET
                    print(f'[{spent} budget]: agent stopped at {budget.render()} (limits: {budget.limits()})')
                    TRACE.emit('budget_stop', reason = spent, tokens = budget.tokens,
                               cost = round(budget.cost, 6), elapsed = round(budget.elapsed, 4))
                    break
                turns += 1
                self.printed = ''
                self.tools = self._expose(cfg)
                executer.hidden = {definition.name for definition in self.definitions} - set(self._exposed)
                TRACE.emit('turn', turn = turns, tools = len(self.tools))
                if self.last_prompt_tokens >= cfg.compact_limit:
                    self.message = COMPACT.compact_content(client, self.message, self.session_memory, cfg = cfg)
                    self._save_memory(quiet=True)
                    self.last_prompt_tokens = 0
                response, truncated = retry_call(lambda: self._request_agent(client, cfg = cfg), cfg = cfg)
                if truncated is not None:
                    note = ('\n\n[Your previous response was cut off at the output token limit. Be more concise, or take an action instead of continuing to reason.]')
                    self.message.append({
                        'role': 'assistant', 'content': truncated + note,
                        'reasoning_content': self.last_reasoning
                    })
                    cached = record_usage(self.last_usage) if self.last_usage is not None else 0
                    out_tokens = getattr(self.last_usage, 'completion_tokens', 0) or 0
                    print(f'[ctx]: {last_prompt} / {cfg.compact_limit} tokens, out {out_tokens} [TRUNCATED]')
                    TRACE.emit('usage', turn = turns, prompt = last_prompt, completion = out_tokens,
                               total = budget.tokens, cached = cached,
                               cost = round(budget.cost, 6), truncated = True)
                    self._save_memory(quiet=True)
                    continue
                if response.usage:
                    cached = record_usage(response.usage)
                    print(f'[ctx]: {last_prompt} / {cfg.compact_limit} tokens, out {response.usage.completion_tokens}')
                    TRACE.emit('usage', turn = turns, prompt = response.usage.prompt_tokens,
                               completion = response.usage.completion_tokens, total = budget.tokens,
                               cached = cached, cost = round(budget.cost, 6),
                               truncated = False)
                message = response.choices[0].message
                if message.tool_calls:
                    print()
                    d = message.model_dump(exclude_none = True)
                    self.message.append(d)
                    # The turn began before the request; a long request can spend
                    # the wall budget on its own. Check again before the batch, so
                    # a spent budget costs one request rather than a request plus
                    # a whole batch of tool calls.
                    spent = budget.exceeded()
                    if spent is not None:
                        stopped_by = spent
                        outcome = OUTCOME.TIMEOUT if spent == STOP_WALL else OUTCOME.BUDGET
                        self._fill_skipped(message.tool_calls,
                                           f'{spent} budget was spent before this batch could run',
                                           cfg = cfg)
                        print(f'[{spent} budget]: agent stopped at {budget.render()} '
                              f'(limits: {budget.limits()}), {len(message.tool_calls)} call(s) skipped')
                        TRACE.emit('budget_stop', reason = spent, tokens = budget.tokens,
                                   cost = round(budget.cost, 6), elapsed = round(budget.elapsed, 4),
                                   skipped = len(message.tool_calls))
                        self._save_memory(quiet=True)
                        break
                    try:
                        results = executer.execute_batch(message.tool_calls, cfg = cfg)
                        for tool_call, res in zip(message.tool_calls, results):
                            used = tool_call.function.name
                            calls += 1
                            by_tool[used] = by_tool.get(used, 0) + 1
                            content = CLIP.clip(res.content)
                            if res.ok:
                                ok += 1
                                if used in WRITE_TOOLS:
                                    mutated += 1
                                    unverified += 1
                                    target = _edited_path(tool_call)
                                    if target:
                                        changed.add(target)
                                elif used in VERIFY_TOOLS:
                                    verdict = verification_touches(
                                        _command_of(tool_call), content, changed)
                                    if verdict == 'unrelated' and cfg.verify_targets_changed:
                                        # Something ran, but nothing that could
                                        # fail because of the change.
                                        unrelated += 1
                                    else:
                                        unverified = 0
                                        changed.clear()
                                    verification = verdict
                                if used == 'run_todo':
                                    print(f'\n-=-=-=-=-= Todo List -=-=-=-=-=\n{res.content}')
                            else:
                                by_tag[res.tag] = by_tag.get(res.tag, 0) + 1
                            self.message.append(
                                {'role': 'tool', 'tool_call_id': tool_call.id, 'content': content}
                            ) 
                            log_tool(tool_call, res, cfg = cfg)
                    except KeyboardInterrupt:
                        self._fill_interrupted(message.tool_calls, cfg = cfg)
                        raise
                    self._save_memory(quiet=True)
                else:
                    if cfg.verify_required and unverified > 0 and nudges < cfg.verify_nudges:
                        # The prompt asks for verification; this is what makes it
                        # happen, at the cost of one bounded extra turn.
                        nudges += 1
                        self.message.append(message.model_dump(exclude_none=True))
                        self.message.append({'role': 'user', 'content': verify_nudge(changed)})
                        print(f'\n[verify]: {unverified} edit(s) since the last covering run, '
                              f'asking the agent to verify')
                        TRACE.emit('verify_nudge', unverified = unverified, nudge = nudges,
                                   mutations = mutated, turn = turns,
                                   files = sorted(changed), unrelated = unrelated,
                                   verification = verification)
                        self._save_memory(quiet=True)
                        continue
                    outcome = OUTCOME.COMPLETED
                    verified = unverified == 0
                    self.message.append(
                        message.model_dump(exclude_none=True)
                    )
                    self._save_memory(quiet=True)
                    break
            else:
                outcome = OUTCOME.EXHAUSTED
                self._save_memory(quiet=True)
                print(f'[agent done]: the agnet run out of the turn for the actions')
        except KeyboardInterrupt:
            outcome = OUTCOME.INTERRUPTED
            print(f'[intterupted]: the action was intterupted')
            self._save_memory(quiet=True)
        except Exception as e:
            outcome = OUTCOME.ERROR
            err = f'{type(e).__name__}:{e}'
            print(f'[run failed]: agent run failed: {err}')
            self._save_memory(quiet=True)
        # Every exit path reports the same truth: a run that ended with an edit
        # nothing has exercised is not verified, whether it finished, ran out of
        # turns, stopped on a budget or failed outright.
        verified = unverified == 0
        result = Result(
            outcome = outcome,
            calls = calls,
            turns = turns,
            ok = ok,
            last_prompt=last_prompt,
            prompt_total = ACCOUNT.prompt_total,
            completion_total=ACCOUNT.completion_total,
            failed_by_tag=by_tag,
            calls_by_tool=by_tool,
            wall = time.time() -start,
            err = err,
            cost = budget.cost,
            stopped_by = stopped_by,
            verified = verified,
            mutations = mutated,
            model_calls = ACCOUNT.calls,
            usage_by_source = ACCOUNT.by_source,
            cached_total = ACCOUNT.cached_total,
            cost_naive = budget.uncached_cost,
            verification = verification,
            unverified_files = tuple(sorted(changed))
        )
        TRACE.emit('run_end', outcome = outcome, turns = turns, calls = calls, ok = ok,
                   failed_by_tag = by_tag, calls_by_tool = by_tool,
                   prompt_tokens = ACCOUNT.prompt_total,
                   completion_tokens = ACCOUNT.completion_total,
                   cached_tokens = ACCOUNT.cached_total,
                   cost = round(budget.cost, 6), stopped_by = stopped_by,
                   verified = verified, mutations = mutated, nudges = nudges,
                   wall = round(result.wall, 4), err = err,
                   model_calls = ACCOUNT.calls, usage_by_source = ACCOUNT.by_source,
                   verification = verification, unverified_files = sorted(changed))
        ACCOUNT.detach()
        return result

    def run_task(self, task: str, cfg = CONFIG) -> Result:
        SELECTION.register(self.definitions)
        try:
            return self._run_task(task, cfg = cfg)
        finally:
            # The catalogue is a frame, so a later agent in this process starts
            # from its own tools rather than inheriting this run's pins.
            SELECTION.release()

    def _run_task(self, task: str, cfg = CONFIG) -> Result:
        self.task = task
        TRACE.configure(cfg.trace_path)
        TRACE.emit('run_start', mode = 'task', profile = cfg.profile, model = cfg.model_main,
                   task = task, turns_limit = cfg.max_turns_main,
                   limits = Budget.from_config(cfg).limits())
        self.message = list(self.system)
        client = OpenAI(
            api_key = cfg.api_key,
            base_url = cfg.base_url,
            max_retries = 0
        ) 
        executer = ToolExecution(self.regis, _always_allow, cfg = cfg)
        self.message.append(
            {'role': 'user', 'content': task}
        )
        result = self._run_turn(client, executer, cfg = cfg)
        self._save_memory(quiet=True)
        TRACE.close()
        return result

    def dump_run(self, result: Result, task: str, path: str|None = None, cfg = CONFIG) -> None:
        payload = {
            'profile': cfg.profile,
            'model': cfg.model_main,
            'thinking': cfg.think_main,
            'task': task,
            'max_turns_main': cfg.max_turns_main,
            **asdict(result)
        }
        print(f'{MARK}{json.dumps(payload, ensure_ascii=False)}')
        if path:
            try:
                p = Path(path)
                atomic_write(p, json.dumps(payload, indent=2, ensure_ascii=False))
            except OSError as e:
                print(f'[dump failed]: {e}')
        return
                                            
    def run(self, cfg = CONFIG) -> None:
        SELECTION.register(self.definitions)
        TRACE.configure(cfg.trace_path)
        self.message = self._load_memory(cfg = cfg)
        client = OpenAI(
            api_key = cfg.api_key,
            base_url = cfg.base_url,
            max_retries = 0
        )
        # Every exit from the loop -- quit, EOF, a double interrupt -- closes the
        # trace here, so a REPL session does not leave the file handle open (and
        # on Windows, locked) for the rest of the process's life. The tool
        # catalogue is a frame for the same reason: it ends with the session.
        try:
            self._repl(client, cfg = cfg)
        finally:
            TRACE.close()
            SELECTION.release()
        return

    def _repl(self, client: OpenAI, cfg = CONFIG) -> None:
        pending_exits = False
        while True:
            try:
                user_input = input("\nPlease enter your command -> ")
            except KeyboardInterrupt:
                if not pending_exits:
                    print(f'Please click ctrl+c again to exit the program ')
                    pending_exits = True
                    continue
                print('\nBye!')
                self._save_memory(quiet = True)
                break
            except EOFError:
                print('\nBye!')
                self._save_memory(quiet = True)
                break

            if user_input.strip().lower() in ['quit', 'exit', 'bye']:
                self._save_memory()
                print('\nBye!')
                break

            self.message.append(
                {'role': 'user', 'content': user_input}
            )
            self.task = user_input
            pending_exits = False
            TRACE.emit('run_start', mode = 'repl', profile = cfg.profile, model = cfg.model_main,
                       task = user_input, turns_limit = cfg.max_turns_main,
                       limits = Budget.from_config(cfg).limits())
            executer = ToolExecution(self.regis, _ask_human, cfg = cfg)
            result = self._run_turn(client, executer, cfg = cfg)
            self._save_memory(quiet=True)
            print(
                f'[outcome]: {result.outcome}, [calls]: {result.calls}, [turns]: {result.turns}, [ok]: {result.ok}, '
                f'[calls_tool]: {result.calls_by_tool}, [failed]: {result.failed_by_tag}, '
                f'[last_prompt]: {result.last_prompt}, [last_prompt_tokens]: {result.prompt_total}, [completion_tokens]: {result.completion_total}, '
                f'[cost]: ${result.cost:.4f}, [stopped_by]: {result.stopped_by or "none"}, '
                f'[verified]: {result.verified}, [mutations]: {result.mutations}, '
                f'[wall]: {result.wall:.1f}s, '
                f'[err]: {result.err}'
            )
        return
                
