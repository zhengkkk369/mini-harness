import os
import fnmatch

from pathlib import Path
from dataclasses import dataclass, field
from dotenv import find_dotenv, load_dotenv

from mini_harness.bench_profile import BENCH_OVERRIDE

load_dotenv(find_dotenv(usecwd=True))

def normalize_no_proxy() -> str|None:
    """Rewrite bracketed IPv6 entries in NO_PROXY into a form httpx can parse.

    ``[::1]`` is a legitimate way to write an IPv6 loopback in NO_PROXY, but
    httpx 0.28 builds a URLPattern from every entry and raises
    ``InvalidURL: Invalid port: ':1]'`` on the bracketed form. Because that
    happens while the client is being constructed, a correctly configured
    machine cannot reach any API at all. Strip the brackets and leave the rest
    of the list alone.
    """
    raw = os.environ.get('NO_PROXY') or os.environ.get('no_proxy')
    if not raw:
        return None
    fixed = raw.replace('[::1]', '::1').replace('[::]', '::')
    if fixed != raw:
        os.environ['NO_PROXY'] = fixed
        os.environ['no_proxy'] = fixed
    return fixed

normalize_no_proxy()

def default_workspace() -> Path:
    env = os.environ.get("MINI_HARNESS_WORK_SPACE")
    return Path(env).resolve() if env else Path.cwd()

TRUE_WORDS = frozenset({'1', 'true', 'yes', 'on'})
FALSE_WORDS = frozenset({'0', 'false', 'no', 'off'})

def env_flag(value: str, name: str) -> bool:
    token = value.strip().lower()
    if token in TRUE_WORDS:
        return True
    if token in FALSE_WORDS:
        return False
    raise ValueError(f'{name} must be one of {sorted(TRUE_WORDS | FALSE_WORDS)}, got {value!r}')

def env_list(value: str) -> tuple:
    return tuple(part.strip() for part in value.split(',') if part.strip())

@dataclass(frozen = True)
class Config:
    # default_factory, not a direct call: a direct call would freeze the
    # workspace at import time and ignore the environment passed to
    # build_config().
    work_space: Path = field(default_factory = default_workspace)
    profile: str = 'local'
    provider: str = 'deepseek'
    api_key_env: str = 'DEEPSEEK_API_KEY'
    reasoning_effort: str|None = None

    # 'deepseek-v4-flash' is the legacy spelling of this model. The provider
    # still accepts it and serves the same weights at the same price, but the
    # current name is deepseek-flash.
    model_main: str = 'deepseek-flash'
    model_sub: str = 'deepseek-flash'
    base_url: str = 'https://api.deepseek.com'
    max_turns_main: int = 50
    max_turns_sub: int = 20
    max_tokens_main: int = 100000
    max_tokens_sub: int = 50000
    temp_set: float = 0.5
    think_main: str = 'enabled'
    think_sub: str = 'enabled'

    max_read_size: int = 512000
    max_hits: int = 200
    bash_timeout: int = 90
    read_limit: int = 60000
    bash_limit: int = 30000
    guard_read: bool = True
    guard_write: bool = True
    session_path: str|None = None
    trace_path: str|None = None
    clip_limit: int = 80000
    track_files: bool = True
    edit_require_read: bool = True
    write_require_read: bool = True
    thrash_notice: int = 0
    diff_echo_lines: int = 40

    # Run budgets. wall_budget is seconds, token_budget counts prompt plus
    # completion tokens, cost_budget is US dollars. Cost is only tracked when
    # both prices are set.
    wall_budget: float|None = None
    token_budget: int|None = None
    cost_budget: float|None = None
    price_in: float|None = None
    price_out: float|None = None
    price_cache_in: float|None = None

    # Tool execution. A batch runs concurrently only when every call is
    # side-effect free, so read-only batches are the only ones that overlap.
    parallel_tools: bool = True
    max_parallel_tools: int = 4

    # Dispatch policy, applied before the gate and before approval.
    policy_deny_tools: tuple = ()
    policy_deny_patterns: tuple = ()
    read_only: bool = False

    # Verification. When required, finishing with unverified edits costs one
    # extra turn asking the agent to run something first.
    verify_required: bool = True
    verify_nudges: int = 1

    # Tool exposure. Zero sends every tool on every request; a positive budget
    # ranks the tools against the task and keeps the best of them, with the
    # built-ins and anything find_tools pulled in always present.
    tool_budget: int = 0

    # Retrievable memory. Compaction is lossy, so what it removes stays
    # searchable through the recall tool.
    recall_enabled: bool = True
    recall_limit: int = 5
    recall_snippet: int = 400

    # MCP servers to bridge tools from, each written as 'name=command line'.
    # Remote tools default to risky, because a server can do anything.
    mcp_servers: tuple = ()
    mcp_timeout: float = 30.0
    mcp_risky: bool = True

    max_retry: int = 5
    retry_base: float = 2.0
    rate_retry: int = 6
    rate_base: float = 30.0
    rate_cap: float = 120.0

    compact_limit: int = 300000
    recent_keep: int = 20

    AGREE: frozenset = frozenset(
        {
            'yes', 'y', 'ok', 'sure'
        }
    )
    deny_name: tuple = (
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "id_rsa*",
            "id_ed25519*",
            "id_ecdsa*",
            ".netrc",
            ".npmrc",
            ".pypirc",
            "*credential*",
            "*secret*",
        )
    deny_dir: frozenset = frozenset({".ssh", ".aws", ".gnupg"})
    bash_env_deny: tuple = (
        '*KEY*',
        '*TOKEN*',
        '*SECRET*',
        '*PASSWORD*',
        '*CREDENTIAL*',
        '*_PWD',
        '*AUTH*'
    )
    system_prompt: str = """
Role: You are Mini Harness, my coding agent.

Style: Careful, precise, evidence-driven. Verify rather than assume.

--- Environment ---

E1. A human is watching this session and can answer you. When the task is ambiguous,
    when several readings are reasonable, or when an action is destructive and you
    are unsure, ask before acting. A short question now is cheaper than undoing the
    wrong work later.

E2. Every run_bash call starts a fresh process. Working directory, environment
    variables, and shell state do NOT persist between calls.
        Correct:  cd sandbox && python test.py
        Wrong:    cd sandbox   ... then a separate call ...   python test.py
    The same applies to export, source, and virtualenv activation. Chain them into
    one command. Commands already start in the workspace root, so you do not need to
    cd there.

E3. Background processes must redirect all output or run_bash will block until it
    times out.
        Correct:  nohup ./server > /dev/null 2>&1 &
        Wrong:    ./server &

E4. Prefer non-interactive flags. A command waiting for input will hang until the
    timeout. Use -y / --yes / --non-interactive.

E5. Paths are relative to the workspace root.
      - Reading is limited to the workspace. Sensitive files (.env, *.key, *.pem,
        credentials, .ssh/) are refused, and are skipped by glob_file and grep_file.
      - Writing is limited to ./sandbox. Write to "sandbox/xxx.py", not "xxx.py".
    The tools enforce these, not you. A PermissionError means you stepped outside.

E6. Tools that can change things need my approval before each call and may be denied:
    run_bash, run_sandbox, run_subagent, and anything bridged in from an MCP server.
    If denied, do not retry the same call. Say what you needed it for and propose an
    alternative.

E7. - Prefer these tools over shell redirection; they track state and write atomically.
    The file tools remember what you have read in this turn.
    - edit_file requires that you have already read the file and that it has not
      changed since. If a build step, a script, or another tool modified it, read
      it again.
    - write_file creates new files. Replacing an existing file requires having read
      it end to end plus overwrite=true.
    - When a call is refused for either reason the current content of the file is
      returned with the refusal. Read it and repeat the call.
    - Line numbers in tool output are display only. Never put them in old_string or
      new_string.

E8. Prefer run_sandbox to execute generated code: a fresh Python 3.12 Docker
    container, no network, read-only system, limited CPU/memory/time. It starts at
    /workspace, which maps to the host sandbox/ directory. Use "python demo.py"
    there for the host file "sandbox/demo.py". Only files in sandbox/ persist.
    Docker and the python:3.12-slim image must already be installed. If unavailable,
    explain the setup needed. run_bash is a host shell, not an isolated sandbox.

--- Workflow ---

W1. Locate: use glob_file and grep_file to find the files that matter before opening
    anything.

W2. Understand: use read_file and run_bash to read the actual content. Never act on
    a guess about what a file contains.

W3. Change: use edit_file for targeted edits, write_file for new files.

W4. Verify: run the code, run the tests, inspect the output.

--- Discipline ---

D1. Before a batch of tool calls, say in one or two sentences what you are about to
    do. Not the full reasoning -- just the intent, so I can stop you early if you are
    heading the wrong way.

D2. Finish with verification. Before you stop, run whatever proves the work is done.
    If you cannot verify something, state plainly what remains unverified.

D3. Write files with write_file and edit_file, not with shell redirection. The file
    tools write atomically and respect the workspace limits; `echo > file` does not.

D4. Never modify, delete, or disable a test just to make it pass. If you believe the
    test itself is wrong, say so and ask before touching it.

D5. Do not make unrequested changes. Fix what was asked and leave working code alone.
    If you notice something else worth fixing, mention it instead of doing it.

D6. Do not use emoji.

--- Tools ---

O1. run_todo: use it for any task with more than two steps, and keep it updated as
    you go. I use it to follow your progress. Skip it for simple questions.

O2. run_subagent (explore_agent, coding_agent, planning_agent): use it when a subtask
    is genuinely separable. A subagent spends its own turns and returns only a
    summary, so it is not free.

O3. recall: once context compaction has replaced earlier turns with a summary, use
    recall(query) to search the removed text instead of guessing or re-reading
    files. It is the only way back to a detail the summary dropped, such as an
    exact path, value, command or error message.

O4. find_tools: when the tool you would reach for is not in the current list, call
    find_tools(query) with what you want to do. The matching tools become callable
    from the next turn. Do not claim a capability is missing before trying this.
    """

    @property
    def sandbox_dir(self) -> Path:
        return self.work_space/'sandbox'

    @property
    def thinking_main(self) -> dict:
        if self.provider == 'openai':
            return {}
        return {
            'thinking': {
                'type': self.think_main
            }
        }

    @property
    def thinking_sub(self) -> dict:
        if self.provider == 'openai':
            return {}
        return {
            'thinking': {
                'type': self.think_sub
            }
        }

    @property
    def api_key(self) -> str|None:
        return os.environ.get('MINI_HARNESS_API_KEY') or os.environ.get(self.api_key_env)

    def request_options(self, sub: bool = False) -> dict:
        options = {'model': self.model_sub if sub else self.model_main}
        limit = self.max_tokens_sub if sub else self.max_tokens_main
        if self.provider == 'openai':
            options['max_completion_tokens'] = limit
            if self.reasoning_effort:
                options['reasoning_effort'] = self.reasoning_effort
        else:
            options.update(
                max_tokens=limit,
                temperature=self.temp_set,
                extra_body=self.thinking_sub if sub else self.thinking_main,
            )
        return options

    def request_messages(self, messages: list) -> list:
        if self.provider != 'openai':
            return messages
        return [
            {key: value for key, value in message.items()
             if key not in {'reasoning_content', 'annotations', 'parsed'}}
            for message in messages
        ]

    @property
    def bash_env(self) -> dict:
        secrets = {'MINI_HARNESS_API_KEY', 'OPENAI_API_KEY', 'DEEPSEEK_API_KEY',
                   self.api_key_env.upper()}
        return {
            key: value for key, value in os.environ.items()
            if key.upper() not in secrets
            and not any(fnmatch.fnmatch(key.upper(), name) for name in self.bash_env_deny)
        }


def build_config() -> Config:
    model = os.environ.get('MINI_HARNESS_MODEL')
    prefix = model.partition('/')[0] if model else None
    inferred = prefix if prefix in {'openai', 'deepseek'} else (
        'deepseek' if not model or model.startswith('deepseek-') else 'openai'
    )
    provider = os.environ.get('MINI_HARNESS_PROVIDER', inferred).lower()
    if provider not in {'openai', 'deepseek'}:
        raise ValueError(f'Unsupported provider: {provider}')
    if prefix in {'openai', 'deepseek'}:
        if prefix != provider:
            raise ValueError('Model prefix conflicts with MINI_HARNESS_PROVIDER')
        model = model.partition('/')[2]
    overrides = dict(BENCH_OVERRIDE) if os.environ.get('MINI_HARNESS_PROFILE') == 'bench' else {}
    overrides.update(provider=provider, api_key_env=os.environ.get(
        'MINI_HARNESS_API_KEY_ENV', f'{provider.upper()}_API_KEY'))
    if provider == 'openai':
        overrides.update(
            model_main='gpt-4.1', model_sub='gpt-4.1',
            base_url=os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
            max_tokens_main=8192, max_tokens_sub=8192, compact_limit=64000,
            think_main='default', think_sub='default',
        )
    if model is not None:
        if not model.strip():
            raise ValueError('MINI_HARNESS_MODEL must not be empty')
        overrides.update(model_main=model, model_sub=model)
    for env_name, field_name in {
        'MINI_HARNESS_BASE_URL': 'base_url',
        'MINI_HARNESS_SUB_MODEL': 'model_sub',
        'MINI_HARNESS_REASONING_EFFORT': 'reasoning_effort',
    }.items():
        if value := os.environ.get(env_name):
            overrides[field_name] = value
    if workspace := os.environ.get('MINI_HARNESS_WORK_SPACE'):
        overrides['work_space'] = Path(workspace).resolve()
    if overrides.get('reasoning_effort') and provider == 'openai':
        overrides.update(think_main=overrides['reasoning_effort'],
                         think_sub=overrides['reasoning_effort'])
    for name in ('max_tokens_main', 'max_tokens_sub', 'compact_limit', 'token_budget'):
        if value := os.environ.get(f'MINI_HARNESS_{name.upper()}'):
            if int(value) <= 0:
                raise ValueError(f'{name} must be positive')
            overrides[name] = int(value)
    for name in ('wall_budget', 'cost_budget', 'price_in', 'price_out', 'price_cache_in'):
        if value := os.environ.get(f'MINI_HARNESS_{name.upper()}'):
            if float(value) <= 0:
                raise ValueError(f'{name} must be positive')
            overrides[name] = float(value)
    if trace := os.environ.get('MINI_HARNESS_TRACE'):
        overrides['trace_path'] = trace
    for name in ('max_parallel_tools', 'verify_nudges', 'recall_limit', 'recall_snippet',
                 'tool_budget'):
        if value := os.environ.get(f'MINI_HARNESS_{name.upper()}'):
            if int(value) <= 0:
                raise ValueError(f'{name} must be positive')
            overrides[name] = int(value)
    for variable, name in (('MINI_HARNESS_PARALLEL_TOOLS', 'parallel_tools'),
                           ('MINI_HARNESS_READ_ONLY', 'read_only'),
                           ('MINI_HARNESS_VERIFY_REQUIRED', 'verify_required'),
                           ('MINI_HARNESS_RECALL', 'recall_enabled'),
                           ('MINI_HARNESS_MCP_RISKY', 'mcp_risky')):
        if value := os.environ.get(variable):
            overrides[name] = env_flag(value, variable)
    if timeout := os.environ.get('MINI_HARNESS_MCP_TIMEOUT'):
        if float(timeout) <= 0:
            raise ValueError('mcp_timeout must be positive')
        overrides['mcp_timeout'] = float(timeout)
    if servers := os.environ.get('MINI_HARNESS_MCP_SERVERS'):
        # A command may contain spaces, so entries are separated by semicolons.
        overrides['mcp_servers'] = tuple(part.strip() for part in servers.split(';') if part.strip())
    for field_name, variable in (('policy_deny_tools', 'MINI_HARNESS_DENY_TOOLS'),
                                 ('policy_deny_patterns', 'MINI_HARNESS_DENY_PATTERNS')):
        if value := os.environ.get(variable):
            overrides[field_name] = env_list(value)
    sub_model = overrides.get('model_sub')
    if sub_model and sub_model.partition('/')[0] in {'openai', 'deepseek'}:
        prefix, _, name = sub_model.partition('/')
        if prefix != provider or not name.strip():
            raise ValueError('Sub-model must use the same provider as the main model')
        overrides['model_sub'] = name
    return Config(**overrides)

CONFIG = build_config()
if not CONFIG.api_key:
    raise RuntimeError(f'[api error]: set {CONFIG.api_key_env} or MINI_HARNESS_API_KEY')
