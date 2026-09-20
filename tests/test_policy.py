"""Dispatch policy: deny rules and read-only mode."""

import pytest

from mini_harness.policy import ALLOW, DENY, Decision, Policy
from mini_harness.tool import box
from mini_harness.tool.tag import TAG
from tests.conftest import REGISTRY, call, write


def make_policy(**overrides):
    fields = {
        'deny_tools': frozenset(),
        'deny_patterns': (),
        'read_only': False,
        'mutating_tools': frozenset(),
    }
    fields.update(overrides)
    return Policy(**fields)


def runner(cfg, confirm=box._always_allow, policy=None):
    return box.ToolExecution(dict(REGISTRY), confirm, cfg=cfg, policy=policy)


# --------------------------------------------------------------------------- decide


def test_nothing_is_refused_by_default():
    decision = make_policy().decide(call('run_bash', command='ls'))

    assert decision.decision == ALLOW
    assert decision.denied is False
    assert decision.reason == ''


def test_a_denied_tool_names_itself_in_the_reason():
    decision = make_policy(deny_tools=frozenset({'run_sandbox'})).decide(call('run_sandbox', command='ls'))

    assert decision.denied is True
    assert 'run_sandbox' in decision.reason


def test_read_only_refuses_mutating_tools_only():
    policy = make_policy(read_only=True, mutating_tools=frozenset({'edit_file'}))

    assert policy.decide(call('edit_file', file_path='sandbox/a.py', old_string='a', new_string='b')).denied
    assert not policy.decide(call('read_file', file_path='a.py')).denied


def test_a_pattern_matches_the_command_not_the_json():
    policy = make_policy(deny_patterns=('rm -rf*',))

    assert policy.decide(call('run_bash', command='rm -rf /')).denied
    assert not policy.decide(call('run_bash', command='ls -la')).denied


def test_a_pattern_falls_back_to_the_arguments_without_a_command():
    policy = make_policy(deny_patterns=('*sandbox/prod*',))

    assert policy.decide(call('write_file', file_path='sandbox/prod/cfg', content='x')).denied
    assert not policy.decide(call('write_file', file_path='sandbox/dev/cfg', content='x')).denied


def test_the_first_matching_pattern_is_reported():
    policy = make_policy(deny_patterns=('ls*', 'never*'))
    assert 'ls*' in policy.decide(call('run_bash', command='ls -la')).reason


def test_a_denied_tool_beats_a_harmless_command():
    policy = make_policy(deny_tools=frozenset({'run_bash'}), deny_patterns=('never*',))
    assert policy.decide(call('run_bash', command='ls')).denied


def test_unparseable_arguments_are_matched_as_raw_text():
    broken = call('run_bash', command='ls')
    broken.function.arguments = 'not json'

    assert not make_policy(deny_patterns=('ls*',)).decide(broken).denied
    assert make_policy(deny_patterns=('not*',)).decide(broken).denied


def test_from_config_reads_every_policy_field(cfg_factory):
    cfg = cfg_factory(policy_deny_tools=('run_sandbox',), policy_deny_patterns=('rm -rf*',),
                      read_only=True)

    policy = Policy.from_config(cfg, mutating_tools=frozenset({'edit_file'}))

    assert policy.decide(call('run_sandbox', command='ls')).denied
    assert policy.decide(call('run_bash', command='rm -rf /')).denied
    assert policy.decide(call('edit_file', file_path='sandbox/a', old_string='a', new_string='b')).denied
    assert not policy.decide(call('read_file', file_path='a.py')).denied


# --------------------------------------------------------------------------- dispatch


def test_the_executor_refuses_a_policy_denied_call(cfg_factory):
    cfg = cfg_factory(policy_deny_patterns=('rm -rf*',))
    result = runner(cfg).execute_tool(call('run_bash', command='rm -rf /'), cfg=cfg)

    assert result.ok is False
    assert result.tag == TAG.POLICY_DENIED
    assert 'policy' in result.content


def test_a_policy_denial_never_reaches_the_human(cfg_factory):
    asked = []

    def confirm(tool_call, cfg=None):
        asked.append(tool_call.function.name)
        return True

    cfg = cfg_factory(policy_deny_tools=('run_bash',))
    result = runner(cfg, confirm=confirm).execute_tool(call('run_bash', command='ls'), cfg=cfg)

    assert result.tag == TAG.POLICY_DENIED
    assert asked == []


def test_policy_denial_is_counted_under_its_own_tag(cfg_factory, workspace):
    """A policy refusal must be distinguishable from a user refusal."""
    cfg = cfg_factory(policy_deny_tools=('run_bash',))
    result = runner(cfg).execute_tool(call('run_bash', command='ls'), cfg=cfg)

    assert result.tag != TAG.DENIED
    assert result.tag == TAG.POLICY_DENIED


def test_read_only_keeps_reads_and_blocks_writes(cfg_factory, workspace):
    write(workspace / 'sandbox' / 'f.py', 'alpha\n')
    cfg = cfg_factory(read_only=True)
    execu = runner(cfg)

    allowed = execu.execute_tool(call('read_file', file_path='sandbox/f.py'), cfg=cfg)
    blocked = execu.execute_tool(
        call('edit_file', file_path='sandbox/f.py', old_string='alpha', new_string='b'), cfg=cfg)

    assert allowed.ok is True
    assert blocked.tag == TAG.POLICY_DENIED
    assert (workspace / 'sandbox' / 'f.py').read_text(encoding='utf-8') == 'alpha\n'


def test_read_only_blocks_the_shell_and_subagents(cfg_factory):
    cfg = cfg_factory(read_only=True)
    execu = runner(cfg)

    for tool_call in (call('run_bash', command='ls'),
                      call('run_sandbox', command='ls'),
                      call('run_subagent', task_description='t', prompt='p', agent_type='explore_agent')):
        assert execu.execute_tool(tool_call, cfg=cfg).tag == TAG.POLICY_DENIED


def test_a_policy_can_be_injected_directly(cfg):
    execu = runner(cfg, policy=Policy(deny_tools=frozenset({'read_file'})))
    assert execu.execute_tool(call('read_file', file_path='anything'), cfg=cfg).tag == TAG.POLICY_DENIED


def test_deny_list_tools_are_read_from_the_config(cfg_factory):
    cfg = cfg_factory(policy_deny_tools=('glob_file',))
    assert runner(cfg).execute_tool(call('glob_file', pattern='*.py'), cfg=cfg).tag == TAG.POLICY_DENIED
