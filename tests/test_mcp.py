"""The MCP bridge: handshake, tool discovery, dispatch and failure handling.

Everything here runs against tests/fake_mcp_server.py, a stdio JSON-RPC stub, so
the suite stays offline and deterministic.
"""

import json
import shlex
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_harness import mcp
from mini_harness.mcp import (
    MCPBridge, MCPClient, MCPError, model_from_schema, parse_servers, tool_name,
)
from mini_harness.tool import box
from mini_harness.trace import TRACE
from tests.conftest import REGISTRY, call

STUB = Path(__file__).resolve().parent / 'fake_mcp_server.py'


def stub_spec(*flags, name='stub'):
    return f'{name}={shlex.join([sys.executable, str(STUB), *flags])}'


@pytest.fixture
def bridge_of(cfg_factory):
    """Build and start a bridge, and always shut its servers down."""
    bridges = []

    def build(*specs, **overrides):
        cfg = cfg_factory(mcp_servers=tuple(specs), **overrides)
        bridge = MCPBridge(cfg=cfg).start()
        bridges.append(bridge)
        return bridge

    yield build
    for bridge in bridges:
        bridge.close()


def client_of(cfg, *flags):
    return MCPClient('stub', [sys.executable, str(STUB), *flags], timeout=cfg.mcp_timeout)


def registry_with(definitions):
    return {**REGISTRY, **{definition.name: definition for definition in definitions}}


# --------------------------------------------------------------------------- specs


def test_a_spec_is_a_name_and_a_command():
    assert parse_servers(['fs=python -m server --root /tmp']) == [
        ('fs', ['python', '-m', 'server', '--root', '/tmp'])]


def test_no_specs_means_no_servers():
    assert parse_servers(()) == []
    assert parse_servers(None) == []


@pytest.mark.parametrize('spec', ['no equals sign', '=python -c pass', 'name=', '', 7])
def test_a_malformed_spec_is_rejected(spec):
    with pytest.raises(ValueError, match='name=command'):
        parse_servers([spec])


def test_unbalanced_quotes_are_reported():
    with pytest.raises(ValueError, match='cannot parse the command'):
        parse_servers(['bad=python "unclosed'])


# --------------------------------------------------------------------------- schema conversion


def test_required_properties_have_no_default():
    model = model_from_schema('M', {'type': 'object',
                                    'properties': {'a': {'type': 'string'}},
                                    'required': ['a']})
    with pytest.raises(ValidationError):
        model.model_validate({})
    assert model.model_validate({'a': 'x'}).a == 'x'


def test_optional_properties_default_to_none():
    model = model_from_schema('M', {'type': 'object', 'properties': {'a': {'type': 'string'}}})
    assert model.model_validate({}).a is None


@pytest.mark.parametrize('declared,value', [
    ('string', 'x'), ('integer', 3), ('number', 1.5), ('boolean', True),
    ('array', [1, 2]), ('object', {'k': 1}),
])
def test_declared_types_are_mapped(declared, value):
    model = model_from_schema('M', {'type': 'object', 'properties': {'v': {'type': declared}},
                                    'required': ['v']})
    assert model.model_validate({'v': value}).v == value


def test_an_unknown_type_still_validates():
    model = model_from_schema('M', {'type': 'object', 'properties': {'v': {'type': 'exotic'}},
                                    'required': ['v']})
    assert model.model_validate({'v': object}).v is object


def test_a_schema_without_properties_yields_an_empty_model():
    model = model_from_schema('M', {'type': 'object'})
    assert model.model_validate({}) is not None


def test_undeclared_arguments_are_allowed():
    """A converted schema may not describe everything the server accepts."""
    model = model_from_schema('M', {'type': 'object', 'properties': {}})
    assert model.model_validate({'surprise': 1}).surprise == 1


# --------------------------------------------------------------------------- naming


def test_bridged_names_say_which_server_they_came_from():
    assert tool_name('fs', 'read_file') == 'fs__read_file'


def test_bridged_names_are_registry_safe():
    assert tool_name('my server', 'no schema/allowed') == 'my_server__no_schema_allowed'


def test_bridged_names_are_truncated_to_the_api_limit():
    assert len(tool_name('s' * 80, 't' * 80)) == 64


# --------------------------------------------------------------------------- handshake


def test_a_client_starts_and_lists_tools(cfg):
    with client_of(cfg) as client:
        tools = client.list_tools()
    assert [tool['name'] for tool in tools] == [
        'echo', 'add', 'optional', 'fail', 'slow', 'no schema']
    assert client.server_info == {'name': 'stub', 'version': '1'}
    assert client.pages == 1


def test_a_paginated_tool_list_is_assembled(cfg):
    """Ignoring nextCursor would silently show a subset of the tools."""
    with client_of(cfg, '--paginate=2') as client:
        tools = client.list_tools()

    assert [tool['name'] for tool in tools] == [
        'echo', 'add', 'optional', 'fail', 'slow', 'no schema']
    assert client.pages == 3


def test_an_endless_cursor_is_bounded(cfg):
    with client_of(cfg, '--paginate-loop') as client:
        tools = client.list_tools()

    assert len(tools) == mcp.MAX_PAGES
    assert any('cursor' in note for note in client.noise)


def test_the_bridge_registers_tools_from_every_page(bridge_of):
    bridge = bridge_of(stub_spec('--paginate=2'))
    assert len(bridge.definitions) == 6


def test_the_trace_records_how_many_pages_were_read(cfg_factory, session_dir):
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        cfg = cfg_factory(mcp_servers=(stub_spec('--paginate=2'),))
        with MCPBridge(cfg=cfg):
            pass
    finally:
        TRACE.configure(None)

    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    server = next(event for event in events if event['event'] == 'mcp_server')
    assert server['pages'] == 3
    assert len(server['tools']) == 6


def test_the_command_is_resolved_through_path(monkeypatch):
    """On Windows npx is npx.CMD, and CreateProcess will not find the bare name."""
    seen = {}

    def fake_popen(command, **kwargs):
        seen['command'] = command
        raise OSError('stop here')

    monkeypatch.setattr(mcp.shutil, 'which',
                        lambda name: 'C:/tools/npx.CMD' if name == 'npx' else None)
    monkeypatch.setattr(mcp.subprocess, 'Popen', fake_popen)

    client = MCPClient('stub', ['npx', '-y', 'some-server'])
    with pytest.raises(OSError):
        client.start()

    assert seen['command'] == ['C:/tools/npx.CMD', '-y', 'some-server']


def test_an_unresolvable_command_is_left_alone(monkeypatch):
    seen = {}

    def fake_popen(command, **kwargs):
        seen['command'] = command
        raise FileNotFoundError('no such command')

    monkeypatch.setattr(mcp.shutil, 'which', lambda name: None)
    monkeypatch.setattr(mcp.subprocess, 'Popen', fake_popen)

    with pytest.raises(FileNotFoundError):
        MCPClient('stub', ['definitely-not-here']).start()

    assert seen['command'] == ['definitely-not-here']


def test_a_refused_handshake_is_an_error(cfg):
    with pytest.raises(MCPError, match='initialize failed'):
        client_of(cfg, '--bad-handshake').start()


def test_a_silent_server_times_out(cfg_factory):
    cfg = cfg_factory(mcp_timeout=0.5)
    client = client_of(cfg, '--silent')
    try:
        with pytest.raises(MCPError, match='timed out'):
            client.start()
    finally:
        client.close()


def test_noise_and_notifications_do_not_break_the_link(cfg):
    with client_of(cfg, '--noisy') as client:
        tools = client.list_tools()

    assert [tool['name'] for tool in tools][0] == 'echo'
    assert client.noise and 'not json' in client.noise[0]
    assert any(n.get('method') == 'notifications/message' for n in client.notifications)


def test_a_server_request_is_answered_with_an_error(cfg):
    """Otherwise a server that asks us something would wait forever."""
    with client_of(cfg, '--noisy') as client:
        assert client.list_tools()          # still usable after the server's request


def test_invalid_utf8_from_a_server_does_not_kill_the_link(cfg):
    """Regression: text mode decoded with the locale codec died on one bad byte."""
    with client_of(cfg, '--binary') as client:
        assert [tool['name'] for tool in client.list_tools()][0] == 'echo'
        assert client.call_tool('echo', {'text': 'still here'}) == 'echo: still here'


def test_a_dead_server_unblocks_a_waiting_call(cfg_factory):
    """Regression: the reader died and every later call sat out its full timeout."""
    cfg = cfg_factory(mcp_timeout=30.0)
    client = client_of(cfg).start()
    try:
        killer = threading.Timer(0.3, client.process.kill)
        killer.start()
        start = time.time()
        with pytest.raises(MCPError, match='not running'):
            client.call_tool('slow', {})
        assert time.time() - start < 10, 'the call waited for the timeout instead'
    finally:
        killer.cancel()
        client.close()


def test_a_closed_client_refuses_immediately(cfg):
    client = client_of(cfg).start()
    client.close()
    start = time.time()
    with pytest.raises(MCPError, match='not running'):
        client.list_tools()
    assert time.time() - start < 1


# --------------------------------------------------------------------------- calling


def test_a_text_result_is_returned(cfg):
    with client_of(cfg) as client:
        assert client.call_tool('echo', {'text': 'hi'}) == 'echo: hi'


def test_non_ascii_round_trips_exactly(cfg):
    """The stub sends raw UTF-8; the locale codec would mangle or reject it."""
    with client_of(cfg) as client:
        assert client.call_tool('echo', {'text': 'release — ops ✅'}) == 'echo: release — ops ✅'


def test_non_text_blocks_are_json_encoded(cfg):
    with client_of(cfg) as client:
        rendered = client.call_tool('optional', {'note': 'n'})
    assert 'note=n' in rendered
    assert '"type": "image"' in rendered


def test_a_tool_error_becomes_an_mcp_error(cfg):
    with client_of(cfg) as client:
        with pytest.raises(MCPError, match='boom'):
            client.call_tool('fail', {})


def test_an_unknown_tool_is_an_mcp_error(cfg):
    with client_of(cfg) as client:
        with pytest.raises(MCPError, match='unknown tool'):
            client.call_tool('nope', {})


def test_a_slow_tool_times_out(cfg_factory):
    cfg = cfg_factory(mcp_timeout=0.5)
    with client_of(cfg) as client:
        with pytest.raises(MCPError, match='timed out'):
            client.call_tool('slow', {})


def test_calling_after_close_is_an_error(cfg):
    client = client_of(cfg).start()
    client.close()
    with pytest.raises(MCPError, match='not running'):
        client.call_tool('echo', {'text': 'x'})


# --------------------------------------------------------------------------- definitions


def test_a_definition_is_prefixed_and_keeps_the_server_schema(cfg):
    with client_of(cfg) as client:
        tool = next(t for t in client.list_tools() if t['name'] == 'echo')
        definition = client.definition(tool)

    assert definition.name == 'stub__echo'
    assert definition.description == 'Echo the text back.'
    assert definition.schema == tool['inputSchema']
    assert definition.risky is True
    assert definition.wants_cfg is True


def test_a_missing_required_argument_is_refused_before_the_call(cfg):
    with client_of(cfg) as client:
        definition = client.definition(client.list_tools()[0])

    with pytest.raises(ValidationError):
        definition.parameters.model_validate_json('{}')


def test_a_tool_without_a_usable_schema_still_works(cfg):
    with client_of(cfg) as client:
        tool = {'name': 'plain', 'description': 'no schema'}
        definition = client.definition(tool)

    assert definition.schema == {'type': 'object', 'properties': {}}
    assert definition.parameters.model_validate_json('{}') is not None


def test_remote_tools_are_risky_by_default(bridge_of):
    bridge = bridge_of(stub_spec())
    assert all(definition.risky for definition in bridge.definitions)


def test_risky_follows_the_configuration(bridge_of):
    bridge = bridge_of(stub_spec(name='relaxed'), mcp_risky=False)
    assert all(not definition.risky for definition in bridge.definitions)


# --------------------------------------------------------------------------- bridge


def test_the_bridge_registers_every_tool_it_finds(bridge_of):
    bridge = bridge_of(stub_spec())
    names = [definition.name for definition in bridge.definitions]
    assert 'stub__echo' in names
    assert 'stub__no_schema' in names
    assert len(names) == 6


def test_a_broken_server_does_not_stop_the_run(bridge_of):
    bridge = bridge_of(stub_spec(), 'broken=this-command-does-not-exist')

    assert 'broken' in bridge.failures
    assert 'stub__echo' in [d.name for d in bridge.definitions]


def test_a_refusing_server_is_recorded_and_skipped(bridge_of):
    bridge = bridge_of(stub_spec('--bad-handshake', name='grumpy'))

    assert 'grumpy' in bridge.failures
    assert bridge.definitions == []


def test_closing_the_bridge_stops_the_servers(bridge_of):
    bridge = bridge_of(stub_spec())
    processes = [client.process for client in bridge.clients]

    bridge.close()

    assert all(process.poll() is not None for process in processes)
    assert bridge.definitions == []


def test_a_server_does_not_receive_the_agents_filtered_secrets(cfg_factory, monkeypatch):
    """bash_env exists to keep secrets out of the agent's shell.

    A server therefore cannot authenticate through a variable whose name looks
    like a credential. That is a deliberate consequence, not an oversight, and
    it is the reason a server needing a token must read it from its own config.
    """
    monkeypatch.setenv('SOME_SERVICE_TOKEN', 'leak-me')

    cfg = cfg_factory()

    assert 'SOME_SERVICE_TOKEN' not in cfg.bash_env
    assert 'DEEPSEEK_API_KEY' not in cfg.bash_env


# --------------------------------------------------------------------------- executor integration


def test_a_bridged_tool_dispatches_like_a_local_one(cfg_factory):
    cfg = cfg_factory(mcp_servers=(stub_spec(),))
    with MCPBridge(cfg=cfg) as bridge:
        runner = box.ToolExecution(registry_with(bridge.definitions), box._always_allow, cfg=cfg)
        result = runner.execute_tool(call('stub__echo', text='hello'), cfg=cfg)

    assert result.ok is True
    assert result.content == 'echo: hello'


def test_a_bridged_tool_reports_the_server_error_as_a_failure(cfg_factory):
    cfg = cfg_factory(mcp_servers=(stub_spec(),))
    with MCPBridge(cfg=cfg) as bridge:
        runner = box.ToolExecution(registry_with(bridge.definitions), box._always_allow, cfg=cfg)
        result = runner.execute_tool(call('stub__fail'), cfg=cfg)

    assert result.ok is False
    assert result.tag.startswith(box.TAG.EXECUTE_FAILED)
    assert 'boom' in result.content


def test_a_bridged_tool_can_be_denied_by_policy(cfg_factory):
    cfg = cfg_factory(mcp_servers=(stub_spec(),), policy_deny_tools=('stub__echo',))
    with MCPBridge(cfg=cfg) as bridge:
        runner = box.ToolExecution(registry_with(bridge.definitions), box._always_allow, cfg=cfg)
        result = runner.execute_tool(call('stub__echo', text='x'), cfg=cfg)

    assert result.tag == box.TAG.POLICY_DENIED


def test_a_bridged_tool_asks_for_approval(cfg_factory):
    asked = []

    def confirm(tool_call, cfg=None):
        asked.append(tool_call.function.name)
        return True

    cfg = cfg_factory(mcp_servers=(stub_spec(),))
    with MCPBridge(cfg=cfg) as bridge:
        runner = box.ToolExecution(registry_with(bridge.definitions), confirm, cfg=cfg)
        result = runner.execute_tool(call('stub__echo', text='x'), cfg=cfg)

    assert result.ok is True
    assert asked == ['stub__echo']


# --------------------------------------------------------------------------- tracing


def test_the_bridge_and_its_calls_are_traced(cfg_factory, session_dir):
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        cfg = cfg_factory(mcp_servers=(stub_spec(),))
        with MCPBridge(cfg=cfg) as bridge:
            runner = box.ToolExecution(registry_with(bridge.definitions), box._always_allow, cfg=cfg)
            runner.execute_tool(call('stub__echo', text='x'), cfg=cfg)
    finally:
        TRACE.configure(None)

    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    kinds = [event['event'] for event in events]
    assert 'mcp_server' in kinds
    assert 'mcp_call' in kinds
    server = next(event for event in events if event['event'] == 'mcp_server')
    assert server['server'] == 'stub'
    assert 'echo' in server['tools']
    call_event = next(event for event in events if event['event'] == 'mcp_call')
    assert call_event['ok'] is True
