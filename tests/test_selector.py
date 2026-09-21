"""On-demand tool exposure: the ranking, the budget, and the escape hatch.

The whole point of hiding a tool is that the model can still get it back, so
most of these tests are about the ways a tool survives the budget rather than
about the ranking score itself.
"""

import pytest
from pydantic import BaseModel, ConfigDict

from mini_harness.agent import CORE_TOOLS, DeepSeekAgent
from mini_harness.selector import SELECTION, describe, rank_definitions, select
from mini_harness.tool import box
from mini_harness.tool.box import TOOLS, FindToolsInput, ToolDefinition, find_tools
from mini_harness.tool.tag import OUTCOME, TAG
from tests.conftest import call, executor, write
from tests.test_agent import Stream, completion, model_message


class Anything(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = ""


def fake_tool(name, description):
    def run(args: Anything, cfg=None) -> str:
        return f"{name} ran"

    return ToolDefinition(name, description, Anything, run, False)


FAKES = [
    fake_tool("browser_open", "open a web page in a headless browser and read it"),
    fake_tool("sql_query", "run a SQL query against the postgres database"),
    fake_tool("pdf_render", "render a PDF document to images"),
    fake_tool("email_send", "send an email message to a recipient"),
    fake_tool("chart_plot", "draw a chart from numeric data"),
]


# ------------------------------------------------------------------ describing and ranking


def test_describe_covers_the_name_and_the_description():
    text = describe(FAKES[0])

    assert "browser_open" in text
    assert "headless browser" in text


def test_rank_puts_the_best_match_first():
    ordered = rank_definitions("query the postgres database", FAKES)

    assert ordered[0].name == "sql_query"


def test_rank_keeps_every_tool_and_the_unmatched_in_their_given_order():
    ordered = rank_definitions("open a web page", FAKES)

    assert len(ordered) == len(FAKES)
    assert {tool.name for tool in ordered} == {tool.name for tool in FAKES}
    assert [tool.name for tool in ordered[1:]] == ["sql_query", "pdf_render", "email_send", "chart_plot"]


def test_rank_without_definitions_is_empty():
    assert rank_definitions("anything", []) == []


def test_rank_with_no_shared_words_still_returns_everything():
    ordered = rank_definitions("zzzz yyyy", FAKES)

    assert [tool.name for tool in ordered] == [tool.name for tool in FAKES]


# ------------------------------------------------------------------ the budget


def test_a_zero_budget_exposes_everything():
    assert select("anything", FAKES, always=("browser_open",), budget=0) == FAKES


def test_a_tool_set_that_fits_the_budget_is_left_alone():
    assert select("anything", FAKES[:3], always=("browser_open",), budget=3) == FAKES[:3]


def test_the_budget_keeps_the_always_exposed_tools_and_ranks_the_rest():
    chosen = select("query the postgres database", FAKES, always=("browser_open",), budget=3)

    # sql_query matches on every content word; the rest share none and keep
    # their catalogue order.
    assert [tool.name for tool in chosen] == ["browser_open", "sql_query", "pdf_render"]


def test_the_always_exposed_tools_keep_their_catalogue_order():
    chosen = select("query the postgres database", FAKES, always=("pdf_render", "browser_open"), budget=3)

    assert [tool.name for tool in chosen][:2] == ["browser_open", "pdf_render"]


def test_a_budget_that_pins_everything_exposes_only_the_pinned():
    chosen = select("query the postgres database", FAKES, always=("browser_open", "sql_query"), budget=2)

    assert [tool.name for tool in chosen] == ["browser_open", "sql_query"]


def test_a_pulled_in_tool_is_never_hidden_again():
    SELECTION.register(FAKES)
    SELECTION.activate(["pdf_render", "chart_plot"])

    chosen = select("open a web page", FAKES, always=("browser_open",), budget=3)

    assert [tool.name for tool in chosen] == ["browser_open", "pdf_render", "chart_plot"]


# ------------------------------------------------------------------ the Selection singleton


def test_register_replaces_the_catalogue_and_forgets_the_pinned():
    SELECTION.register(FAKES)
    SELECTION.activate(["sql_query"])
    SELECTION.register(FAKES[:2])

    assert [tool.name for tool in SELECTION.catalog] == ["browser_open", "sql_query"]
    assert SELECTION.active == set()


def test_reset_empties_both_halves():
    SELECTION.register(FAKES)
    SELECTION.activate(["sql_query"])
    SELECTION.reset()

    assert SELECTION.catalog == []
    assert SELECTION.active == set()


# ------------------------------------------------------------------ frames, not a slot


def test_a_nested_registration_restores_what_was_underneath():
    """An inner agent must not take the outer agent's catalogue with it."""
    SELECTION.register(FAKES)
    SELECTION.activate(["sql_query"])
    SELECTION.register(FAKES[:2])

    assert [tool.name for tool in SELECTION.catalog] == ["browser_open", "sql_query"]
    assert SELECTION.active == set()

    SELECTION.release()

    assert [tool.name for tool in SELECTION.catalog] == [tool.name for tool in FAKES]
    assert SELECTION.active == {"sql_query"}


def test_release_without_a_frame_is_harmless():
    SELECTION.reset()

    SELECTION.release()

    assert SELECTION.catalog == []


def test_activating_without_a_frame_pins_nothing():
    SELECTION.reset()

    SELECTION.activate(["sql_query"])

    assert SELECTION.active == set()


def test_a_pin_inside_a_frame_is_scoped_to_it():
    SELECTION.register(TOOLS)
    find_tools(FindToolsInput(query="search the earlier conversation", limit=1))
    pinned = set(SELECTION.active)
    assert pinned

    SELECTION.register(TOOLS)

    assert SELECTION.active == set()


# ------------------------------------------------------------------ find_tools


def test_find_tools_pulls_the_matching_tool_in_and_names_it():
    SELECTION.register(FAKES)

    report = find_tools(FindToolsInput(query="open a web page in a browser", limit=2))

    assert "browser_open" in SELECTION.active
    assert "browser_open" in report
    assert "available from the next turn" in report


def test_find_tools_respects_its_limit():
    SELECTION.register(FAKES)

    find_tools(FindToolsInput(query="open a web page in a browser", limit=2))

    assert len(SELECTION.active) == 2


def test_find_tools_falls_back_to_the_built_in_catalogue():
    """With no run in progress it still reports, but there is nothing to pin.

    A pin means "for the rest of this run"; outside a run there is no run to
    outlive, and pinning into a frame nobody will release would leak.
    """
    SELECTION.reset()

    report = find_tools(FindToolsInput(query="read a file"))

    assert f"of {len(TOOLS)} tools match" in report
    assert SELECTION.active == set()


def test_find_tools_pulled_in_tools_survive_the_next_selection():
    SELECTION.register(FAKES)
    find_tools(FindToolsInput(query="send an email message", limit=1))

    chosen = select("unrelated words", FAKES, always=(), budget=2)

    assert [tool.name for tool in chosen] == ["email_send", "browser_open"]


# ------------------------------------------------------------------ the hidden call


def bridged_surface():
    """What a bridged server's tools look like once they reach the registry."""
    return [
        box.ToolDefinition('fs__read_text_file', 'read the complete contents of a file from the filesystem server',
                           Anything, lambda args, cfg=None: 'bridged read', False),
        box.ToolDefinition('fs__write_file', 'create or overwrite a file on the filesystem server',
                           Anything, lambda args, cfg=None: 'bridged write', True),
        box.ToolDefinition('memory__search_nodes', 'search the knowledge graph for entities and relations',
                           Anything, lambda args, cfg=None: 'bridged search', False),
    ]


def test_a_budget_takes_bridged_tools_first_not_the_core_ones(cfg_factory):
    """The built-ins are the agent's basic capability; the bridge is the surface
    that grows without bound, so a budget has to bite there."""
    cfg = cfg_factory(tool_budget=6)
    definitions = [*TOOLS, *bridged_surface()]
    agent = DeepSeekAgent(definitions, cfg=cfg)
    agent.task = 'search the knowledge graph for entities'

    exposed = {entry['function']['name'] for entry in agent._expose(cfg)}

    assert CORE_TOOLS <= exposed
    assert 'memory__search_nodes' in exposed, 'the ranked match should earn its place'
    assert 'fs__read_text_file' not in exposed
    assert 'fs__write_file' not in exposed


def test_a_hidden_bridged_tool_can_be_pulled_back_by_name(cfg):
    """The escape hatch has to work for tools this process cannot change."""
    definitions = [*TOOLS, *bridged_surface()]
    registry = {tool.name: tool for tool in definitions}
    execution = box.ToolExecution(registry, box._always_allow, cfg=cfg)
    execution.hidden = {'fs__read_text_file'}
    SELECTION.register(definitions)

    refused = execution.execute_tool(call('fs__read_text_file'), cfg=cfg)
    report = find_tools(box.FindToolsInput(query='read the complete contents of a file', limit=1))
    execution.hidden = set()
    allowed = execution.execute_tool(call('fs__read_text_file', call_id='second'), cfg=cfg)

    assert refused.tag == TAG.HIDDEN_TOOL
    assert 'fs__read_text_file' in report
    assert 'fs__read_text_file' in SELECTION.active
    assert allowed.ok
    assert allowed.content == 'bridged read'


def test_a_call_to_a_hidden_tool_is_refused_with_the_way_back(cfg):
    execution = executor(cfg)
    execution.hidden = {"run_bash"}

    result = execution.execute_tool(call("run_bash", command="ls"), cfg=cfg)

    assert not result.ok
    assert result.tag == TAG.HIDDEN_TOOL
    assert "find_tools" in result.content


def test_a_hidden_tool_runs_once_it_is_pulled_back_in(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    execution = executor(cfg)
    execution.hidden = {"read_file"}

    refused = execution.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    SELECTION.register(TOOLS)
    SELECTION.activate(["read_file"])
    execution.hidden = set()
    allowed = execution.execute_tool(call("read_file", call_id="second", file_path="sandbox/f.py"), cfg=cfg)

    assert refused.tag == TAG.HIDDEN_TOOL
    assert allowed.ok


def test_a_visible_tool_is_unaffected_by_the_hidden_set(cfg):
    execution = executor(cfg)
    execution.hidden = set()

    result = execution.execute_tool(call("run_todo", items=[{"content": "x", "activeForm": "x", "status": "pending"}]), cfg=cfg)

    assert result.ok


# ------------------------------------------------------------------ the agent side


def test_the_built_in_catalogue_starts_with_only_the_core_tools_exposed(cfg):
    assert CORE_TOOLS == {"find_tools", "read_file", "grep_file", "glob_file"}


def test_an_agent_without_a_budget_exposes_every_tool(cfg):
    agent = DeepSeekAgent(TOOLS, cfg=cfg)

    assert len(agent._expose(cfg)) == len(TOOLS)


def test_a_budget_shrinks_the_request_and_keeps_the_core_tools(cfg_factory):
    cfg = cfg_factory(tool_budget=6)
    agent = DeepSeekAgent(TOOLS, cfg=cfg)
    agent.task = "query the postgres database"

    names = [entry["function"]["name"] for entry in agent._expose(cfg)]

    assert len(names) == 6
    assert CORE_TOOLS <= set(names)


def test_the_request_is_only_rebuilt_when_the_exposed_set_changes(cfg_factory):
    cfg = cfg_factory(tool_budget=6)
    agent = DeepSeekAgent(TOOLS, cfg=cfg)
    agent.task = "query the postgres database"

    first = agent._expose(cfg)

    assert agent._expose(cfg) is first


def test_a_run_refuses_a_hidden_tool_and_points_at_find_tools(cfg_factory):
    from tests.test_agent import FakeClient

    cfg = cfg_factory(tool_budget=4)
    agent = DeepSeekAgent(TOOLS, cfg=cfg)
    agent.session_memory = cfg.session_path
    client = FakeClient([
        Stream(completion(model_message("", [call("run_bash", command="ls")]))),
        Stream(completion(model_message("done"))),
    ])
    agent.message.append({"role": "user", "content": "list the files"})

    result = agent._run_turn(client, executor(cfg), cfg=cfg)

    assert result.outcome == OUTCOME.COMPLETED
    assert result.failed_by_tag == {TAG.HIDDEN_TOOL: 1}
    assert len(client.stream_calls[0]["tools"]) == 4
    assert any("find_tools" in str(message.get("content", "")) for message in agent.message)
