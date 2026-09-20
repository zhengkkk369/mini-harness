"""ToolExecution: the validation, feedback and file-state gate pipeline."""

import json

import pytest
from pydantic import BaseModel, ConfigDict

from mini_harness.tool import box
from mini_harness.tool.block import TODO
from mini_harness.tool.tag import LEVEL, TAG
from mini_harness.tool.box import ToolDefinition
from tests.conftest import ALWAYS_ALLOW, ALWAYS_DENY, call, executor, write


def make_executor(cfg, confirm=ALWAYS_ALLOW, **tools):
    """Real file tools plus any ad-hoc tool the test adds."""
    registry = {name: t for name, t in ((t.name, t) for t in box.TOOLS)
                if name in {"glob_file", "grep_file", "read_file", "write_file", "edit_file"}}
    registry.update(tools)
    return box.ToolExecution(registry, confirm, cfg=cfg)


def echo_tool(risky=False):
    """A tool with the same signature convention as the real ones."""

    class EchoInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        text: str

    def echo(args: EchoInput, cfg=None) -> str:
        return args.text

    return ToolDefinition("echo_tool", "echo", EchoInput, echo, risky)


# --------------------------------------------------------------------- dispatch basics


def test_unknown_tool_is_reported(cfg):
    result = executor(cfg).execute_tool(call("nope_tool"), cfg=cfg)
    assert not result.ok
    assert result.tag == TAG.UNKNOWN_TOOL
    assert "unknown" in result.content


def test_malformed_json_arguments_are_reported(cfg):
    registry = {"echo_tool": echo_tool()}
    broken = call("echo_tool")
    broken.function.arguments = "{not json"

    result = make_executor(cfg, **registry).execute_tool(broken, cfg=cfg)

    assert not result.ok
    assert result.tag == TAG.INVALID_ARGS
    assert "cannot unpack the json format" in result.content


def test_extra_arguments_are_rejected_before_execution(cfg):
    registry = {"echo_tool": echo_tool()}

    result = make_executor(cfg, **registry).execute_tool(
        call("echo_tool", text="hi", surprise=True), cfg=cfg
    )

    assert result.tag == TAG.INVALID_ARGS


def test_successful_call_returns_the_tool_output(cfg):
    registry = {"echo_tool": echo_tool()}
    result = make_executor(cfg, **registry).execute_tool(call("echo_tool", text="hi"), cfg=cfg)
    assert result.ok
    assert result.content == "hi"
    assert result.tag == TAG.SUCCESS


def test_raising_tool_is_wrapped_with_the_exception_type(cfg):
    def explode(_args, cfg=None):
        raise ZeroDivisionError("boom")

    class In(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry = {"boom": ToolDefinition("boom", "boom", In, explode, False)}

    result = make_executor(cfg, **registry).execute_tool(call("boom"), cfg=cfg)

    assert not result.ok
    assert "ZeroDivisionError" in result.content
    assert result.tag == f"{TAG.EXECUTE_FAILED}:ZeroDivisionError"


def test_non_string_results_are_json_encoded(cfg):
    class In(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry = {"obj": ToolDefinition("obj", "obj", In, lambda _a, cfg=None: {"a": 1}, False)}
    assert json.loads(make_executor(cfg, **registry).execute_tool(call("obj"), cfg=cfg).content) == {"a": 1}


# --------------------------------------------------------------------- duplicate guard


def test_identical_consecutive_call_is_refused(cfg):
    registry = {"echo_tool": echo_tool()}
    runner = make_executor(cfg, **registry)

    assert runner.execute_tool(call("echo_tool", text="same"), cfg=cfg).ok
    second = runner.execute_tool(call("echo_tool", text="same"), cfg=cfg)

    assert not second.ok
    assert second.tag == TAG.DEDUP


def test_changed_arguments_are_allowed_again(cfg):
    registry = {"echo_tool": echo_tool()}
    runner = make_executor(cfg, **registry)

    runner.execute_tool(call("echo_tool", text="one"), cfg=cfg)
    assert runner.execute_tool(call("echo_tool", text="two"), cfg=cfg).ok


def test_invalid_arguments_do_not_set_the_last_call(cfg):
    registry = {"echo_tool": echo_tool()}
    runner = make_executor(cfg, **registry)
    broken = call("echo_tool", text="hi")
    broken.function.arguments = "{oops"

    assert not runner.execute_tool(broken, cfg=cfg).ok
    assert runner.execute_tool(call("echo_tool", text="hi"), cfg=cfg).ok


def test_a_denied_risky_call_is_not_remembered_as_the_last_call(cfg):
    runner = make_executor(cfg, confirm=ALWAYS_DENY, echo_tool=echo_tool(risky=True))

    assert runner.execute_tool(call("echo_tool", text="x"), cfg=cfg).tag == TAG.DENIED
    assert not runner.execute_tool(call("echo_tool", text="x"), cfg=cfg).ok


# --------------------------------------------------------------------- approval flow


def test_risky_tool_requires_confirmation(cfg):
    seen = []

    def deny(tool_call, cfg=None):
        seen.append(tool_call.function.name)
        return False

    runner = make_executor(cfg, confirm=deny, echo_tool=echo_tool(risky=True))
    result = runner.execute_tool(call("echo_tool", text="hi"), cfg=cfg)

    assert seen == ["echo_tool"]
    assert result.tag == TAG.DENIED
    assert "denied by the user" in result.content


def test_approved_risky_tool_runs(cfg):
    runner = make_executor(cfg, confirm=ALWAYS_ALLOW, echo_tool=echo_tool(risky=True))
    assert runner.execute_tool(call("echo_tool", text="hi"), cfg=cfg).ok


def test_non_risky_tool_never_asks(cfg):
    asked = []

    def confirm(tool_call, cfg=None):
        asked.append(tool_call.function.name)
        return True

    runner = make_executor(cfg, confirm=confirm, echo_tool=echo_tool(risky=False))
    runner.execute_tool(call("echo_tool", text="hi"), cfg=cfg)

    assert asked == []


# --------------------------------------------------------------------- edit_file gate


def test_edit_without_a_prior_read_is_refused_with_content(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    runner = make_executor(cfg)

    result = runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="b"), cfg=cfg)

    assert not result.ok
    assert result.tag == TAG.NEED_READ
    assert "alpha" in result.content
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "alpha\n"


def test_edit_after_a_full_read_succeeds(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    result = runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="beta"), cfg=cfg)

    assert result.ok
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "beta\n"


def test_a_partial_read_is_enough_for_edit(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "a\nb\nc\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py", offset=1, limit=1), cfg=cfg)
    assert runner.files[str((workspace / "sandbox" / "f.py").resolve())].level == LEVEL.PARTIAL
    assert runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="c", new_string="C"), cfg=cfg).ok


def test_edit_on_an_external_change_is_refused_as_stale(cfg, workspace):
    target = write(workspace / "sandbox" / "f.py", "alpha 0123456789\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    write(target, "alpha rewritten by something else\n")
    result = runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="b"), cfg=cfg)

    assert result.tag == TAG.STALE
    assert "changed after you last read it" in result.content


def test_edit_is_allowed_when_only_the_mtime_changed(cfg, workspace):
    """mtime alone must not invalidate a read: content is what the edit replaces."""
    import os

    target = write(workspace / "sandbox" / "f.py", "alpha\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    os.utime(target, (1_000_000, 1_000_000))
    result = runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="b"), cfg=cfg)

    assert result.ok
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "b\n"


def test_edit_of_a_missing_file_bypasses_the_read_gate(cfg):
    result = make_executor(cfg).execute_tool(
        call("edit_file", file_path="sandbox/ghost.py", old_string="a", new_string="b"), cfg=cfg
    )
    assert result.tag.startswith(TAG.EXECUTE_FAILED)


def test_edit_gate_can_be_disabled(cfg_factory, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    loose = cfg_factory(edit_require_read=False)
    result = make_executor(loose).execute_tool(
        call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="b"), cfg=loose
    )
    assert result.ok


# --------------------------------------------------------------------- write_file gate


def test_write_over_an_existing_file_needs_overwrite(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")

    result = make_executor(cfg).execute_tool(call("write_file", file_path="sandbox/f.py", content="new"), cfg=cfg)

    assert result.tag == TAG.EXISTS
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "alpha\n"


def test_write_over_a_file_read_only_partially_needs_a_full_read(cfg_factory, workspace):
    write(workspace / "sandbox" / "f.py", "".join(f"{i}\n" for i in range(200)))
    cfg = cfg_factory(read_limit=60)
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    result = runner.execute_tool(
        call("write_file", file_path="sandbox/f.py", content="x", overwrite=True), cfg=cfg
    )

    assert result.tag == TAG.NEED_FULL
    assert "end to end" in result.content


def test_write_after_a_full_read_succeeds(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    result = runner.execute_tool(
        call("write_file", file_path="sandbox/f.py", content="brand new\n", overwrite=True), cfg=cfg
    )

    assert result.ok
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "brand new\n"


def test_write_to_a_new_path_is_never_gated(cfg):
    assert make_executor(cfg).execute_tool(
        call("write_file", file_path="sandbox/fresh.py", content="x"), cfg=cfg
    ).ok


def test_write_over_a_stale_read_is_refused(cfg, workspace):
    target = write(workspace / "sandbox" / "f.py", "alpha one\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    write(target, "alpha two, longer content\n")
    result = runner.execute_tool(
        call("write_file", file_path="sandbox/f.py", content="x", overwrite=True), cfg=cfg
    )

    assert result.tag == TAG.STALE


def test_write_gate_can_be_disabled(cfg_factory, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    loose = cfg_factory(write_require_read=False)
    assert make_executor(loose).execute_tool(
        call("write_file", file_path="sandbox/f.py", content="new"), cfg=loose
    ).ok


# --------------------------------------------------------------------- read tracking


def test_grep_marks_hit_files_as_partially_read(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "beta = 1\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("grep_file", pattern="beta", path="."), cfg=cfg)

    record = runner.files.get(str((workspace / "sandbox" / "f.py").resolve()))
    assert record is not None
    assert record.level == LEVEL.PARTIAL


def test_grep_of_an_empty_result_marks_nothing(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "beta = 1\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("grep_file", pattern="zzz", path="."), cfg=cfg)

    assert runner.files == {}


def test_reading_a_clipped_file_keeps_the_full_level(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "a\nb\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)

    assert runner.files[str((workspace / "sandbox" / "f.py").resolve())].level == LEVEL.FULL


def test_tracking_can_be_switched_off_entirely(cfg_factory, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    cfg = cfg_factory(track_files=False)
    runner = make_executor(cfg)

    assert runner.execute_tool(
        call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="b"), cfg=cfg
    ).ok
    assert runner.files == {}


def test_edit_counter_is_incremented(cfg, workspace):
    key = str((workspace / "sandbox" / "f.py").resolve())
    write(workspace / "sandbox" / "f.py", "alpha\n")
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="alpha one"), cfg=cfg)
    result = runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha one", new_string="alpha two"), cfg=cfg)

    assert runner.files[key].edits == 2
    assert result.ok
    assert "modified this file" not in result.content


def test_thrash_notice_fires_at_the_configured_count(cfg_factory, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    cfg = cfg_factory(thrash_notice=2)
    runner = make_executor(cfg)

    runner.execute_tool(call("read_file", file_path="sandbox/f.py"), cfg=cfg)
    runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="alpha one"), cfg=cfg)
    result = runner.execute_tool(call("edit_file", file_path="sandbox/f.py", old_string="alpha one", new_string="alpha two"), cfg=cfg)

    assert "modified this file 2 times" in result.content


# --------------------------------------------------------------------- registry wiring


def test_default_registry_exposes_every_built_in_tool():
    assert len(box.TOOLS) == 10
    assert {t.name for t in box.TOOLS} == {
        "glob_file", "grep_file", "read_file", "write_file", "edit_file",
        "run_bash", "run_sandbox", "run_todo", "recall", "run_subagent",
    }


def test_only_shell_and_subagent_tools_are_risky():
    assert {t.name for t in box.TOOLS if t.risky} == {"run_bash", "run_sandbox", "run_subagent"}


def test_tool_schemas_are_exportable_json():
    api_tools = box._to_api_tool(box.TOOLS)
    assert len(api_tools) == 10
    for entry in api_tools:
        assert entry["type"] == "function"
        assert entry["function"]["parameters"]["type"] == "object"


def test_run_todo_through_the_executor_prints_nothing_extra(cfg):
    """The todo tool is dispatched like any other tool and returns its render."""
    result = executor(cfg).execute_tool(
        call("run_todo", items=[{"content": "step", "activeForm": "stepping", "status": "pending"}]), cfg=cfg
    )
    assert result.ok
    assert result.content == "[ ] step\n0 / 1 completed"
    assert TODO.items


def test_log_tool_prefixes_failures(cfg, capsys):
    ok_call = call("read_file", file_path="f.py")
    box.log_tool(ok_call, box.ToolItem("body", True, TAG.SUCCESS), cfg=cfg)
    assert capsys.readouterr().out == 'read_file: {"file_path": "f.py"}\n'

    box.log_tool(ok_call, box.ToolItem("body", False, TAG.STALE), cfg=cfg)
    assert "Failed stale" in capsys.readouterr().out


def test_log_tool_clips_long_arguments(cfg, capsys):
    box.log_tool(call("run_bash", command="x" * 200), box.ToolItem("", True, TAG.SUCCESS), cfg=cfg)
    assert "clipped at 100 chars" in capsys.readouterr().out
