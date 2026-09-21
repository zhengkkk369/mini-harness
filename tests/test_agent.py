"""The agent loop: outcomes, truncation recovery, memory, telemetry, dedup."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion

from mini_harness import main as cli
from mini_harness.agent import DeepSeekAgent, Result
from mini_harness.tool import box
from mini_harness.tool.box import TOOLS, ToolDefinition
from mini_harness.tool.tag import OUTCOME
from mini_harness.trace import TRACE
from tests.conftest import REGISTRY, call, executor, write


def model_message(content="", tool_calls=None, reasoning=""):
    return SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning,
        model_dump=lambda **_: {"role": "assistant", "content": content},
    )


def completion(message, prompt_tokens=100, completion_tokens=20, cached_tokens=None):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    if cached_tokens is not None:
        usage.prompt_cache_hit_tokens = cached_tokens
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )


class Stream:
    """Stands in for openai's ChatCompletionStream context manager."""

    def __init__(self, complete, error=None, usage=None):
        self.complete = complete
        self.error = error
        self.usage = usage

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        if self.usage is not None:
            yield SimpleNamespace(type="chunk", chunk=SimpleNamespace(usage=self.usage, choices=[]))
        return iter(())

    def get_final_completion(self):
        if self.error is not None:
            raise self.error
        return self.complete


class ReasoningStream(Stream):
    """Emits reasoning_content deltas, as DeepSeek does."""

    def __init__(self, *deltas):
        super().__init__(completion(model_message("answer")))
        self.deltas = deltas

    def __iter__(self):
        for delta in self.deltas:
            message = SimpleNamespace(reasoning_content=delta)
            yield SimpleNamespace(type="chunk", chunk=SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=message)]))


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.stream_calls = []
        self.create_calls = []

    def _stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return self.script.pop(0)

    def _create(self, **kwargs):
        self.create_calls.append(kwargs)
        return completion(model_message("summary"))

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(stream=self._stream, create=self._create))


def make_agent(cfg, *script):
    agent = DeepSeekAgent(TOOLS, cfg=cfg)
    agent.session_memory = Path(cfg.session_path)
    return agent, FakeClient(script)


def send(agent, client, cfg, execu=None, **kwargs):
    agent.message.append({"role": "user", "content": "do the thing"})
    return agent._run_turn(client, execu if execu is not None else executor(cfg), cfg=cfg, **kwargs)


def stub_shell_executor(cfg, seen=None, confirm=None):
    """An executor whose run_bash records its command instead of spawning one."""
    def stub_shell(args, cfg=None):
        if seen is not None:
            seen.append(args.command)
        return 'ok'

    registry = dict(REGISTRY)
    registry['run_bash'] = ToolDefinition('run_bash', 'stub shell', box.RunBashInput, stub_shell, True)
    return box.ToolExecution(registry, confirm or box._always_allow, cfg=cfg)


# --------------------------------------------------------------------------- happy path


def test_tool_call_then_answer_completes(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    cfg = cfg
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]))),
        Stream(completion(model_message("all done"))),
    )

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.COMPLETED
    assert result.calls == 1
    assert result.turns == 2
    assert result.ok == 1
    assert result.calls_by_tool == {"read_file": 1}
    assert result.failed_by_tag == {}
    assert result.prompt_total == 200
    assert result.completion_total == 40
    assert result.wall >= 0
    assert result.err == ""


def test_tool_result_is_appended_as_a_tool_message(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    tool_call = call("read_file", file_path="sandbox/f.py")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [tool_call]))),
        Stream(completion(model_message("done"))),
    )

    send(agent, client, cfg)

    tool_messages = [m for m in agent.message if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == tool_call.id
    assert "alpha" in tool_messages[0]["content"]


def test_identical_consecutive_tool_calls_are_deduplicated(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]))),
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]))),
        Stream(completion(model_message("done"))),
    )

    result = send(agent, client, cfg)

    assert result.calls == 2
    assert result.ok == 1
    assert result.failed_by_tag == {"duplicate": 1}


def test_an_edit_without_a_read_is_refused_and_reported(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="beta"),
        ]))),
        Stream(completion(model_message("done"))),
    )

    result = send(agent, client, cfg)

    assert result.failed_by_tag == {"need_read": 1}
    assert result.ok == 0
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "alpha\n"


def test_usage_is_reported_to_the_caller(cfg):
    agent, client = make_agent(cfg, Stream(completion(model_message("hi"), 321, 12)))
    result = send(agent, client, cfg)
    assert (result.last_prompt, result.prompt_total, result.completion_total) == (321, 321, 12)


def test_reasoning_content_is_preserved_on_the_response(cfg):
    stream = Stream(
        completion(model_message("hi", reasoning="because")),
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )
    agent, client = make_agent(cfg, stream)

    agent._request_agent(client, cfg=cfg)

    assert agent.last_usage.prompt_tokens == 100


def test_reasoning_chunks_are_accumulated(cfg):
    agent, client = make_agent(cfg)
    client.script = [ReasoningStream("step one ", "step two")]

    agent._request_agent(client, cfg=cfg)

    assert agent.last_reasoning == "step one step two"


def test_request_carries_tools_and_stream_usage(cfg):
    agent, client = make_agent(cfg, Stream(completion(model_message("hi"))))
    agent.message.append({"role": "user", "content": "hello"})

    agent._request_agent(client, cfg=cfg)

    request = client.stream_calls[0]
    assert request["stream_options"] == {"include_usage": True}
    assert request["tools"] == agent.tools
    assert request["messages"][-1] == {"role": "user", "content": "hello"}


# --------------------------------------------------------------------------- truncation


def truncated_error(prompt_tokens=500, completion_tokens=1000):
    raw = ChatCompletion.model_validate({
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant", "content": "partial"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    })
    return LengthFinishReasonError(completion=raw)


def test_a_truncated_response_is_retried_as_a_new_turn(cfg):
    agent, client = make_agent(
        cfg,
        Stream(None, error=truncated_error()),
        Stream(completion(model_message("short answer"), 600, 30)),
    )

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.COMPLETED
    assert result.turns == 2
    assert len(client.stream_calls) == 2


def test_truncation_is_fed_back_as_a_concise_hint(cfg):
    agent, client = make_agent(
        cfg,
        Stream(None, error=truncated_error()),
        Stream(completion(model_message("short answer"))),
    )

    send(agent, client, cfg)

    truncated_message = agent.message[2]
    assert truncated_message["role"] == "assistant"
    assert "cut off at the output token limit" in truncated_message["content"]
    assert "Be more concise" in truncated_message["content"]


def test_truncation_records_the_token_usage(cfg):
    agent, client = make_agent(
        cfg,
        Stream(None, error=truncated_error(prompt_tokens=700, completion_tokens=900)),
        Stream(completion(model_message("done"), 800, 40)),
    )

    result = send(agent, client, cfg)

    assert result.prompt_total == 1500
    assert result.completion_total == 940


# --------------------------------------------------------------------------- outcomes


def test_exhausting_the_turn_budget_is_reported(cfg_factory):
    cfg = cfg_factory(max_turns_main=2)
    write(cfg.work_space / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]))),
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]))),
    )

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.EXHAUSTED
    assert result.turns == 2


def test_a_failing_request_becomes_an_error_outcome(cfg):
    agent, client = make_agent(cfg, Stream(None, error=ValueError("api exploded")))
    result = send(agent, client, cfg)
    assert result.outcome == OUTCOME.ERROR
    assert "api exploded" in result.err


def test_keyboard_interrupt_during_a_tool_is_recorded(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")

    def interrupt(tool_call, cfg=None):
        raise KeyboardInterrupt

    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("run_bash", command="ls")]))),
    )
    agent.message.append({"role": "user", "content": "go"})

    result = agent._run_turn(client, executor(cfg, interrupt), cfg=cfg)

    assert result.outcome == OUTCOME.INTERRUPTED


def test_a_denied_risky_tool_returns_a_denial_note(cfg, workspace):
    def deny(tool_call, cfg=None):
        return False

    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("run_bash", command="rm -rf /")]))),
        Stream(completion(model_message("understood"))),
    )
    agent.message.append({"role": "user", "content": "go"})
    result = agent._run_turn(client, executor(cfg, deny), cfg=cfg)

    assert result.calls == 1
    assert result.ok == 0
    assert result.failed_by_tag == {"denied": 1}
    assert "denied by the user" in [m for m in agent.message if m["role"] == "tool"][-1]["content"]


def test_wall_budget_stops_the_run(cfg_factory, monkeypatch):
    cfg = cfg_factory(wall_budget=0.0)
    agent, client = make_agent(cfg, Stream(completion(model_message("never used"))))
    result = send(agent, client, cfg)
    assert result.outcome == OUTCOME.TIMEOUT
    assert result.turns == 0
    assert result.stopped_by == "wall"


def test_token_budget_stops_the_run(cfg_factory):
    """A spent budget does not start new work.

    The request that spent the budget is already paid for, but the tool calls it
    asked for are not run: their results could only be used by the next request,
    which the budget check at the top of the turn will refuse. Running them would
    be wasted work and, for a write, an unwanted side effect on a finished run.
    """
    cfg = cfg_factory(token_budget=100)
    write(cfg.work_space / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 90, 20)),
    )

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.BUDGET
    assert result.stopped_by == "tokens"
    assert result.turns == 1
    assert result.calls == 0
    assert (result.prompt_total, result.completion_total) == (90, 20)
    assert len(client.stream_calls) == 1
    skipped = [message for message in agent.message if message["role"] == "tool"]
    assert len(skipped) == 1
    assert "skipped" in skipped[0]["content"]


def test_cost_budget_stops_the_run(cfg_factory):
    cfg = cfg_factory(cost_budget=0.5, price_in=1.0, price_out=0.0)
    write(cfg.work_space / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 600_000, 0)),
    )

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.BUDGET
    assert result.stopped_by == "cost"
    assert result.cost == pytest.approx(0.6)


def test_a_run_within_budget_reports_no_stop(cfg_factory):
    cfg = cfg_factory(token_budget=10_000, cost_budget=100.0, price_in=1.0, price_out=1.0)
    agent, client = make_agent(cfg, Stream(completion(model_message("done"), 10, 5)))

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.COMPLETED
    assert result.stopped_by == ""
    assert result.cost > 0


def test_cost_stays_zero_without_prices(cfg_factory):
    cfg = cfg_factory(cost_budget=1.0)
    agent, client = make_agent(cfg, Stream(completion(model_message("done"), 1000, 1000)))

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.COMPLETED
    assert result.cost == 0.0


def test_cost_accounts_for_cached_input(cfg_factory):
    """Cached input can be a fraction of the normal rate, so it must be split."""
    cfg = cfg_factory(price_in=1.0, price_out=0.0, price_cache_in=0.1)
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("done"), 1_000_000, 0, cached_tokens=900_000)),
    )

    result = send(agent, client, cfg)

    # 100k uncached at 1.0 plus 900k cached at 0.1
    assert result.cost == pytest.approx(0.19)


def test_an_unknown_cache_shape_bills_the_prompt_in_full(cfg_factory):
    cfg = cfg_factory(price_in=1.0, price_out=0.0, price_cache_in=0.1)
    agent, client = make_agent(cfg, Stream(completion(model_message("done"), 1_000_000, 0)))

    result = send(agent, client, cfg)

    assert result.cost == pytest.approx(1.0)


def test_interrupted_tool_calls_are_closed_off(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(cfg)
    agent.message.append({"role": "assistant", "content": "", "tool_calls": []})
    pending = [call("read_file", file_path="sandbox/f.py")]

    agent._fill_interrupted(pending, cfg=cfg)

    filler = [m for m in agent.message if m["role"] == "tool"]
    assert filler[0]["tool_call_id"] == pending[0].id
    # The placeholder names the tool and gives the reason; the API only needs the
    # id to match, but a human reading the session needs to know what happened.
    assert filler[0]["content"].startswith("[read_file skipped]:")
    assert "interrupted" in filler[0]["content"]


# --------------------------------------------------------------------------- memory


def test_run_task_starts_from_a_fresh_system_message(cfg, workspace, patch_openai):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(cfg, Stream(completion(model_message("done"))))
    agent.message.append({"role": "user", "content": "old task"})

    patch_openai(client)
    agent.run_task("new task", cfg=cfg)

    assert agent.message[0]["role"] == "system"
    assert agent.message[1] == {"role": "user", "content": "new task"}


def test_memory_is_written_after_a_run(cfg, workspace, patch_openai):
    agent, client = make_agent(cfg, Stream(completion(model_message("done"))))

    patch_openai(client)
    agent.run_task("do it", cfg=cfg)

    saved = json.loads(Path(cfg.session_path).read_text(encoding="utf-8"))
    assert saved[0]["role"] == "system"
    assert saved[1]["content"] == "do it"


def test_memory_can_be_resumed_when_the_user_agrees(cfg, workspace, monkeypatch, capsys):
    Path(cfg.session_path).write_text(
        json.dumps([{"role": "system", "content": "old"}, {"role": "user", "content": "previous"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda *_: "yes")
    agent, _ = make_agent(cfg)

    agent._load_memory(cfg=cfg)

    assert [m["content"] for m in agent.message] == ["old", "previous"]


def test_memory_is_ignored_when_the_user_declines(cfg, workspace, monkeypatch):
    Path(cfg.session_path).write_text(
        json.dumps([{"role": "user", "content": "previous"}]), encoding="utf-8"
    )
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    agent, _ = make_agent(cfg)

    agent._load_memory(cfg=cfg)

    assert len(agent.message) == 1
    assert agent.message[0]["role"] == "system"


def test_a_corrupt_memory_file_is_reported_and_ignored(cfg, workspace, monkeypatch, capsys):
    Path(cfg.session_path).write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda *_: "yes")
    agent, _ = make_agent(cfg)

    agent._load_memory(cfg=cfg)

    assert "cannot load memory" in capsys.readouterr().out
    assert len(agent.message) == 1


def test_missing_memory_needs_no_question(cfg, workspace, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("should not ask"))
    agent, _ = make_agent(cfg)
    assert len(agent._load_memory(cfg=cfg)) == 1


# --------------------------------------------------------------------------- telemetry


def test_dump_run_emits_a_marked_json_payload(cfg, workspace, capsys, tmp_path):
    result = Result("completed", 2, 3, 2, {}, {}, 10, 10, 5, 1.5)
    agent, _ = make_agent(cfg)
    out = tmp_path / "tele.json"

    agent.dump_run(result, "the task", str(out), cfg=cfg)

    line = [ln for ln in capsys.readouterr().out.splitlines() if "MINI_HARNESS_RUN" in ln][0]
    payload = json.loads(line.split("####MINI_HARNESS_RUN####", 1)[1])
    assert payload["task"] == "the task"
    assert payload["model"] == cfg.model_main
    assert payload["outcome"] == "completed"
    assert json.loads(out.read_text(encoding="utf-8"))["calls"] == 2


def test_dump_run_survives_an_unwritable_path(cfg, workspace, capsys, tmp_path):
    blocked = tmp_path / "adir"
    blocked.mkdir()
    agent, _ = make_agent(cfg)

    agent.dump_run(Result("completed", 0, 1, 0, {}, {}, 0, 0, 0, 0.1), "task", str(blocked), cfg=cfg)

    assert "dump failed" in capsys.readouterr().out


def test_tool_schemas_match_the_registry(cfg):
    agent, _ = make_agent(cfg)
    assert [t["function"]["name"] for t in agent.tools] == [t.name for t in TOOLS]


# --------------------------------------------------------------------------- retrievable memory


def test_a_compacted_run_can_recall_what_compaction_removed(cfg_factory, workspace, session_dir):
    """The end-to-end loop: compact, lose the detail, then search it back."""
    cfg = cfg_factory(compact_limit=10, recent_keep=2)
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 100, 5)),
        Stream(completion(model_message("", [call("recall", query="PLUMBUS")]), 100, 5)),
        Stream(completion(model_message("done"), 100, 5)),
    )
    agent.message.append({"role": "user", "content": "the code word is PLUMBUS, remember it"})

    result = agent._run_turn(client, executor(cfg), cfg=cfg)

    assert result.outcome == OUTCOME.COMPLETED
    journal = session_dir / "mini_harness_history.jsonl"
    assert journal.exists()
    assert "PLUMBUS" in journal.read_text(encoding="utf-8")
    # the live context no longer carries it; only the tool result does
    assert "PLUMBUS" not in str(agent.message[1])
    assert any("PLUMBUS" in m["content"] for m in agent.message if m["role"] == "tool")


def test_the_compaction_summary_points_at_recall(cfg_factory, workspace):
    cfg = cfg_factory(compact_limit=10, recent_keep=2)
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 100, 5)),
        Stream(completion(model_message("done"), 100, 5)),
    )
    agent.message.append({"role": "user", "content": "remember the code word PLUMBUS"})

    agent._run_turn(client, executor(cfg), cfg=cfg)

    summary = [m for m in agent.message if m["role"] == "assistant" and m.get("content")]
    assert any("call recall(query)" in str(m["content"]) for m in summary)


# --------------------------------------------------------------------------- verification

def test_an_unverified_edit_costs_one_extra_turn(cfg_factory, workspace):
    cfg = cfg_factory(verify_required=True, verify_nudges=1)
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 10, 5)),
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="beta")]), 10, 5)),
        Stream(completion(model_message("I changed the file"), 10, 5)),
        Stream(completion(model_message("still nothing run"), 10, 5)),
    )

    result = send(agent, client, cfg)

    assert result.outcome == OUTCOME.COMPLETED
    assert result.turns == 4
    assert result.mutations == 1
    assert result.verified is False
    assert len(client.stream_calls) == 4


def test_the_nudge_asks_for_a_run(cfg_factory, workspace):
    cfg = cfg_factory(verify_required=True, verify_nudges=1)
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 10, 5)),
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="beta")]), 10, 5)),
        Stream(completion(model_message("done"), 10, 5)),
        Stream(completion(model_message("done again"), 10, 5)),
    )

    send(agent, client, cfg)

    nudges = [m for m in agent.message
              if m["role"] == "user" and "have not run anything since" in m["content"]]
    assert len(nudges) == 1


def test_a_run_after_an_edit_counts_as_verification(cfg_factory, workspace):
    cfg = cfg_factory(verify_required=True)
    write(workspace / "sandbox" / "f.py", "alpha\n")
    seen = []
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 10, 5)),
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="beta")]), 10, 5)),
        Stream(completion(model_message("", [call("run_bash", command="echo ok")]), 10, 5)),
        Stream(completion(model_message("verified"), 10, 5)),
    )

    result = send(agent, client, cfg, execu=stub_shell_executor(cfg, seen))

    assert result.turns == 4
    assert result.mutations == 1
    assert result.verified is True
    assert seen == ["echo ok"]
    assert len(client.stream_calls) == 4


def test_verification_can_be_switched_off(cfg_factory, workspace):
    cfg = cfg_factory(verify_required=False)
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 10, 5)),
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="beta")]), 10, 5)),
        Stream(completion(model_message("done"), 10, 5)),
    )

    result = send(agent, client, cfg)

    assert result.turns == 3
    assert result.mutations == 1
    assert result.verified is False
    assert len(client.stream_calls) == 3


def test_the_nudge_count_is_bounded(cfg_factory, workspace):
    cfg = cfg_factory(verify_required=True, verify_nudges=2)
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 10, 5)),
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="beta")]), 10, 5)),
        Stream(completion(model_message("answer one"), 10, 5)),
        Stream(completion(model_message("answer two"), 10, 5)),
        Stream(completion(model_message("answer three"), 10, 5)),
    )

    result = send(agent, client, cfg)

    assert result.turns == 5
    assert len(client.stream_calls) == 5
    assert result.verified is False


def test_a_read_only_run_needs_no_verification(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 10, 5)),
        Stream(completion(model_message("done"), 10, 5)),
    )

    result = send(agent, client, cfg)

    assert result.turns == 2
    assert result.mutations == 0
    assert result.verified is True


def test_a_refused_edit_is_not_a_mutation(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="b")]), 10, 5)),
        Stream(completion(model_message("done"), 10, 5)),
    )

    result = send(agent, client, cfg)

    assert result.mutations == 0
    assert result.verified is True
    assert result.turns == 2


def test_a_policy_denial_is_reported_as_its_own_tag(cfg_factory, workspace):
    cfg = cfg_factory(policy_deny_tools=("run_bash",))
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("run_bash", command="ls")]), 10, 5)),
        Stream(completion(model_message("understood"), 10, 5)),
    )

    result = send(agent, client, cfg)

    assert result.failed_by_tag == {"policy_denied": 1}


# --------------------------------------------------------------------------- parallel


def test_the_agent_runs_a_read_batch_concurrently(cfg_factory, session_dir, patch_openai):
    trace = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(trace))
    write(cfg.work_space / "sandbox" / "a.py", "alpha\n")
    write(cfg.work_space / "sandbox" / "b.py", "beta\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [
            call("read_file", file_path="sandbox/a.py"),
            call("read_file", file_path="sandbox/b.py"),
        ]), 20, 5)),
        Stream(completion(model_message("done"), 30, 5)),
    )

    patch_openai(client)
    result = agent.run_task("read both", cfg=cfg)

    assert result.calls == 2 and result.ok == 2
    batch = next(e for e in traced_events(trace) if e["event"] == "batch_parallel")
    assert batch["tools"] == ["read_file", "read_file"]
    tool_messages = [m for m in agent.message if m["role"] == "tool"]
    assert "alpha" in tool_messages[0]["content"]
    assert "beta" in tool_messages[1]["content"]


def test_a_mixed_batch_is_not_parallelized(cfg_factory, session_dir, patch_openai):
    trace = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(trace))
    write(cfg.work_space / "sandbox" / "a.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [
            call("read_file", file_path="sandbox/a.py"),
            call("write_file", file_path="sandbox/new.py", content="x"),
        ]), 20, 5)),
        Stream(completion(model_message("done"), 30, 5)),
        Stream(completion(model_message("verified enough"), 30, 5)),
    )

    patch_openai(client)
    agent.run_task("go", cfg=cfg)

    assert not [e for e in traced_events(trace) if e["event"] == "batch_parallel"]


# --------------------------------------------------------------------------- trace


def traced_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_run_writes_a_trace_file(cfg_factory, session_dir, patch_openai):
    trace = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(trace))
    write(cfg.work_space / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 10, 5)),
        Stream(completion(model_message("done"), 12, 6)),
    )

    patch_openai(client)
    agent.run_task("read the file", cfg=cfg)

    events = traced_events(trace)
    names = [e["event"] for e in events]
    assert names[0] == "run_start"
    assert names[-1] == "run_end"
    assert names.count("turn") == 2
    assert "tool_call" in names and "tool_result" in names
    assert events[0]["task"] == "read the file"
    assert events[0]["limits"] == "none"
    assert events[-1]["outcome"] == OUTCOME.COMPLETED
    usage = [e for e in events if e["event"] == "usage"]
    assert [e["total"] for e in usage] == [15, 33]
    result = next(e for e in events if e["event"] == "tool_result")
    assert result["tool"] == "read_file" and result["ok"] is True
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


def test_no_trace_file_without_a_path(cfg, session_dir, patch_openai):
    agent, client = make_agent(cfg, Stream(completion(model_message("done"))))

    patch_openai(client)
    agent.run_task("hi", cfg=cfg)

    assert not (session_dir / "trace.jsonl").exists()


def test_a_budget_stop_is_traced(cfg_factory, session_dir, patch_openai):
    trace = session_dir / "trace.jsonl"
    cfg = cfg_factory(token_budget=50, trace_path=str(trace))
    write(cfg.work_space / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]), 60, 0)),
    )

    patch_openai(client)
    agent.run_task("read", cfg=cfg)

    events = traced_events(trace)
    stop = next(e for e in events if e["event"] == "budget_stop")
    assert stop["reason"] == "tokens"
    assert stop["tokens"] == 60
    assert events[-1]["outcome"] == OUTCOME.BUDGET
    assert events[-1]["stopped_by"] == "tokens"


def test_tool_events_name_the_failure_tag(cfg_factory, session_dir, patch_openai):
    trace = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(trace))
    write(cfg.work_space / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [
            call("edit_file", file_path="sandbox/f.py", old_string="alpha", new_string="b"),
        ]), 10, 5)),
        Stream(completion(model_message("done"), 10, 5)),
    )

    patch_openai(client)
    agent.run_task("edit it", cfg=cfg)

    result = next(e for e in traced_events(trace) if e["event"] == "tool_result")
    assert result["ok"] is False
    assert result["tag"] == "need_read"


def test_usage_events_carry_that_turn_s_cached_count(cfg_factory, session_dir, patch_openai):
    """Per-turn events carry a delta; only run_end is cumulative."""
    trace = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(trace))
    write(cfg.work_space / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(
        cfg,
        Stream(completion(model_message("", [call("read_file", file_path="sandbox/f.py")]),
                          100, 5, cached_tokens=40)),
        Stream(completion(model_message("done"), 200, 5, cached_tokens=150)),
    )

    patch_openai(client)
    agent.run_task("read it", cfg=cfg)

    events = traced_events(trace)
    usage = [event for event in events if event["event"] == "usage"]
    assert [event["cached"] for event in usage] == [40, 150]
    assert events[-1]["cached_tokens"] == 190


def test_run_start_records_the_configured_limits(cfg_factory, session_dir, patch_openai):
    trace = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(trace), token_budget=1000, wall_budget=60.0)
    agent, client = make_agent(cfg, Stream(completion(model_message("done"))))

    patch_openai(client)
    agent.run_task("hi", cfg=cfg)

    assert traced_events(trace)[0]["limits"] == "tokens=1000,wall=60.0"


# --------------------------------------------------------------------------- CLI


def test_exit_codes_cover_every_outcome():
    assert cli.EXIT == {
        OUTCOME.COMPLETED: 0, OUTCOME.ERROR: 1, OUTCOME.EXHAUSTED: 3,
        OUTCOME.TIMEOUT: 4, OUTCOME.BUDGET: 5, OUTCOME.INTERRUPTED: 130,
    }


def test_main_without_a_task_enters_the_repl(cfg, monkeypatch, tmp_path):
    entered = []
    monkeypatch.setenv("MINI_HARNESS_WORK_SPACE", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["mini-harness"])
    monkeypatch.setattr(DeepSeekAgent, "run", lambda self, cfg=None: entered.append(True))

    cli.main()

    assert entered == [True]


def test_main_with_a_task_dumps_and_exits(cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_HARNESS_WORK_SPACE", str(tmp_path))
    out = tmp_path / "tele.json"
    monkeypatch.setattr("sys.argv", ["mini-harness", "--task", "hello", "--telemetry-out", str(out)])
    monkeypatch.setattr(DeepSeekAgent, "run_task", lambda self, task, cfg=None: Result(
        OUTCOME.COMPLETED, 0, 1, 0, {}, {}, 1, 1, 1, 0.1))

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["task"] == "hello"


def test_main_maps_a_failed_outcome_to_a_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_HARNESS_WORK_SPACE", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["mini-harness", "--task", "hello"])
    monkeypatch.setattr(DeepSeekAgent, "run_task", lambda self, task, cfg=None: Result(
        OUTCOME.EXHAUSTED, 0, 50, 0, {}, {}, 1, 1, 1, 0.1))

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 3


def test_main_announces_the_budgets_and_trace(cfg_factory, monkeypatch, capsys):
    cfg = cfg_factory(token_budget=5000, wall_budget=60.0, trace_path="trace.jsonl")
    monkeypatch.setattr("sys.argv", ["mini-harness", "--task", "hello"])
    monkeypatch.setattr(DeepSeekAgent, "run_task", lambda self, task, cfg=None: Result(
        OUTCOME.COMPLETED, 0, 1, 0, {}, {}, 1, 1, 1, 0.1))

    with pytest.raises(SystemExit):
        cli.main(cfg=cfg)

    banner = capsys.readouterr().out.splitlines()[0]
    assert banner.startswith("[mini_harness]:")
    assert "budget = tokens=5000,wall=60.0" in banner
    assert "trace = trace.jsonl" in banner


def test_main_reports_an_unconfigured_trace_as_off(cfg, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mini-harness", "--task", "hello"])
    monkeypatch.setattr(DeepSeekAgent, "run_task", lambda self, task, cfg=None: Result(
        OUTCOME.COMPLETED, 0, 1, 0, {}, {}, 1, 1, 1, 0.1))

    with pytest.raises(SystemExit):
        cli.main(cfg=cfg)

    banner = capsys.readouterr().out.splitlines()[0]
    assert "budget = none" in banner
    assert "trace = off" in banner


def test_run_summary_separates_its_fields(cfg, workspace, monkeypatch, capsys):
    """Regression: adjacent literals used to glue '[ok]: 1[calls_tool]:' together."""
    agent, client = make_agent(cfg)
    replies = iter(["do the thing", "quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(replies))
    monkeypatch.setattr("mini_harness.agent.OpenAI", lambda **_: client)
    monkeypatch.setattr(DeepSeekAgent, "_run_turn", lambda self, *a, **k: Result(
        OUTCOME.COMPLETED, 4, 6, 3, {"stale": 1}, {"read_file": 4}, 900, 4200, 700, 12.34))

    agent.run(cfg=cfg)

    summary = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[outcome]")][0]
    assert "[ok]: 3, [calls_tool]" in summary
    assert "[failed]: {'stale': 1}, [last_prompt]: 900" in summary
    assert "[wall]: 12.3s" in summary
    assert "[err]: \n" in summary + "\n"


# --------------------------------------------------------------------------- repl lifetime


def repl(cfg, client, monkeypatch, replies):
    agent, fake = make_agent(cfg)
    answers = iter(replies)
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr("mini_harness.agent.OpenAI", lambda **_: fake)
    return agent


def test_the_repl_closes_the_trace_when_the_user_quits(cfg_factory, session_dir, monkeypatch):
    path = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(path))
    agent = repl(cfg, None, monkeypatch, ["quit"])

    agent.run(cfg=cfg)

    assert TRACE.enabled is False
    assert TRACE._handle is None
    # An open handle keeps the file locked on Windows; deleting it proves the
    # handle is really gone rather than merely flagged as disabled.
    path.unlink()
    assert not path.exists()


def test_the_repl_closes_the_trace_on_end_of_input(cfg_factory, session_dir, monkeypatch):
    path = session_dir / "trace.jsonl"
    cfg = cfg_factory(trace_path=str(path))
    agent = repl(cfg, None, monkeypatch, [])

    def end_of_input(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)
    agent.run(cfg=cfg)

    assert TRACE._handle is None
    path.unlink()
    assert not path.exists()
