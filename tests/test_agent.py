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
from mini_harness.tool.box import TOOLS
from mini_harness.tool.tag import OUTCOME
from tests.conftest import call, executor, write


def model_message(content="", tool_calls=None, reasoning=""):
    return SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning,
        model_dump=lambda **_: {"role": "assistant", "content": content},
    )


def completion(message, prompt_tokens=100, completion_tokens=20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
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


def send(agent, client, cfg, **kwargs):
    agent.message.append({"role": "user", "content": "do the thing"})
    return agent._run_turn(client, executor(cfg), cfg=cfg, **kwargs)


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


def test_interrupted_tool_calls_are_closed_off(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    agent, client = make_agent(cfg)
    agent.message.append({"role": "assistant", "content": "", "tool_calls": []})
    pending = [call("read_file", file_path="sandbox/f.py")]

    agent._fill_interrupted(pending, cfg=cfg)

    filler = [m for m in agent.message if m["role"] == "tool"]
    assert filler[0]["tool_call_id"] == pending[0].id
    assert "intterupted" in filler[0]["content"]


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


# --------------------------------------------------------------------------- CLI


def test_exit_codes_cover_every_outcome():
    assert cli.EXIT == {
        OUTCOME.COMPLETED: 0, OUTCOME.ERROR: 1, OUTCOME.EXHAUSTED: 3,
        OUTCOME.TIMEOUT: 4, OUTCOME.INTERRUPTED: 130,
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
