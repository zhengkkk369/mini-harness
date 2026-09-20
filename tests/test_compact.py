"""Context compaction: cut-point selection, summary injection, audit trail."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini_harness.compact import COMPACT
from mini_harness.trace import TRACE


class FakeClient:
    """Captures the summarisation request and returns a canned summary."""

    def __init__(self, summary="SUMMARY", error=None):
        self.summary = summary
        self.error = error
        self.calls = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.summary)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def __getattr__(self, name):
        if name == "chat":
            return SimpleNamespace(completions=SimpleNamespace(create=self._create))
        raise AttributeError(name)


def conversation(user_turns=3, tool_runs=2):
    """system, then user/assistant/tool blocks."""
    messages = [{"role": "system", "content": "system prompt"}]
    for turn in range(user_turns):
        messages.append({"role": "user", "content": f"question {turn}"})
        for run in range(tool_runs):
            messages.append({
                "role": "assistant",
                "content": f"thinking {turn}.{run}",
                "tool_calls": [{"id": f"c{turn}{run}", "type": "function",
                                "function": {"name": "read_file", "arguments": '{"file_path": "a.py"}'}}],
            })
            messages.append({"role": "tool", "tool_call_id": f"c{turn}{run}", "content": f"result {turn}.{run}"})
        messages.append({"role": "assistant", "content": f"answer {turn}"})
    return messages


# --------------------------------------------------------------------------- cut point


def test_cut_keeps_the_recent_window(cfg_factory):
    messages = conversation(user_turns=6, tool_runs=3)
    cfg = cfg_factory(recent_keep=4)
    # message[-4] is a tool result, so the cut steps back to its assistant call
    assert COMPACT._get_cut(messages, cfg=cfg) == len(messages) - 5


def test_cut_never_lands_on_a_tool_message(cfg_factory):
    messages = conversation(user_turns=6, tool_runs=3)
    cut = COMPACT._get_cut(messages, cfg=cfg_factory(recent_keep=6))
    assert messages[cut]["role"] != "tool"


def test_cut_walks_back_over_a_run_of_tool_messages(cfg_factory):
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "ra"},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
    ]
    cut = COMPACT._get_cut(messages, cfg=cfg_factory(recent_keep=1))
    assert cut == 2


def test_cut_never_goes_below_the_system_message(cfg_factory):
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
    ] + [{"role": "tool", "tool_call_id": "a", "content": "r"} for _ in range(3)]

    cut = COMPACT._get_cut(messages, cfg=cfg_factory(recent_keep=-5))

    assert cut == 2
    assert messages[cut]["role"] != "tool"


def test_cut_is_bounded_by_the_message_list(cfg_factory):
    """Regression: a recent_keep larger than the list indexed past the end."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "a", "content": "r"},
        {"role": "tool", "tool_call_id": "b", "content": "r"},
    ]

    assert COMPACT._get_cut(messages, cfg=cfg_factory(recent_keep=50)) == 1


# --------------------------------------------------------------------------- compaction


def test_compaction_replaces_the_old_prefix_with_one_summary(cfg_factory):
    messages = conversation(user_turns=6, tool_runs=3)
    cfg = cfg_factory(recent_keep=5)
    client = FakeClient(summary="the summary")

    result = COMPACT.compact_content(client, messages, cfg=cfg)

    assert result[0] == messages[0]
    assert result[1]["role"] == "assistant"
    assert "the summary" in result[1]["content"]
    assert result[2:] == messages[len(messages) - 5:]
    assert len(result) < len(messages)


def test_compaction_passes_the_summarisation_prompt_to_the_sub_model(cfg_factory):
    messages = conversation()
    cfg = cfg_factory(recent_keep=5)
    client = FakeClient()

    COMPACT.compact_content(client, messages, cfg=cfg)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == cfg.model_sub
    assert call["stream"] is False
    assert call["messages"][0]["role"] == "system"
    prompt = call["messages"][1]["content"]
    assert "total goal of user" in prompt
    assert "question 0" in prompt
    assert "[tool call]: read_file" in prompt
    assert "[tool result]: result 0.0" in prompt


def test_compaction_prompt_truncates_long_tool_payloads(cfg_factory):
    old = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "a" * 500},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "run_bash", "arguments": "c" * 500}}]},
        {"role": "tool", "tool_call_id": "x", "content": "r" * 500},
    ]
    prompt = COMPACT._get_user(old, cfg=cfg_factory())

    assert "[user]: goal" in prompt
    assert "[assistant]: " + "a" * 500 in prompt
    assert "[tool result]: " + "r" * 200 + "\n" in prompt
    assert "r" * 201 not in prompt
    assert "[tool call]: run_bash: " + "c" * 200 + "\n" in prompt
    assert "c" * 201 not in prompt


def test_a_short_conversation_is_left_untouched(cfg_factory):
    messages = conversation(user_turns=1, tool_runs=1)
    client = FakeClient()

    assert COMPACT.compact_content(client, messages, cfg=cfg_factory(recent_keep=50)) == messages
    assert client.calls == []


def test_a_conversation_that_cannot_be_cut_is_left_untouched(cfg_factory):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    client = FakeClient()

    assert COMPACT.compact_content(client, messages, cfg=cfg_factory(recent_keep=2)) == messages
    assert client.calls == []


def test_removed_messages_are_written_to_the_audit_log(cfg_factory, session_dir):
    messages = conversation(user_turns=6, tool_runs=3)
    session = session_dir / "session.json"
    cfg = cfg_factory(recent_keep=5)

    COMPACT.compact_content(FakeClient(), messages, session, cfg=cfg)

    history = session_dir / "mini_harness_history.jsonl"
    lines = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["removed"] == messages[1:len(messages) - 5]
    assert lines[0]["ts"] > 0


def test_repeated_compaction_appends_to_the_same_log(cfg_factory, session_dir):
    session = session_dir / "session.json"
    cfg = cfg_factory(recent_keep=5)

    first = COMPACT.compact_content(FakeClient(), conversation(user_turns=6, tool_runs=3), session, cfg=cfg)
    COMPACT.compact_content(FakeClient(), conversation(user_turns=6, tool_runs=3), session, cfg=cfg)
    assert len(first) < len(conversation(user_turns=6, tool_runs=3))

    history = session_dir / "mini_harness_history.jsonl"
    assert len(history.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert first[0]["role"] == "system"


def test_compaction_failure_returns_the_original_messages(cfg_factory, capsys):
    messages = conversation(user_turns=6, tool_runs=3)
    cfg = cfg_factory(recent_keep=5)
    client = FakeClient(error=RuntimeError("summariser down"))

    result = COMPACT.compact_content(client, messages, cfg=cfg)

    assert result == messages
    assert "compact failed" in capsys.readouterr().out


def test_compaction_reports_and_survives_an_unwritable_history(cfg_factory, session_dir):
    messages = conversation(user_turns=6, tool_runs=3)
    cfg = cfg_factory(recent_keep=5)
    blocked = session_dir / "mini_harness_history.jsonl"
    blocked.mkdir()

    result = COMPACT.compact_content(FakeClient(), messages, session_dir / "session.json", cfg=cfg)

    assert "the summary" in result[1]["content"]


# --------------------------------------------------------------------------- trace


def compact_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_compaction_is_traced(cfg_factory, session_dir, tmp_path):
    path = tmp_path / "trace.jsonl"
    messages = conversation(user_turns=6, tool_runs=3)
    TRACE.configure(path)

    COMPACT.compact_content(FakeClient(), messages, session_dir / "session.json",
                            cfg=cfg_factory(recent_keep=5))
    TRACE.configure(None)

    event = compact_events(path)[0]
    assert event["event"] == "compact"
    assert event["ok"] is True
    assert event["kept"] >= 5
    # the system message is never removed, so the parts sum to one less
    assert event["removed"] + event["kept"] == len(messages) - 1


def test_a_failed_compaction_is_traced(cfg_factory, session_dir, tmp_path):
    path = tmp_path / "trace.jsonl"
    TRACE.configure(path)

    COMPACT.compact_content(FakeClient(error=RuntimeError("down")),
                            conversation(user_turns=6, tool_runs=3),
                            session_dir / "session.json", cfg=cfg_factory(recent_keep=5))
    TRACE.configure(None)

    event = compact_events(path)[0]
    assert event["event"] == "compact"
    assert event["ok"] is False
    assert event["error"] == "RuntimeError"
