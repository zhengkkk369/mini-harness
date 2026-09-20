"""Config construction, provider inference, and the bench profile."""

import os
from pathlib import Path

import pytest

from mini_harness.bench_profile import BENCH_OVERRIDE, SESSION_NAME, _log_dir, session_path
from mini_harness.config import Config, build_config


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for name in (
        "MINI_HARNESS_MODEL", "MINI_HARNESS_SUB_MODEL", "MINI_HARNESS_PROVIDER",
        "MINI_HARNESS_BASE_URL", "MINI_HARNESS_API_KEY_ENV", "MINI_HARNESS_PROFILE",
        "MINI_HARNESS_LOG_DIR", "MINI_HARNESS_WORK_SPACE", "MINI_HARNESS_REASONING_EFFORT",
        "MINI_HARNESS_MAX_TOKENS_MAIN", "MINI_HARNESS_MAX_TOKENS_SUB", "MINI_HARNESS_COMPACT_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- defaults


def test_default_profile_is_local_deepseek():
    cfg = build_config()
    assert cfg.profile == "local"
    assert cfg.provider == "deepseek"
    assert cfg.model_main == "deepseek-v4-flash"
    assert cfg.api_key_env == "DEEPSEEK_API_KEY"
    assert cfg.guard_read and cfg.guard_write


def test_config_is_frozen():
    cfg = build_config()
    with pytest.raises(Exception):
        cfg.model_main = "something-else"


def test_api_key_falls_back_to_the_generic_variable(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_API_KEY", "generic-key")
    assert build_config().api_key == "generic-key"


def test_empty_model_is_rejected(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_MODEL", "   ")
    with pytest.raises(ValueError, match="must not be empty"):
        build_config()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_config()


def test_model_prefix_selects_the_provider(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_MODEL", "openai/gpt-4.1")
    cfg = build_config()
    assert cfg.provider == "openai"
    assert cfg.model_main == "gpt-4.1"
    assert cfg.api_key_env == "OPENAI_API_KEY"


def test_model_prefix_conflicting_with_the_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_MODEL", "openai/gpt-4.1")
    monkeypatch.setenv("MINI_HARNESS_PROVIDER", "deepseek")
    with pytest.raises(ValueError, match="conflicts"):
        build_config()


def test_bare_model_name_keeps_the_deepseek_provider(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_MODEL", "deepseek-v4-flash")
    cfg = build_config()
    assert cfg.provider == "deepseek"
    assert cfg.model_main == "deepseek-v4-flash"


def test_non_deepseek_bare_name_switches_to_openai(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_MODEL", "gpt-4.1-mini")
    assert build_config().provider == "openai"


def test_sub_model_must_match_the_provider(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_SUB_MODEL", "openai/gpt-4.1")
    with pytest.raises(ValueError, match="Sub-model"):
        build_config()


def test_sub_model_prefix_is_stripped(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_SUB_MODEL", "deepseek/deepseek-v4-flash")
    assert build_config().model_sub == "deepseek-v4-flash"


def test_openai_defaults_are_applied(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_PROVIDER", "openai")
    cfg = build_config()
    assert (cfg.max_tokens_main, cfg.max_tokens_sub, cfg.compact_limit) == (8192, 8192, 64000)
    assert cfg.think_main == "default"


def test_token_overrides_must_be_positive(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_MAX_TOKENS_MAIN", "0")
    with pytest.raises(ValueError, match="must be positive"):
        build_config()


@pytest.mark.parametrize("variable,field", [
    ("MINI_HARNESS_MAX_TOKENS_MAIN", "max_tokens_main"),
    ("MINI_HARNESS_MAX_TOKENS_SUB", "max_tokens_sub"),
    ("MINI_HARNESS_COMPACT_LIMIT", "compact_limit"),
])
def test_token_overrides_are_read_from_the_environment(monkeypatch, variable, field):
    monkeypatch.setenv(variable, "1234")
    assert getattr(build_config(), field) == 1234


def test_workspace_defaults_to_the_working_directory():
    assert Config().work_space == Path.cwd()


def test_workspace_can_be_pointed_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_HARNESS_WORK_SPACE", str(tmp_path))
    assert build_config().work_space == tmp_path.resolve()


# --------------------------------------------------------------------------- request shape


def test_deepseek_request_options_request_thinking():
    cfg = build_config()
    options = cfg.request_options()
    assert options["model"] == cfg.model_main
    assert options["max_tokens"] == cfg.max_tokens_main
    assert options["extra_body"] == {"thinking": {"type": "enabled"}}


def test_subagent_options_use_the_sub_model(cfg_factory):
    cfg = build_config()
    assert cfg.request_options(sub=True)["model"] == cfg.model_sub


def test_openai_request_options_use_completion_tokens(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_PROVIDER", "openai")
    cfg = build_config()
    options = cfg.request_options()
    assert options["max_completion_tokens"] == cfg.max_tokens_main
    assert "extra_body" not in options


def test_openai_messages_drop_reasoning_content(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_PROVIDER", "openai")
    cfg = build_config()
    cleaned = cfg.request_messages([
        {"role": "assistant", "content": "hi", "reasoning_content": "why", "annotations": []},
    ])
    assert cleaned == [{"role": "assistant", "content": "hi"}]


def test_deepseek_messages_are_passed_through_unchanged():
    cfg = build_config()
    messages = [{"role": "assistant", "content": "hi", "reasoning_content": "why"}]
    assert cfg.request_messages(messages) is messages


def test_bash_env_always_drops_api_keys(cfg_factory):
    os.environ["MINI_HARNESS_API_KEY"] = "leak"
    try:
        assert "MINI_HARNESS_API_KEY" not in cfg_factory().bash_env
    finally:
        del os.environ["MINI_HARNESS_API_KEY"]


# --------------------------------------------------------------------------- bench profile


def test_bench_override_disables_the_sandbox_and_raises_limits():
    assert BENCH_OVERRIDE["profile"] == "bench"
    assert BENCH_OVERRIDE["guard_read"] is False
    assert BENCH_OVERRIDE["guard_write"] is False
    assert BENCH_OVERRIDE["max_turns_main"] == 300
    assert BENCH_OVERRIDE["bash_timeout"] == 300
    assert BENCH_OVERRIDE["deny_name"] == ()


def test_bench_profile_activates_from_the_environment(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_PROFILE", "bench")
    cfg = build_config()
    assert cfg.profile == "bench"
    assert not cfg.guard_read and not cfg.guard_write
    assert cfg.system_prompt == BENCH_OVERRIDE["system_prompt"]


def test_bench_session_path_defaults_to_the_container_log_dir():
    assert BENCH_OVERRIDE["session_path"] == "/logs/agent/mini_harness_session.json"


def test_bench_session_name_matches_the_trajectory_builder():
    """bench/atif.py reads this exact file name from the agent log dir."""
    assert SESSION_NAME == "mini_harness_session.json"
    assert Path("bench/atif.py").read_text(encoding="utf-8").count(f'"{SESSION_NAME}"') >= 1


def test_bench_log_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_HARNESS_LOG_DIR", str(tmp_path))
    assert _log_dir() == str(tmp_path)
    assert session_path() == f"{tmp_path}/mini_harness_session.json"


def test_bench_log_dir_tolerates_a_trailing_separator(monkeypatch):
    monkeypatch.setenv("MINI_HARNESS_LOG_DIR", "/logs/agent/")
    assert _log_dir() == "/logs/agent/"
    assert session_path() == "/logs/agent/mini_harness_session.json"


def test_bench_session_path_is_writable_on_this_host(monkeypatch, cfg_factory):
    """Regression: the bench profile used to hardcode /logs/agent and fail off Linux."""
    logs = cfg_factory.workspace.parent
    monkeypatch.setenv("MINI_HARNESS_PROFILE", "bench")
    monkeypatch.setenv("MINI_HARNESS_LOG_DIR", str(logs))
    # The module-level override is built at import; the test replaces it with
    # the one build_config() would produce now that the environment is set.
    import mini_harness.config as config_module
    import mini_harness.bench_profile as bench_profile

    monkeypatch.setattr(config_module, "BENCH_OVERRIDE", dict(bench_profile.BENCH_OVERRIDE,
                                                              session_path=bench_profile.session_path()))

    cfg = build_config()

    assert cfg.session_path == f"{logs}/mini_harness_session.json"
    assert not cfg.session_path.startswith("/logs")


def test_bench_prompt_forbids_asking_questions():
    assert "unattended" in BENCH_OVERRIDE["system_prompt"]
