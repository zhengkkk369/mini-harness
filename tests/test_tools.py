"""Behaviour of the nine file and shell tools, exercised offline."""

import os
import re
import sys

from dataclasses import replace

import pytest
from pydantic import ValidationError

from mini_harness.tool import box
from mini_harness.tool.block import CLIP, TODO
from tests.conftest import write


# --------------------------------------------------------------------------- glob_file


def test_glob_without_slash_searches_recursively(cfg, workspace):
    write(workspace / "a.py", "x = 1\n")
    write(workspace / "deep" / "b.py", "y = 2\n")
    write(workspace / "deep" / "c.txt", "z\n")

    found = box.glob_file(box.GlobFileInput(pattern="*.py"), cfg=cfg)

    assert "a.py" in found
    assert str(workspace / "deep" / "b.py") in found
    assert "c.txt" not in found


def test_glob_with_slash_stays_in_directory(cfg, workspace):
    write(workspace / "src" / "a.py", "x = 1\n")
    write(workspace / "other" / "b.py", "y = 2\n")

    found = box.glob_file(box.GlobFileInput(pattern="src/*.py"), cfg=cfg)

    assert "a.py" in found
    assert "b.py" not in found


def test_glob_reports_no_matches(cfg, workspace):
    assert box.glob_file(box.GlobFileInput(pattern="*.rs"), cfg=cfg) == "no matches"


def test_glob_skips_sensitive_names(cfg, workspace):
    write(workspace / "keep.py", "x = 1\n")
    write(workspace / ".env", "DEEPSEEK_API_KEY=leak\n")

    found = box.glob_file(box.GlobFileInput(pattern="*"), cfg=cfg)

    assert "keep.py" in found
    assert ".env" not in found


def test_glob_sorts_by_modification_time(cfg, workspace):
    old = write(workspace / "old.py", "old\n")
    new = write(workspace / "new.py", "new\n")
    os.utime(old, (1_000_000, 1_000_000))

    lines = box.glob_file(box.GlobFileInput(pattern="*.py"), cfg=cfg).splitlines()

    assert lines[0].endswith("new.py")
    assert lines[1].endswith("old.py")


# --------------------------------------------------------------------------- grep_file


def test_grep_matches_with_path_and_line_number(cfg, workspace):
    write(workspace / "pkg" / "m.py", "import os\nvalue = 1\ntarget = 2\n")

    hits = box.grep_file(box.GrepFileInput(pattern="target", path="pkg", glob="*.py"), cfg=cfg)

    assert re.fullmatch(r"\[.+\]: 3: target = 2", hits)


def test_grep_can_target_a_single_file(cfg, workspace):
    target = write(workspace / "one.py", "needle\n")
    hits = box.grep_file(box.GrepFileInput(pattern="needle", path="one.py"), cfg=cfg)
    assert hits == f"[{target.resolve()}]: 1: needle"


def test_grep_reports_no_matches(cfg, workspace):
    write(workspace / "one.py", "x\n")
    assert box.grep_file(box.GrepFileInput(pattern="zzz", path="."), cfg=cfg) == "no matches"


def test_grep_skips_denied_files_and_hidden_dirs(cfg, workspace):
    write(workspace / "visible.py", "secret_token\n")
    write(workspace / ".env", "secret_token\n")
    write(workspace / ".hidden" / "deep.py", "secret_token\n")

    hits = box.grep_file(box.GrepFileInput(pattern="secret_token"), cfg=cfg)

    assert "visible.py" in hits
    assert ".env" not in hits
    assert ".hidden" not in hits


def test_grep_truncates_at_max_hits(cfg_factory, workspace):
    write(workspace / "many.txt", "".join(f"hit {i}\n" for i in range(30)))

    hits = box.grep_file(
        box.GrepFileInput(pattern="hit", path="many.txt"), cfg=cfg_factory(max_hits=5)
    )

    assert len(hits.splitlines()) == 6
    assert "grep truncated at 5 hits" in hits


def test_grep_reports_truncation_within_a_single_file(cfg_factory, workspace):
    """Regression: the notice was only appended when another file followed."""
    write(workspace / "only.txt", "".join(f"hit {i}\n" for i in range(30)))

    hits = box.grep_file(box.GrepFileInput(pattern="hit", path="only.txt"), cfg=cfg_factory(max_hits=2))

    assert "truncated at 2 hits" in hits
    assert len(hits.splitlines()) == 3


def test_grep_rejects_an_invalid_regex(cfg):
    with pytest.raises(ValidationError, match="invalid regex"):
        box.GrepFileInput(pattern="[unclosed")


def test_grep_honours_the_glob_filter(cfg, workspace):
    write(workspace / "a.py", "needle\n")
    write(workspace / "a.txt", "needle\n")

    hits = box.grep_file(box.GrepFileInput(pattern="needle", glob="*.txt"), cfg=cfg)

    assert "a.txt" in hits
    assert "a.py" not in hits


# --------------------------------------------------------------------------- read_file


def test_read_renders_cat_n_style_line_numbers(cfg, workspace):
    write(workspace / "f.py", "first\nsecond\n")

    body = box.read_file(box.ReadFileInput(file_path="f.py"), cfg=cfg)

    assert body.splitlines() == ["     1\tfirst", "     2\tsecond"]


def test_read_window_reports_a_continuation_hint(cfg, workspace):
    write(workspace / "f.py", "".join(f"line{i}\n" for i in range(1, 11)))

    body = box.read_file(box.ReadFileInput(file_path="f.py", offset=3, limit=2), cfg=cfg)

    assert body.splitlines()[0] == "     3\tline3"
    assert "[Showing lines 3-4 of 10, use offset 5 to continue]" in body


def test_read_to_the_end_has_no_hint(cfg, workspace):
    write(workspace / "f.py", "a\nb\n")
    body = box.read_file(box.ReadFileInput(file_path="f.py", offset=2), cfg=cfg)
    assert body.splitlines() == ["     2\tb"]


def test_read_offset_beyond_end_is_an_error(cfg, workspace):
    write(workspace / "f.py", "a\n")
    with pytest.raises(ValueError, match="offset error"):
        box.read_file(box.ReadFileInput(file_path="f.py", offset=5), cfg=cfg)


def test_read_of_a_missing_file_is_an_error(cfg, workspace):
    with pytest.raises(FileNotFoundError):
        box.read_file(box.ReadFileInput(file_path="nope.py"), cfg=cfg)


def test_blind_read_refuses_an_oversized_file(cfg_factory, workspace):
    write(workspace / "big.txt", "x" * 5000)
    with pytest.raises(ValueError, match="oversize"):
        box.read_file(box.ReadFileInput(file_path="big.txt"), cfg=cfg_factory(max_read_size=1000))


def test_windowed_read_is_exempt_from_the_size_cap(cfg_factory, workspace):
    write(workspace / "big.txt", "".join(f"{i}\n" for i in range(2000)))
    body = box.read_file(
        box.ReadFileInput(file_path="big.txt", offset=1, limit=1), cfg=cfg_factory(max_read_size=10)
    )
    assert body.splitlines()[0] == "     1\t0"
    assert "use offset 2 to continue" in body


def test_read_of_binary_content_is_flagged(cfg, workspace):
    (workspace / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")

    body = box.read_file(box.ReadFileInput(file_path="blob.bin"), cfg=cfg)

    assert "Binary or non-UTF-8" in body


def test_read_of_an_empty_file_is_flagged(cfg, workspace):
    write(workspace / "empty.txt", "")
    assert "file is empty" in box.read_file(box.ReadFileInput(file_path="empty.txt"), cfg=cfg)


def test_read_normalises_crlf_line_endings(cfg, workspace):
    (workspace / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    body = box.read_file(box.ReadFileInput(file_path="crlf.txt"), cfg=cfg)
    assert body.splitlines() == ["     1\ta", "     2\tb"]


def test_read_output_is_clipped_at_read_limit(cfg_factory, workspace):
    write(workspace / "wide.txt", "".join(f"line{i}\n" for i in range(500)))
    body = box.read_file(box.ReadFileInput(file_path="wide.txt"), cfg=cfg_factory(read_limit=200))
    assert len(body) < 400
    assert "use offset" in body


# --------------------------------------------------------------------------- write_file


def test_write_creates_the_file_and_reports_size(cfg, workspace):
    message = box.write_file(box.WriteFileInput(file_path="sandbox/new.py", content="a\nb\n"), cfg=cfg)

    assert message == f"Created {(workspace / 'sandbox' / 'new.py').resolve()} (2 lines, 4 chars)"
    assert (workspace / "sandbox" / "new.py").read_text(encoding="utf-8") == "a\nb\n"


def test_write_creates_missing_parent_directories(cfg, workspace):
    box.write_file(box.WriteFileInput(file_path="sandbox/deep/nested/x.py", content="x\n"), cfg=cfg)
    assert (workspace / "sandbox" / "deep" / "nested" / "x.py").is_file()


def test_write_leaves_no_temporary_file_behind(cfg, workspace):
    box.write_file(box.WriteFileInput(file_path="sandbox/one.py", content="x\n"), cfg=cfg)
    assert [p.name for p in (workspace / "sandbox").iterdir()] == ["one.py"]


def test_write_reports_overwrite_for_an_existing_file(cfg, workspace):
    write(workspace / "sandbox" / "one.py", "old\n")
    message = box.write_file(box.WriteFileInput(file_path="sandbox/one.py", content="new\n"), cfg=cfg)
    assert message.startswith("Overwrote")


def test_write_outside_sandbox_is_refused(cfg, workspace):
    with pytest.raises(PermissionError):
        box.write_file(box.WriteFileInput(file_path="outside.py", content="x\n"), cfg=cfg)


# --------------------------------------------------------------------------- edit_file


def test_edit_replaces_and_echoes_real_line_numbers(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\nbeta\ngamma\n")

    message = box.edit_file(
        box.EditFileInput(file_path="sandbox/f.py", old_string="beta", new_string="BETA"), cfg=cfg
    )

    assert "Replaced 1 occurrence(s)" in message
    assert "(1 -> 1 lines, file now 3 lines)" in message
    assert "     2\tBETA" in message
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_edit_missing_old_string_is_refused(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    with pytest.raises(ValueError, match="old content is not in"):
        box.edit_file(
            box.EditFileInput(file_path="sandbox/f.py", old_string="zzz", new_string="y"), cfg=cfg
        )


def test_edit_ambiguous_old_string_is_refused(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "dup\ndup\n")
    with pytest.raises(ValueError, match="appears in the .* 2 times"):
        box.edit_file(
            box.EditFileInput(file_path="sandbox/f.py", old_string="dup", new_string="one"), cfg=cfg
        )


def test_edit_replace_all_rewrites_every_occurrence(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "dup\ndup\n")

    message = box.edit_file(
        box.EditFileInput(file_path="sandbox/f.py", old_string="dup", new_string="one", replace_all=True),
        cfg=cfg,
    )

    assert "Replaced 2 occurrence(s)" in message
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "one\none\n"


def test_edit_of_a_missing_file_is_an_error(cfg, workspace):
    with pytest.raises(FileNotFoundError):
        box.edit_file(
            box.EditFileInput(file_path="sandbox/none.py", old_string="a", new_string="b"), cfg=cfg
        )


def test_edit_strips_line_number_prefixes_from_old_string(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\nbeta\n")

    message = box.edit_file(
        box.EditFileInput(file_path="sandbox/f.py", old_string="     2\tbeta", new_string="BETA"),
        cfg=cfg,
    )

    assert "Replaced 1 occurrence(s)" in message
    assert (workspace / "sandbox" / "f.py").read_text(encoding="utf-8") == "alpha\nBETA\n"


def test_edit_refuses_line_numbers_in_both_strings(cfg, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\nbeta\n")
    with pytest.raises(ValueError, match="line number"):
        box.edit_file(
            box.EditFileInput(file_path="sandbox/f.py", old_string="     2\tbeta", new_string="     2\tBETA"),
            cfg=cfg,
        )


def test_edit_refuses_identical_strings(cfg):
    with pytest.raises(ValidationError, match="equal to the new content"):
        box.EditFileInput(file_path="sandbox/f.py", old_string="same", new_string="same")


def test_edit_refuses_non_utf8_files(cfg, workspace):
    (workspace / "sandbox" / "blob.py").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        box.edit_file(
            box.EditFileInput(file_path="sandbox/blob.py", old_string="a", new_string="b"), cfg=cfg
        )


def test_edit_diff_echo_can_be_disabled(cfg_factory, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    message = box.edit_file(
        box.EditFileInput(file_path="sandbox/f.py", old_string="alpha", new_string="beta"),
        cfg=cfg_factory(diff_echo_lines=0),
    )
    assert message.count("\n") == 1


def test_edit_truncates_a_long_diff_echo(cfg_factory, workspace):
    write(workspace / "sandbox" / "f.py", "alpha\n")
    message = box.edit_file(
        box.EditFileInput(file_path="sandbox/f.py", old_string="alpha", new_string="\n".join("b" for _ in range(60))),
        cfg=cfg_factory(diff_echo_lines=5),
    )
    assert "[showing first 5 of 60 changed lines]" in message


# --------------------------------------------------------------------------- run_bash
#
# These exercise the wrapper by recording the subprocess call instead of
# spawning a real child: capturing a child's output requires pipes, which not
# every environment permits. The one real execution below uses no pipes.


class FakeRun:
    def __init__(self, stdout="", stderr="", returncode=0, error=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.error = error
        self.command = None
        self.options = {}

    def __call__(self, command, **options):
        self.command = command
        self.options = options
        if self.error is not None:
            raise self.error
        return box.subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def capture_run(monkeypatch):
    def install(**kwargs):
        fake = FakeRun(**kwargs)
        monkeypatch.setattr(box.subprocess, "run", fake)
        return fake

    return install


def test_run_bash_returns_stdout(cfg, workspace, capture_run):
    fake = capture_run(stdout="42\n")

    assert box.run_bash(box.RunBashInput(command="python -c 1"), cfg=cfg) == "42\n"
    assert fake.command == "python -c 1"


def test_run_bash_labels_stderr(cfg, capture_run):
    capture_run(stderr="boom\n")
    assert box.run_bash(box.RunBashInput(command="x"), cfg=cfg) == "[stderr]: boom\n"


def test_run_bash_reports_a_nonzero_exit_code(cfg, capture_run):
    capture_run(stdout="partial", returncode=3)
    text = box.run_bash(box.RunBashInput(command="x"), cfg=cfg)
    assert text.endswith("[exit code: 3]")


def test_run_bash_reports_no_output(cfg, capture_run):
    capture_run()
    assert box.run_bash(box.RunBashInput(command="true"), cfg=cfg) == "no result"


def test_run_bash_runs_in_a_shell_at_the_workspace_root(cfg, workspace, capture_run):
    fake = capture_run()
    cfg = replace(cfg, bash_timeout=17)

    box.run_bash(box.RunBashInput(command="x"), cfg=cfg)

    assert fake.options["shell"] is True
    assert fake.options["capture_output"] is True
    assert fake.options["text"] is True
    assert fake.options["timeout"] == 17
    assert fake.options["cwd"] == workspace
    assert fake.options["env"] == cfg.bash_env


def test_run_bash_reports_a_timeout(cfg, capture_run):
    capture_run(error=box.subprocess.TimeoutExpired("x", 90))

    with pytest.raises(TimeoutError, match="time out of"):
        box.run_bash(box.RunBashInput(command="x"), cfg=cfg)


def test_run_bash_truncates_large_output(cfg, capture_run):
    capture_run(stdout="x" * 5000)

    text = box.run_bash(box.RunBashInput(command="x"), cfg=replace(cfg, bash_limit=500))

    assert text.startswith("x" * 500)
    assert "truncated at 500 of 5000" in text


def test_run_bash_strips_secret_environment_variables(cfg):
    os.environ["SOME_SERVICE_TOKEN"] = "leak-me"
    try:
        assert "SOME_SERVICE_TOKEN" not in cfg.bash_env
    finally:
        del os.environ["SOME_SERVICE_TOKEN"]


def test_bash_env_keeps_ordinary_variables(cfg):
    os.environ["MINI_HARNESS_TEST_MARKER"] = "kept"
    try:
        assert cfg.bash_env.get("MINI_HARNESS_TEST_MARKER") == "kept"
    finally:
        del os.environ["MINI_HARNESS_TEST_MARKER"]


# --------------------------------------------------------------------------- run_todo


def test_todo_renders_each_status(cfg):
    text = box.run_todo(box.RunTodoInput(items=[
        {"content": "read code", "activeForm": "reading code", "status": "completed"},
        {"content": "fix bug", "activeForm": "fixing bug", "status": "in_progress"},
        {"content": "run tests", "activeForm": "running tests", "status": "pending"},
    ]), cfg=cfg)

    assert text == "[x] read code\n[>] fixing bug\n[ ] run tests\n1 / 3 completed"


def test_todo_allows_only_one_in_progress_item(cfg):
    with pytest.raises(ValidationError, match="only one thing can be in progress"):
        box.RunTodoInput(items=[
            {"content": "a", "activeForm": "a", "status": "in_progress"},
            {"content": "b", "activeForm": "b", "status": "in_progress"},
        ])


def test_todo_rejects_an_empty_list(cfg):
    with pytest.raises(ValidationError):
        box.RunTodoInput(items=[])


def test_todo_state_is_shared_with_the_manager(cfg):
    box.run_todo(box.RunTodoInput(items=[
        {"content": "one", "activeForm": "one", "status": "completed"},
    ]), cfg=cfg)
    assert TODO.render() == "[x] one\n1 / 1 completed"


# --------------------------------------------------------------------------- limits


def test_clip_leaves_short_content_alone(cfg):
    assert CLIP.clip("short", cfg=cfg) == "short"


def test_clip_truncates_long_content(cfg_factory):
    cfg = cfg_factory(clip_limit=10)
    text = CLIP.clip("x" * 50, cfg=cfg)
    assert text.startswith("x" * 10)
    assert "clipped at 10 of 50 chars" in text
